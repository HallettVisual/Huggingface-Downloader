"""A Tk desktop app for downloading Hugging Face models at full Xet speed.

The transfer runs in a worker process launched with an interpreter that owns a
modern ``huggingface_hub`` (plus ``hf_xet``) - normally the venv behind the ``hf``
CLI rather than the interpreter running this GUI. That keeps the GUI launchable
from any Python while still getting Xet speed, and lets it read exact byte counts
through huggingface_hub's ``tqdm_class`` hook instead of scraping console output.

The worker reports progress back as JSON lines; see ``worker_main``.
Running ``python hf_downloader.py --worker`` puts this file into that worker mode.
"""

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote

APP_TITLE = "Hugging Face Model Downloader"

WORKER_FLAG = "--worker"

# huggingface_hub gained Xet support in 0.30; below that the worker is pointless.
MIN_HUB_VERSION = (0, 30)

# Subfolders that mark a directory as a ComfyUI "models" root.
COMFY_MODEL_DIRS = (
    "checkpoints", "diffusion_models", "loras", "vae", "unet",
    "clip", "controlnet", "text_encoders", "embeddings",
)


def config_path():
    """Per-user settings file, kept outside the checkout so the repo stays clean."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "hf-model-downloader" / "settings.json"


def find_comfy_models_dir():
    """Best guess at a ComfyUI models folder on this machine, or None."""
    here = Path(__file__).resolve()
    roots = [Path.cwd(), *here.parents[:4], Path.home()]
    for root in roots:
        for candidate in (root / "models", root / "ComfyUI" / "models"):
            try:
                if candidate.is_dir() and any((candidate / d).is_dir() for d in COMFY_MODEL_DIRS):
                    return str(candidate)
            except OSError:
                continue
    return None


def find_hf_cli():
    """Locate the `hf` command line tool, or return an empty string."""
    found = shutil.which("hf")
    if found:
        return found
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    names = ("hf.exe", "hf") if os.name == "nt" else ("hf",)
    for folder in (Path.home() / ".local" / "bin",
                   Path.home() / ".hf-cli" / "venv" / bin_dir):
        for name in names:
            candidate = folder / name
            if candidate.exists():
                return str(candidate)
    return ""


def open_in_file_manager(path):
    """Reveal a folder in the platform's file manager."""
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - Windows-only API
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(n)
    for unit in units:
        if n < 1024 or unit == units[-1]:
            return f"{n:.1f} {unit}"
        n /= 1024


def human_time(seconds):
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# --------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------

def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def worker_main():
    """Run one download job, reporting progress as JSON lines on stdout."""
    from tqdm.auto import tqdm as base_tqdm
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub import errors as hub_errors

    job = json.loads(sys.stdin.read() or "{}")
    repo = job["repo"]
    filename = (job.get("file") or "").strip()
    dest = Path(job["dest"])
    revision = job.get("revision") or None
    repo_type = job.get("repo_type") or "model"
    flatten = bool(job.get("flatten", True))
    max_workers = int(job.get("max_workers") or 8)

    state = {"disk": 0, "net": 0, "total": 0}
    total_known = False
    lock = threading.Lock()
    last_push = [0.0]
    devnull = open(os.devnull, "w")

    def push(force=False):
        now = time.monotonic()
        if not force and now - last_push[0] < 0.12:
            return
        last_push[0] = now
        with lock:
            _emit({"t": "p", **state})

    class Reporting(base_tqdm):
        """Counts bytes for the GUI instead of drawing to a terminal.

        Defining ``update_transfer`` is what makes huggingface_hub fold Xet's
        network and reconstruction counters into this single bar rather than
        creating a second one, which would double the reported total.
        """

        def __init__(self, *args, **kwargs):
            # snapshot_download builds two bars of this class and drives both through
            # update(): one counts bytes written to disk, the other network bytes.
            # Only the disk bar has a meaningful denominator, so it is the one whose
            # format string carries a total.
            bar_format = kwargs.get("bar_format") or ""
            self._is_transfer = bool(bar_format) and "total_fmt" not in bar_format
            kwargs["file"] = devnull
            kwargs["disable"] = False  # disable=True makes tqdm.update() skip counting
            super().__init__(*args, **kwargs)
            self._count(getattr(self, "n", 0) or 0)

        def _count(self, n):
            """Add n to whichever tally this bar represents."""
            if not n:
                return
            with lock:
                if self._is_transfer:
                    state["net"] += n
                else:
                    state["disk"] += n
                    if not total_known:
                        state["total"] = max(state["total"], self.total or 0)

        def display(self, *args, **kwargs):
            return True

        def refresh(self, *args, **kwargs):
            return True

        def set_transfer_postfix_str(self, *args, **kwargs):
            return None

        def update(self, n=1):
            out = super().update(n)
            self._count(n or 0)
            push()
            return out

        def update_transfer(self, n=1):
            with lock:
                state["net"] += n or 0
            push()

        def close(self):
            push(force=True)
            return super().close()

    api = HfApi()

    # Preflight: a known total keeps the progress bar honest from the first byte.
    try:
        if filename:
            info = api.get_paths_info(repo, [filename], revision=revision, repo_type=repo_type)
            if not info:
                _emit({
                    "t": "error",
                    "kind": "entry",
                    "msg": f"'{filename}' is not in {repo}.",
                    "hint": "Check the file path inside the repository.",
                })
                return 1
            sizes = [getattr(item, "size", None) or 0 for item in info]
            if any(sizes):
                state["total"] = sum(sizes)
                total_known = True
        else:
            info = api.repo_info(repo, revision=revision, repo_type=repo_type, files_metadata=True)
            sizes = [sibling.size or 0 for sibling in (info.siblings or [])]
            if any(sizes):
                state["total"] = sum(sizes)
                total_known = True
    except Exception as exc:
        # The real cause is reported properly by the download below; keep this short.
        summary = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        _emit({"t": "log", "msg": f"Could not read size ahead of time: {type(exc).__name__} {summary}"[:160]})

    _emit({"t": "meta", "total": state["total"]})

    # Skip work that is already done.
    if filename and total_known:
        target = dest / (Path(filename).name if flatten else filename)
        try:
            if target.exists() and target.stat().st_size == state["total"]:
                _emit({"t": "done", "path": str(target), "skipped": True})
                return 0
        except OSError:
            pass

    try:
        if filename:
            got = Path(hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(dest),
                tqdm_class=Reporting,
            ))
            if flatten and got.parent != dest:
                final = dest / got.name
                if final.exists():
                    final.unlink()
                shutil.move(str(got), str(final))
                _prune_empty(got.parent, dest)
                got = final
        else:
            got = Path(snapshot_download(
                repo_id=repo,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(dest),
                max_workers=max_workers,
                tqdm_class=Reporting,
            ))

        push(force=True)
        _emit({"t": "done", "path": str(got)})
        return 0

    except hub_errors.GatedRepoError:
        _emit({
            "t": "error",
            "kind": "gated",
            "msg": f"{repo} is gated and this account has not been granted access.",
            "hint": f"Accept the licence at https://huggingface.co/{repo} then try again.",
        })
    except hub_errors.RepositoryNotFoundError:
        _emit({
            "t": "error",
            "kind": "notfound",
            "msg": f"Repository {repo} was not found.",
            "hint": "Check the name, or run 'hf auth login' if it is private.",
        })
    except hub_errors.RevisionNotFoundError:
        _emit({
            "t": "error",
            "kind": "revision",
            "msg": f"Revision '{revision}' does not exist in {repo}.",
            "hint": "Leave the revision blank to use the default branch.",
        })
    except hub_errors.EntryNotFoundError:
        _emit({
            "t": "error",
            "kind": "entry",
            "msg": f"'{filename}' is not in {repo}.",
            "hint": "Check the file path inside the repository.",
        })
    except OSError as exc:
        out_of_space = getattr(exc, "errno", None) == 28 or "space" in str(exc).lower()
        _emit({
            "t": "error",
            "kind": "disk" if out_of_space else "os",
            "msg": f"{type(exc).__name__}: {exc}",
            "hint": "Free up space on the destination drive." if out_of_space else "",
        })
    except Exception as exc:
        _emit({"t": "error", "kind": "other", "msg": f"{type(exc).__name__}: {exc}", "hint": ""})

    return 1


def _prune_empty(start, stop):
    """Remove directories left behind after flattening, up to but excluding stop."""
    current = Path(start)
    stop = Path(stop)
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


# --------------------------------------------------------------------------
# URL parsing and interpreter discovery
# --------------------------------------------------------------------------

def parse_hf_url(url):
    """Return (repo, file_path, repo_type, revision) for a Hub URL or owner/name."""
    url = (url or "").strip()
    if not url:
        return None, None, "model", None

    url = re.sub(r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co)/", "hf://", url, flags=re.I)
    if not url.startswith("hf://"):
        if re.match(r"^[\w.\-]+/[\w.\-]+$", url):
            return url, "", "model", None
        return None, None, "model", None

    rest = url[len("hf://"):].split("?", 1)[0].split("#", 1)[0].strip("/")

    repo_type = "model"
    for prefix, kind in (("datasets/", "dataset"), ("spaces/", "space")):
        if rest.lower().startswith(prefix):
            repo_type = kind
            rest = rest[len(prefix):]
            break

    parts = rest.split("/")
    if len(parts) < 2:
        return None, None, repo_type, None

    repo = "/".join(parts[:2])
    remainder = parts[2:]
    revision = None
    file_path = ""

    if remainder and remainder[0] in ("resolve", "blob", "tree", "raw"):
        if len(remainder) >= 2:
            revision = unquote(remainder[1])
            file_path = "/".join(remainder[2:])
    else:
        file_path = "/".join(remainder)

    if revision in ("main", "master"):
        revision = None

    return repo, unquote(file_path), repo_type, revision


def probe_worker_python(python_exe):
    """True if python_exe can import a huggingface_hub recent enough to be worth using."""
    try:
        if not python_exe or not Path(python_exe).exists():
            return False
        result = subprocess.run(
            [str(python_exe), "-c", "import huggingface_hub as h, tqdm; print(h.__version__)"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return False
        parts = result.stdout.strip().split(".")
        version = tuple(int(re.sub(r"\D.*$", "", part) or 0) for part in parts[:2])
        return version >= MIN_HUB_VERSION
    except Exception:
        return False


def find_worker_python(hf_exe=None, preferred=None):
    """Locate an interpreter whose huggingface_hub supports Xet."""
    candidates = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.append(Path.home() / ".hf-cli" / "venv" / "Scripts" / "python.exe")
    candidates.append(Path.home() / ".hf-cli" / "venv" / "bin" / "python")
    if hf_exe:
        parent = Path(hf_exe).parent
        candidates += [parent / "python.exe", parent / "python", parent.parent / "python.exe"]
    candidates.append(Path(sys.executable))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if probe_worker_python(candidate):
            return str(candidate)
    return None


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

def gui_main():
    global tk, ttk, filedialog, messagebox
    import tkinter as tk_module
    from tkinter import filedialog as fd_module, messagebox as mb_module, ttk as ttk_module

    tk, ttk, filedialog, messagebox = tk_module, ttk_module, fd_module, mb_module
    HFDownloader().mainloop()


class HFDownloader:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1120x840")
        self.root.minsize(960, 720)

        self.alive = True
        self.pre_partials = set()
        self.ui_queue = queue.Queue()
        self.proc = None
        self.proc_lock = threading.Lock()
        self.cancelled = False
        self.saw_error = False
        self.samples = deque(maxlen=40)
        self.started_at = 0.0

        config = self._load_config()

        models_dir = config.get("tree_root") or find_comfy_models_dir() or ""

        self.url_var = tk.StringVar()
        self.repo_var = tk.StringVar(value=config.get("repo", ""))
        self.file_var = tk.StringVar(value=config.get("file", ""))
        self.revision_var = tk.StringVar(value=config.get("revision", ""))
        self.repo_type_var = tk.StringVar(value=config.get("repo_type", "model"))
        self.dest_var = tk.StringVar(value=config.get("dest", models_dir))
        self.tree_root_var = tk.StringVar(value=models_dir)
        self.mode_var = tk.StringVar(value=config.get("mode", "high"))
        self.flatten_var = tk.BooleanVar(value=config.get("flatten", True))
        self.workers_var = tk.IntVar(value=config.get("max_workers", 8))
        self.hf_path_var = tk.StringVar(value=config.get("hf_exe") or find_hf_cli())
        self.worker_py_var = tk.StringVar(value=config.get("worker_python", ""))

        self.status_var = tk.StringVar(value="Ready")
        self.free_var = tk.StringVar()
        self.speed_var = tk.StringVar(value="")
        self.pct_var = tk.StringVar(value="")

        self._build_ui()
        self._populate_tree()
        self._update_free_space()

        self.dest_var.trace_add("write", lambda *_: self._update_free_space())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._pump()
        self.root.after(200, self._detect_worker_python)

    def mainloop(self):
        self.root.mainloop()

    def _post(self, func, *args):
        """Queue work for the UI thread.

        Tk may only be touched from the thread running the event loop, so worker
        threads leave callbacks here and ``_pump`` runs them.
        """
        self.ui_queue.put((func, args))

    def _pump(self):
        """Drain queued UI work; reschedules itself for the life of the window."""
        if not self.alive:
            return
        while True:
            try:
                func, args = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args)
            except Exception:
                pass
        self.root.after(60, self._pump)

    def _max_workers(self):
        try:
            return max(1, min(32, int(self.workers_var.get())))
        except Exception:
            return 8

    # ---------------------------------------------------------------- config

    def _load_config(self):
        try:
            return json.loads(config_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_config(self):
        data = {
            "repo": self.repo_var.get(),
            "file": self.file_var.get(),
            "revision": self.revision_var.get(),
            "repo_type": self.repo_type_var.get(),
            "dest": self.dest_var.get(),
            "tree_root": self.tree_root_var.get(),
            "mode": self.mode_var.get(),
            "flatten": bool(self.flatten_var.get()),
            "max_workers": self._max_workers(),
            "hf_exe": self.hf_path_var.get(),
            "worker_python": self.worker_py_var.get(),
        }
        try:
            path = config_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        with self.proc_lock:
            running = self.proc is not None and self.proc.poll() is None
        if running and not messagebox.askokcancel(
            APP_TITLE, "A download is still running. Stop it and quit?"
        ):
            return
        self._save_config()
        self.alive = False
        self.cancelled = True
        self._kill_process()
        self.root.destroy()

    # -------------------------------------------------------------------- UI

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text=APP_TITLE, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Paste a Hugging Face link, or enter a repository and file path. "
                 "Leave the file path empty to fetch the whole repository."
        ).pack(anchor="w", pady=(2, 10))

        paned = ttk.Panedwindow(outer, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=(0, 0, 10, 0))
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        left.rowconfigure(16, weight=1)

        ttk.Label(left, text="Hugging Face URL").grid(row=0, column=0, columnspan=2, sticky="w")
        url_row = ttk.Frame(left)
        url_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        url_row.columnconfigure(0, weight=1)
        url_entry = ttk.Entry(url_row, textvariable=self.url_var)
        url_entry.grid(row=0, column=0, sticky="ew")
        url_entry.bind("<Return>", lambda _event: self._parse_url())
        ttk.Button(url_row, text="Parse URL", command=self._parse_url).grid(
            row=0, column=1, padx=(8, 0))

        ttk.Label(left, text="Repository").grid(row=2, column=0, sticky="w")
        ttk.Label(left, text="File path inside repository (blank = whole repo)").grid(
            row=2, column=1, sticky="w")

        ttk.Entry(left, textvariable=self.repo_var).grid(
            row=3, column=0, sticky="ew", padx=(0, 6), pady=(4, 10))
        ttk.Entry(left, textvariable=self.file_var).grid(
            row=3, column=1, sticky="ew", padx=(6, 0), pady=(4, 10))

        options = ttk.Frame(left)
        options.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(options, text="Type").pack(side="left")
        ttk.Combobox(
            options, textvariable=self.repo_type_var, width=9, state="readonly",
            values=["model", "dataset", "space"]
        ).pack(side="left", padx=(6, 14))
        ttk.Label(options, text="Revision").pack(side="left")
        ttk.Entry(options, textvariable=self.revision_var, width=16).pack(side="left", padx=(6, 14))
        ttk.Checkbutton(
            options, text="Save straight into destination (no repo subfolders)",
            variable=self.flatten_var
        ).pack(side="left")

        ttk.Label(left, text="Download destination").grid(row=5, column=0, columnspan=2, sticky="w")
        dest_row = ttk.Frame(left)
        dest_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        dest_row.columnconfigure(0, weight=1)
        ttk.Entry(dest_row, textvariable=self.dest_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(dest_row, text="Browse", command=self._browse_dest).grid(
            row=0, column=1, padx=(8, 0))

        quick_row = ttk.Frame(left)
        quick_row.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 4))
        ttk.Button(quick_row, text="Use models root",
                   command=lambda: self.dest_var.set(self.tree_root_var.get())).pack(side="left")
        ttk.Button(quick_row, text="Detect ComfyUI folder",
                   command=self._detect_models_root).pack(side="left", padx=(8, 0))

        ttk.Label(left, textvariable=self.free_var, font=("Segoe UI", 9, "bold")).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(0, 10))

        mode_box = ttk.LabelFrame(left, text="Transfer mode", padding=8)
        mode_box.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for text, value in (
            ("Xet high performance (fastest)", "high"),
            ("Xet normal", "normal"),
            ("Plain HTTPS, Xet disabled", "http"),
        ):
            ttk.Radiobutton(mode_box, text=text, variable=self.mode_var, value=value).pack(anchor="w")

        workers_row = ttk.Frame(mode_box)
        workers_row.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(workers_row, text="Parallel files (whole-repo downloads)").pack(side="left")
        ttk.Spinbox(workers_row, from_=1, to=32, width=5, textvariable=self.workers_var).pack(
            side="left", padx=(6, 0))

        engine_box = ttk.LabelFrame(left, text="Engine", padding=8)
        engine_box.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        engine_box.columnconfigure(1, weight=1)

        ttk.Label(engine_box, text="hf.exe").grid(row=0, column=0, sticky="w")
        ttk.Entry(engine_box, textvariable=self.hf_path_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(engine_box, text="Browse", command=self._browse_hf).grid(row=0, column=2)

        ttk.Label(engine_box, text="Worker Python").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(engine_box, textvariable=self.worker_py_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
        ttk.Button(engine_box, text="Detect", command=self._detect_worker_python).grid(
            row=1, column=2, pady=(6, 0))

        button_row = ttk.Frame(left)
        button_row.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        self.download_btn = ttk.Button(button_row, text="Download", command=self._start_download)
        self.download_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            button_row, text="Stop", command=self._stop_download, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Open Destination", command=self._open_destination).pack(
            side="left", padx=(8, 0))
        ttk.Button(button_row, text="Clean Partials", command=self._clean_partials).pack(
            side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(left, mode="determinate", maximum=1000)
        self.progress.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        stat_row = ttk.Frame(left)
        stat_row.grid(row=13, column=0, columnspan=2, sticky="ew")
        ttk.Label(stat_row, textvariable=self.pct_var, font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(stat_row, textvariable=self.speed_var).pack(side="right")

        ttk.Label(left, textvariable=self.status_var).grid(
            row=14, column=0, columnspan=2, sticky="w", pady=(2, 4))
        ttk.Label(left, text="Log").grid(row=15, column=0, columnspan=2, sticky="w")

        log_frame = ttk.Frame(left)
        log_frame.grid(row=16, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(log_frame, wrap="word", font=("Consolas", 9), height=8)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        ttk.Label(right, text="Models Tree", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w")

        root_row = ttk.Frame(right)
        root_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        root_row.columnconfigure(0, weight=1)
        ttk.Entry(root_row, textvariable=self.tree_root_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(root_row, text="Refresh", command=self._populate_tree).grid(
            row=0, column=1, padx=(8, 0))

        ttk.Label(right, text="Double click a folder to use it as the destination.").grid(
            row=2, column=0, sticky="w", pady=(0, 6))

        tree_frame = ttk.Frame(right)
        tree_frame.grid(row=3, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.bind("<<TreeviewOpen>>", self._tree_open)
        self.tree.bind("<Double-1>", lambda _event: self._use_selected_tree_folder())

        ttk.Button(right, text="Use Selected Folder", command=self._use_selected_tree_folder).grid(
            row=4, column=0, sticky="ew", pady=(8, 0))

    def _append_log(self, text):
        self.log.insert("end", text if text.endswith("\n") else text + "\n")
        self.log.see("end")

    # ---------------------------------------------------------------- inputs

    def _parse_url(self):
        repo, file_path, repo_type, revision = parse_hf_url(self.url_var.get())
        if not repo:
            messagebox.showerror(APP_TITLE, "Could not parse that Hugging Face URL.")
            return
        self.repo_var.set(repo)
        self.file_var.set(file_path or "")
        self.repo_type_var.set(repo_type)
        self.revision_var.set(revision or "")
        self.status_var.set(
            f"Parsed {repo}" + (f" :: {file_path}" if file_path else " (whole repository)")
        )

    def _browse_dest(self):
        start = self._nearest_existing(self.dest_var.get())
        folder = filedialog.askdirectory(
            title="Choose download destination", initialdir=str(start) if start else None
        )
        if folder:
            self.dest_var.set(os.path.normpath(folder))

    def _browse_hf(self):
        path = filedialog.askopenfilename(
            title="Locate hf.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.hf_path_var.set(path)

    @staticmethod
    def _nearest_existing(path):
        """Closest existing ancestor of path, without creating anything."""
        text = (path or "").strip()
        if not text:
            return None
        try:
            current = Path(text).expanduser()
        except Exception:
            return None
        while True:
            if current.exists():
                return current
            if current.parent == current:
                return None
            current = current.parent

    def _update_free_space(self):
        dest = self.dest_var.get().strip()
        if not dest:
            self.free_var.set("Free space: choose a destination")
            return
        anchor = self._nearest_existing(dest)
        if anchor is None:
            self.free_var.set("Free space: that path is not on an available drive")
            return
        try:
            usage = shutil.disk_usage(anchor)
        except Exception as exc:
            self.free_var.set(f"Free space check failed: {exc}")
            return
        note = "" if Path(dest).exists() else "   (folder will be created)"
        self.free_var.set(
            f"Free space: {human_bytes(usage.free)}   Total: {human_bytes(usage.total)}{note}"
        )

    def _detect_worker_python(self):
        """Probe for a usable interpreter off the UI thread; probing spawns processes."""
        hf_exe = self.hf_path_var.get().strip()
        preferred = self.worker_py_var.get().strip()
        threading.Thread(
            target=lambda: self._post(self._apply_worker_python,
                                      find_worker_python(hf_exe, preferred)),
            daemon=True,
        ).start()

    def _detect_models_root(self):
        found = find_comfy_models_dir()
        if not found:
            messagebox.showinfo(
                APP_TITLE,
                "No ComfyUI models folder was found automatically. "
                "Paste the path into the Models Tree box on the right instead."
            )
            return
        self.tree_root_var.set(found)
        self._populate_tree()
        self._append_log(f"Models folder: {found}")

    def _apply_worker_python(self, found):
        if found:
            self.worker_py_var.set(found)
            self._append_log(f"Download engine: {found}")
        else:
            self.worker_py_var.set("")
            self._append_log(
                "No Python with huggingface_hub >= 0.30 was found, so hf.exe will be used "
                "instead and progress will only be approximate."
            )

    # ------------------------------------------------------------------ tree

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        root = self.tree_root_var.get().strip()
        if not root:
            self.tree.insert("", "end", text="(set a models folder above)")
            return
        root_path = Path(root)
        root_id = self.tree.insert(
            "", "end", text=root_path.name or str(root_path), open=True, values=(str(root_path),)
        )
        if root_path.exists():
            self._insert_children(root_id, root_path)
        else:
            self.tree.insert(root_id, "end", text="(folder not found)")

    def _insert_children(self, parent_id, path):
        try:
            children = sorted(
                (p for p in path.iterdir() if p.is_dir()), key=lambda x: x.name.lower()
            )
        except OSError:
            return
        for child in children:
            node = self.tree.insert(parent_id, "end", text=child.name, values=(str(child),))
            try:
                has_subfolders = any(p.is_dir() for p in child.iterdir())
            except OSError:
                has_subfolders = False
            if has_subfolders:
                self.tree.insert(node, "end", text="")

    def _tree_open(self, _event=None):
        item = self.tree.focus()
        if not item:
            return
        children = self.tree.get_children(item)
        if len(children) == 1 and self.tree.item(children[0], "text") == "":
            self.tree.delete(children[0])
            values = self.tree.item(item, "values")
            if values:
                self._insert_children(item, Path(values[0]))

    def _use_selected_tree_folder(self):
        item = self.tree.focus()
        if not item:
            return
        values = self.tree.item(item, "values")
        if values:
            self.dest_var.set(os.path.normpath(values[0]))

    # -------------------------------------------------------------- download

    def _validate(self):
        repo = self.repo_var.get().strip().strip("/")
        file_path = self.file_var.get().strip().replace("\\", "/").lstrip("/")
        dest = self.dest_var.get().strip()

        if not re.match(r"^[\w.\-]+/[\w.\-]+$", repo):
            messagebox.showerror(
                APP_TITLE, "Repository should look like: owner/name, for example "
                           "black-forest-labs/FLUX.1-dev"
            )
            return None
        if not dest:
            messagebox.showerror(APP_TITLE, "Choose a destination folder.")
            return None

        try:
            Path(dest).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot create destination:\n{exc}")
            return None

        return repo, file_path, dest

    def _start_download(self):
        values = self._validate()
        if not values:
            return
        repo, file_path, dest = values

        self.cancelled = False
        self.saw_error = False
        self.samples.clear()
        self.started_at = time.monotonic()
        self.log.delete("1.0", "end")
        self.progress.configure(value=0)
        self.pct_var.set("")
        self.speed_var.set("")
        self.download_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Starting...")
        self._save_config()

        partials = self._scan_partials(dest)
        self.pre_partials = set(partials)
        if partials:
            self._append_log(
                f"Note: {len(partials)} leftover partial file(s) here using "
                f"{human_bytes(sum(partials.values()))}. Use Clean Partials to reclaim it."
            )

        env = os.environ.copy()
        env.pop("HF_HUB_DISABLE_XET", None)
        env.pop("HF_XET_HIGH_PERFORMANCE", None)
        env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        mode = self.mode_var.get()
        if mode == "high":
            env["HF_XET_HIGH_PERFORMANCE"] = "1"
        elif mode == "http":
            env["HF_HUB_DISABLE_XET"] = "1"

        job = {
            "repo": repo,
            "file": file_path,
            "dest": dest,
            "revision": self.revision_var.get().strip() or None,
            "repo_type": self.repo_type_var.get(),
            "flatten": bool(self.flatten_var.get()),
            "max_workers": self._max_workers(),
        }

        self._append_log(repo + (f" :: {file_path}" if file_path else " :: whole repository"))
        self._append_log(f"Destination: {dest}")
        self._append_log({
            "high": "Mode: Xet high performance",
            "http": "Mode: plain HTTPS, Xet disabled",
        }.get(mode, "Mode: Xet normal"))
        self._append_log("")

        worker_py = self.worker_py_var.get().strip()
        if worker_py:
            target, args = self._run_worker, (worker_py, job, env)
        else:
            target, args = self._run_cli, (job, env)
        threading.Thread(target=target, args=args, daemon=True).start()

    def _spawn(self, cmd, env, stdin=None):
        return subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            # Own process group, so stopping can take the hf/xet children with it.
            start_new_session=(os.name != "nt"),
        )

    def _run_worker(self, worker_py, job, env):
        try:
            cmd = [worker_py, os.path.abspath(__file__), WORKER_FLAG]
            proc = self._spawn(cmd, env, stdin=subprocess.PIPE)
            with self.proc_lock:
                self.proc = proc

            proc.stdin.write(json.dumps(job))
            proc.stdin.close()

            stderr_lines = []
            stderr_thread = threading.Thread(
                target=lambda: stderr_lines.extend(proc.stderr.read().splitlines()), daemon=True
            )
            stderr_thread.start()

            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._post(self._append_log, line)
                    continue
                self._post(self._handle_message, message)

            code = proc.wait()
            stderr_thread.join(timeout=2)

            if self.cancelled:
                self._post(self.status_var.set, "Stopped")
            elif code != 0 and not self.saw_error:
                tail = "\n".join(stderr_lines[-12:])
                self._post(self._append_log, tail or f"Worker exited with code {code}")
                self._post(self.status_var.set, f"Failed (exit code {code})")

        except Exception as exc:
            self._post(self._append_log, f"ERROR: {type(exc).__name__}: {exc}")
            self._post(self.status_var.set, "Failed")
        finally:
            with self.proc_lock:
                self.proc = None
            if self.cancelled or self.saw_error:
                self._post(self._cleanup_new_partials)
            self._post(self._finish_ui)

    def _run_cli(self, job, env):
        """Fallback path: drive hf.exe and estimate progress from disk activity.

        hf.exe prints nothing until it finishes when its output is a pipe, so the
        bar here is driven by watching the partial file grow instead.
        """
        hf_exe = self.hf_path_var.get().strip()
        if not hf_exe or not Path(hf_exe).exists():
            hf_exe = shutil.which("hf")
        if not hf_exe:
            self._post(self._append_log, "ERROR: hf.exe was not found.")
            self._post(self.status_var.set, "Failed")
            self._post(self._finish_ui)
            return

        cmd = [hf_exe, "download", job["repo"]]
        if job["file"]:
            cmd.append(job["file"])
        cmd += ["--local-dir", job["dest"], "--repo-type", job["repo_type"]]
        if job["revision"]:
            cmd += ["--revision", job["revision"]]
        if not job["file"]:
            cmd += ["--max-workers", str(job["max_workers"])]

        self._post(self._append_log, subprocess.list2cmdline(cmd) + "\n")

        stop_watch = threading.Event()
        threading.Thread(
            target=self._watch_disk, args=(Path(job["dest"]), stop_watch), daemon=True
        ).start()

        try:
            proc = self._spawn(cmd, env)
            with self.proc_lock:
                self.proc = proc
            for line in proc.stdout:
                self._post(self._append_log, line.rstrip())
            stderr_text = proc.stderr.read()
            code = proc.wait()
            if stderr_text.strip():
                self._post(self._append_log, stderr_text.strip())
            if self.cancelled:
                self._post(self.status_var.set, "Stopped")
            elif code == 0:
                self._post(self._handle_message, {"t": "done", "path": job["dest"]})
            else:
                self._post(self.status_var.set, f"Failed (exit code {code})")
        except Exception as exc:
            self._post(self._append_log, f"ERROR: {type(exc).__name__}: {exc}")
            self._post(self.status_var.set, "Failed")
        finally:
            stop_watch.set()
            with self.proc_lock:
                self.proc = None
            if self.cancelled or self.saw_error:
                self._post(self._cleanup_new_partials)
            self._post(self._finish_ui)

    def _watch_disk(self, dest, stop_event):
        staging = dest / ".cache" / "huggingface" / "download"
        while not stop_event.wait(0.5):
            total = 0
            try:
                for path in staging.rglob("*.incomplete"):
                    try:
                        total += path.stat().st_size
                    except OSError:
                        pass
            except OSError:
                pass
            if total:
                self._post(
                    self._handle_message, {"t": "p", "net": total, "disk": total, "total": 0}
                )

    def _handle_message(self, message):
        kind = message.get("t")

        if kind == "meta":
            total = message.get("total") or 0
            if total:
                self._append_log(f"Total size: {human_bytes(total)}")
                self._check_space(total)

        elif kind == "p":
            self._update_progress(message)

        elif kind == "log":
            self._append_log(message.get("msg", ""))

        elif kind == "done":
            self.progress.configure(value=1000)
            self.pct_var.set("100%")
            self.speed_var.set("")
            if message.get("skipped"):
                self.status_var.set("Already downloaded")
                self._append_log(f"Already present: {message.get('path')}")
            else:
                self.status_var.set("Download complete")
                self._append_log(f"\nSaved to: {message.get('path')}")
                self._append_log(f"Elapsed: {human_time(time.monotonic() - self.started_at)}")

        elif kind == "error":
            self.saw_error = True
            self.status_var.set("Failed")
            hint = message.get("hint") or ""
            self._append_log(f"\nERROR: {message.get('msg', '')}")
            if hint:
                self._append_log(hint)
            messagebox.showerror(
                APP_TITLE, message.get("msg", "Download failed") + (f"\n\n{hint}" if hint else "")
            )

    @staticmethod
    def _scan_partials(dest):
        """Leftover .incomplete files under dest, mapped to their sizes."""
        staging = Path(dest) / ".cache" / "huggingface" / "download"
        found = {}
        try:
            for path in staging.rglob("*.incomplete"):
                try:
                    found[path] = path.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
        return found

    def _cleanup_new_partials(self):
        """Delete the dead partial left by a run we just interrupted.

        huggingface_hub writes each attempt to a process-unique ``.incomplete``
        file opened with "wb", so an interrupted transfer is never resumed and its
        partial would otherwise sit in the model folder forever.
        """
        reclaimed = 0
        for path, size in self._scan_partials(self.dest_var.get()).items():
            if path in self.pre_partials:
                continue
            try:
                path.unlink()
                reclaimed += size
            except OSError:
                pass
        if reclaimed:
            self._append_log(
                f"Discarded the unfinished partial file, reclaiming {human_bytes(reclaimed)}. "
                "Hugging Face cannot resume a partial download, so retrying starts over."
            )
        self._update_free_space()

    def _clean_partials(self):
        """Remove leftover partials the user is carrying from earlier attempts."""
        cutoff = time.time() - 60
        partials = {}
        for path, size in self._scan_partials(self.dest_var.get()).items():
            try:
                if path.stat().st_mtime < cutoff:  # skip anything still being written
                    partials[path] = size
            except OSError:
                pass

        if not partials:
            messagebox.showinfo(APP_TITLE, "No leftover partial downloads were found here.")
            return

        total = sum(partials.values())
        listing = "\n".join(f"  {human_bytes(s)}  {p.name[:60]}" for p, s in list(partials.items())[:10])
        if not messagebox.askyesno(
            APP_TITLE,
            f"Delete {len(partials)} unfinished partial download(s) and reclaim "
            f"{human_bytes(total)}?\n\n{listing}\n\n"
            "These cannot be resumed, so they only take up space."
        ):
            return

        reclaimed = 0
        for path, size in partials.items():
            try:
                path.unlink()
                reclaimed += size
            except OSError as exc:
                self._append_log(f"Could not delete {path.name}: {exc}")
        self._append_log(f"Reclaimed {human_bytes(reclaimed)} from {len(partials)} partial file(s).")
        self._update_free_space()

    def _check_space(self, total):
        anchor = self._nearest_existing(self.dest_var.get())
        if anchor is None:
            return
        try:
            free = shutil.disk_usage(anchor).free
        except Exception:
            return
        if free < total * 1.1:
            self._append_log(
                f"WARNING: {human_bytes(free)} free but about {human_bytes(total)} is needed."
            )

    def _update_progress(self, message):
        total = message.get("total") or 0
        done = max(message.get("net", 0), message.get("disk", 0))
        now = time.monotonic()
        self.samples.append((now, done))

        speed = None
        if len(self.samples) >= 2:
            first_time, first_bytes = self.samples[0]
            elapsed = now - first_time
            if elapsed > 0.5:
                speed = max(0.0, (done - first_bytes) / elapsed)

        if total > 0:
            fraction = min(1.0, done / total)
            self.progress.configure(value=fraction * 1000)
            self.pct_var.set(f"{fraction * 100:.1f}%   {human_bytes(done)} / {human_bytes(total)}")
            if speed:
                self.speed_var.set(
                    f"{human_bytes(speed)}/s   ETA {human_time((total - done) / speed)}"
                )
        else:
            self.pct_var.set(human_bytes(done))
            if speed:
                self.speed_var.set(f"{human_bytes(speed)}/s")

        self.status_var.set("Downloading...")

    def _finish_ui(self):
        self.download_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._update_free_space()

    def _kill_process(self):
        with self.proc_lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                # terminate() would leave the hf/xet child processes running.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                # terminate() alone would leave the hf/xet children running.
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _stop_download(self):
        self.cancelled = True
        self.status_var.set("Stopping...")
        self.stop_btn.config(state="disabled")
        self._kill_process()

    def _open_destination(self):
        dest = self.dest_var.get().strip()
        if not dest:
            return
        try:
            Path(dest).mkdir(parents=True, exist_ok=True)
            open_in_file_manager(dest)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        sys.exit(worker_main())
    gui_main()
