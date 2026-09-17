"""Regenerate assets/icon.png and assets/icon.ico.

Drawn at 1024 px and downsampled, so small sizes stay crisp. Needs Pillow.
"""

from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "assets"
SIZE = 1024
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]

TOP = (255, 214, 64)      # warm yellow
BOTTOM = (255, 138, 0)    # orange
INK = (255, 255, 255)
SHADOW = (150, 70, 0, 90)


def gradient(size, top, bottom):
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        row = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom))
        for x in range(size):
            px[x, y] = row
    return img


def arrow(draw, dx=0, dy=0, fill=INK):
    s = SIZE
    cx = s // 2 + dx
    # Shaft
    shaft_w = s * 0.17
    draw.rounded_rectangle(
        (cx - shaft_w / 2, s * 0.17 + dy, cx + shaft_w / 2, s * 0.55 + dy),
        radius=shaft_w * 0.3, fill=fill,
    )
    # Head
    head_w = s * 0.50
    draw.polygon(
        [(cx - head_w / 2, s * 0.46 + dy), (cx + head_w / 2, s * 0.46 + dy), (cx, s * 0.72 + dy)],
        fill=fill,
    )
    # Tray the arrow lands in
    bar_h = s * 0.085
    draw.rounded_rectangle(
        (s * 0.20 + dx, s * 0.78 + dy, s * 0.80 + dx, s * 0.78 + bar_h + dy),
        radius=bar_h / 2, fill=fill,
    )


def build():
    s = SIZE
    base = gradient(s, TOP, BOTTOM).convert("RGBA")

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.22), fill=255)

    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    arrow(ImageDraw.Draw(shadow), dx=0, dy=int(s * 0.018), fill=SHADOW)
    base.alpha_composite(shadow)
    arrow(ImageDraw.Draw(base))

    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    icon.paste(base, (0, 0), mask)
    return icon


def main():
    ASSETS.mkdir(exist_ok=True)
    icon = build()
    icon.resize((512, 512), Image.LANCZOS).save(ASSETS / "icon.png", optimize=True)
    frames = [icon.resize((n, n), Image.LANCZOS) for n in ICO_SIZES]
    frames[-1].save(ASSETS / "icon.ico", sizes=[(n, n) for n in ICO_SIZES],
                    append_images=frames[:-1])
    print("wrote", ASSETS / "icon.png", "and", ASSETS / "icon.ico")


if __name__ == "__main__":
    main()
