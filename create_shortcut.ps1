<#
.SYNOPSIS
    Adds Hugging Face Model Downloader to the Start menu (and optionally the desktop)
    with its own icon, ready to pin to the taskbar.

.DESCRIPTION
    The shortcut launches the app with pythonw.exe, so no console window appears, and
    stamps it with the same AppUserModelID the app sets on itself. That shared ID is
    what makes Windows treat the pinned icon and the running window as one button.

.PARAMETER Python
    python.exe or pythonw.exe to run the app with. Defaults to the first Python on
    PATH that has Tk.

.PARAMETER Desktop
    Also put a shortcut on the desktop.

.PARAMETER Remove
    Delete the shortcuts instead of creating them.

.EXAMPLE
    .\create_shortcut.ps1
.EXAMPLE
    .\create_shortcut.ps1 -Desktop -Python "C:\Python312\pythonw.exe"
#>
param(
    [string]$Python,
    [switch]$Desktop,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$AppName = 'HF Model Downloader'
$AppId = 'HallettVisual.HFModelDownloader'   # must match APP_USER_MODEL_ID in hf_downloader.py
$RepoDir = $PSScriptRoot
$Script = Join-Path $RepoDir 'hf_downloader.py'
$Icon = Join-Path $RepoDir 'assets\icon.ico'

$StartMenuLink = Join-Path ([Environment]::GetFolderPath('Programs')) "$AppName.lnk"
$DesktopLink = Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk"

if ($Remove) {
    foreach ($link in $StartMenuLink, $DesktopLink) {
        if (Test-Path $link) {
            Remove-Item $link
            Write-Host "Removed $link"
        }
    }
    Write-Host 'If you pinned it, unpin it from the taskbar too.'
    return
}

foreach ($required in $Script, $Icon) {
    if (-not (Test-Path $required)) { throw "Missing $required - run this from the repository folder." }
}

function Test-TkPython([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    # The Microsoft Store alias in WindowsApps is a stub that opens the Store.
    if ($exe -like '*\WindowsApps\*') { return $false }
    & $exe -c 'import tkinter' 2>$null
    return $LASTEXITCODE -eq 0
}

function Resolve-PythonW {
    $candidates = @()
    if ($Python) { $candidates += $Python }
    $candidates += Get-Command python.exe -All -ErrorAction SilentlyContinue | ForEach-Object Source
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) { $candidates += (& $launcher.Source -3 -c 'import sys; print(sys.executable)' 2>$null) }

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        $dir = Split-Path $candidate -Parent
        $console = Join-Path $dir 'python.exe'
        $windowed = Join-Path $dir 'pythonw.exe'
        if ((Test-Path $windowed) -and (Test-TkPython $console)) { return $windowed }
    }
    throw 'No Python with Tk found. Install Python from python.org, or pass -Python <path to pythonw.exe>.'
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace HfDownloaderShortcut
{
    [ComImport, Guid("000214F9-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IShellLinkW
    {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder file, int cch, IntPtr findData, uint flags);
        void GetIDList(out IntPtr pidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder name, int cch);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder dir, int cch);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string dir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder args, int cch);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string args);
        void GetHotkey(out short hotkey);
        void SetHotkey(short hotkey);
        void GetShowCmd(out int showCmd);
        void SetShowCmd(int showCmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int cch, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr hwnd, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string file);
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey { public Guid FormatId; public uint PropertyId; }

    [StructLayout(LayoutKind.Explicit)]
    public struct PropVariant
    {
        [FieldOffset(0)] public ushort VarType;
        [FieldOffset(8)] public IntPtr Pointer;
    }

    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class ShellLink { }

    public static class Maker
    {
        [DllImport("ole32.dll")]
        static extern int PropVariantClear(ref PropVariant value);

        public static void Create(string linkPath, string target, string arguments,
                                  string workingDir, string icon, string description, string appId)
        {
            var link = (IShellLinkW)new ShellLink();
            link.SetPath(target);
            link.SetArguments(arguments);
            link.SetWorkingDirectory(workingDir);
            link.SetIconLocation(icon, 0);
            link.SetDescription(description);

            // PKEY_AppUserModel_ID
            var key = new PropertyKey { FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), PropertyId = 5 };
            var value = new PropVariant { VarType = 31 /* VT_LPWSTR */, Pointer = Marshal.StringToCoTaskMemUni(appId) };
            var store = (IPropertyStore)link;
            store.SetValue(ref key, ref value);
            store.Commit();
            PropVariantClear(ref value);

            ((IPersistFile)link).Save(linkPath, true);
        }
    }
}
'@

$pythonw = Resolve-PythonW
$arguments = '"' + $Script + '"'
$description = 'Download Hugging Face models with live progress'

$links = @($StartMenuLink)
if ($Desktop) { $links += $DesktopLink }

foreach ($link in $links) {
    [HfDownloaderShortcut.Maker]::Create($link, $pythonw, $arguments, $RepoDir, $Icon, $description, $AppId)
    Write-Host "Created $link"
}

Write-Host ''
Write-Host "Runs with: $pythonw"
Write-Host ''
Write-Host 'To pin it: open Start, find "HF Model Downloader", right-click it and choose Pin to taskbar.'
Write-Host '(Or launch it, then right-click its taskbar button and choose Pin to taskbar.)'
