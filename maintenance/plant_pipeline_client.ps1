# Shared client helpers. Scripts use runtime paths rather than embedded Chinese literals.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited $LASTEXITCODE" }
}
function Get-ClientPython {
    param([string]$Project, [string]$Requested)
    if ($Requested) {
        if (-not (Test-Path -LiteralPath $Requested -PathType Leaf)) { throw "Missing Python: $Requested" }
        return $Requested
    }
    $candidates = @((Join-Path $Project 'envs\mujoco\Scripts\python.exe'),
        'F:\matcha_mujoco_simulation\envs\mujoco\Scripts\python.exe',
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'))
    foreach ($candidate in $candidates) { if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate } }
    throw 'Supply -PythonExe with a working Python 3.11+ installation containing Pillow.'
}
function Assert-RemoteArguments {
    param([string]$Address, [string]$UserName, [string]$Root, [int]$PortNumber)
    if ($Address -notmatch '^[A-Za-z0-9.-]+$' -or $UserName -notmatch '^[A-Za-z0-9_.-]+$' -or
        $Root -notmatch '^/[A-Za-z0-9_./-]+$' -or $Root.Contains('..') -or $PortNumber -lt 1 -or $PortNumber -gt 65535) {
        throw 'Invalid SSH destination or remote project path. Use an absolute Linux path without spaces.'
    }
    Get-Command ssh.exe, scp.exe -ErrorAction Stop | Out-Null
}
