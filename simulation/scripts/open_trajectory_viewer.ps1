param([string]$Replay, [double]$Speed = 1.0)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root 'envs\mujoco\Scripts\python.exe'
$Script = Join-Path $Root 'scripts\replay_watering_viewer.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python missing: $Python" }
$PathPointer = Join-Path $Root 'configs\latest_path.json'
if (Test-Path -LiteralPath $PathPointer) {
    $PathInfo = Get-Content -LiteralPath $PathPointer -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($PathInfo.status -ne 'PATH_PLANNING_COMPLETE') {
        throw "No completed weighted path is available. Run goal screening and weighted planning first."
    }
}
$ReplayArgs = @($Script, '--speed', [string]$Speed)
if ($Replay) { $ReplayArgs += @('--replay', $Replay) }
Write-Host 'Saved simulation playback: SPACE pause, R restart, T reference/actual.'
& $Python @ReplayArgs
if ($LASTEXITCODE -ne 0) { throw "Replay failed: $LASTEXITCODE" }
