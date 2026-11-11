param([string]$Scene)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root 'envs\mujoco\Scripts\python.exe'
if (-not $Scene) {
    $Pointer = Join-Path $Root 'configs\latest_scene.json'
    if (-not (Test-Path -LiteralPath $Pointer)) { throw 'Build the watering scene first.' }
    $Info = Get-Content -LiteralPath $Pointer -Raw -Encoding UTF8 | ConvertFrom-Json
    $Scene = Join-Path $Root $Info.scene
}
if (-not (Test-Path -LiteralPath $Scene)) { throw "Scene does not exist: $Scene" }
Write-Host 'Static layout viewer. Use open_trajectory_viewer.ps1 for the saved planned motion.'
& $Python -m mujoco.viewer "--mjcf=$Scene"
if ($LASTEXITCODE -ne 0) { throw "MuJoCo viewer failed: $LASTEXITCODE" }
