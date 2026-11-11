param(
    [string]$InputMesh = 'F:\植物建模\outputs\tetra_mesh_binary_search_7.ply',
    [string]$OutputDir = '',
    [string]$PythonExe = '',
    [ValidateSet('auto','green','flowers')][string]$SeedMode = 'auto'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) { throw "Mesh not found: $InputMesh" }
if (-not $PythonExe) {
    $BundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $BundledPython) { $PythonExe = $BundledPython }
    else { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
}
$CropArgs = @((Join-Path $PSScriptRoot 'auto_crop_mesh.py'), '--input', $InputMesh, '--seed-mode', $SeedMode)
if ($OutputDir) { $CropArgs += @('--output-dir', $OutputDir) }
& $PythonExe @CropArgs
if ($LASTEXITCODE -ne 0) { throw 'Automatic crop failed. See the reason above.' }
