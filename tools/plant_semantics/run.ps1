# Legacy baseline-only entry. Use run_auto.ps1 for the final five-class result.
param(
  [Parameter(Mandatory=$true)][string]$InputMesh,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [ValidateSet('x','y','z')][string]$UpAxis='y',
  [string]$PythonExecutable=''
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
& $PythonExecutable (Join-Path $PSScriptRoot 'plantseg.py') run --input $InputMesh --output $OutputDirectory --up $UpAxis
if ($LASTEXITCODE -ne 0) { throw "Plant segmentation failed with exit code $LASTEXITCODE" }
