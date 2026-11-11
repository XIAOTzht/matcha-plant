# Legacy stage entry: does not include leaf boundary repair. Use run_auto.ps1.
param(
  [Parameter(Mandatory=$true)][string]$BaseDirectory,
  [Parameter(Mandatory=$true)][string]$DetailDirectory,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [string]$PythonExecutable=''
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
& $PythonExecutable (Join-Path $PSScriptRoot 'complete_five.py') run --base $BaseDirectory --detail $DetailDirectory --output $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw "Complete five-class segmentation failed with exit code $LASTEXITCODE" }
