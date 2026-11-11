# Legacy detail-stage entry for developers. Use run_auto.ps1 for a new input.
param(
  [Parameter(Mandatory=$true)][string]$SourceDirectory,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [string]$PythonExecutable=''
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
foreach ($stage in @('prepare','predict','fuse')) {
  & $PythonExecutable (Join-Path $PSScriptRoot 'refine_detail.py') $stage --source $SourceDirectory --output $OutputDirectory
  if ($LASTEXITCODE -ne 0) { throw "Detail segmentation stage $stage failed with exit code $LASTEXITCODE" }
}
