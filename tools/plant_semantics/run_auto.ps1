<#
.SYNOPSIS
Run the complete five-class plant mesh segmentation, including leaf boundary repair.
.DESCRIPTION
Use a colored triangle mesh PLY. Final results are under OutputDirectory\complete.
Auto estimates only the positive vertical axis; manually orient rotated meshes first.
.EXAMPLE
.\run_auto.ps1 -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new' -UpAxis y
#>
param(
  [Parameter(Mandatory=$true)][string]$InputMesh,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [ValidateSet('auto','x','y','z')][string]$UpAxis='auto',
  [string]$PythonExecutable='',
  [string]$SamCheckpoint=''
)
$ErrorActionPreference='Stop'
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptRoot 'runtime.ps1')
if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) { throw "Input mesh not found: $InputMesh" }
$PythonExecutable=Resolve-PlantPython $PythonExecutable
$SamCheckpoint=Resolve-PlantSam $SamCheckpoint
if ($UpAxis -eq 'auto') {
  $axis = & $PythonExecutable (Join-Path $scriptRoot 'estimate_axis.py') --input $InputMesh
  if ($LASTEXITCODE -ne 0) { throw 'Automatic vertical-axis estimation failed' }
  $axis = ($axis | Select-Object -Last 1).Trim()
  if ($axis -notin @('x','y','z')) { throw "Invalid estimated up axis: $axis" }
  Write-Host "Estimated up axis: $axis"
  $UpAxis=$axis
}
& (Join-Path $scriptRoot 'run_all_five.ps1') -InputMesh $InputMesh -OutputDirectory $OutputDirectory -UpAxis $UpAxis -PythonExecutable $PythonExecutable -SamCheckpoint $SamCheckpoint
if ($LASTEXITCODE -ne 0) { throw 'Automatic plant segmentation failed' }
