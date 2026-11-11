<#
.SYNOPSIS
Run v3 segmentation and additional-view uncertainty review in one command.
.DESCRIPTION
Uses GroundingDINO, CLIPSeg and MobileSAM. Final paths are saved to
OutputDirectory\result_paths.json and result_paths.txt. No SAM3 is used.
ExtraViews is a coverage budget; up to three high views can be added for soil.
.EXAMPLE
.\run_v3.ps1 -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new_v3' -UpAxis y
#>
param(
  [Parameter(Mandatory=$true)][string]$InputMesh,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [ValidateSet('auto','x','y','z')][string]$UpAxis='auto',
  [ValidateRange(0,24)][int]$ExtraViews=12,
  [string]$PythonExecutable='',
  [string]$SamCheckpoint=''
)
$ErrorActionPreference='Stop'
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptRoot 'runtime.ps1')
if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) { throw "Input mesh not found: $InputMesh" }
$InputMesh=(Resolve-Path -LiteralPath $InputMesh).Path
$PythonExecutable=Resolve-PlantPython $PythonExecutable
$SamCheckpoint=Resolve-PlantSam $SamCheckpoint
$OutputDirectory=(New-Item -ItemType Directory -Force -Path $OutputDirectory).FullName
$segmentationPath=Join-Path $OutputDirectory 'segmentation'
$extraPath=Join-Path $OutputDirectory 'extra_views'
$finalPath=Join-Path $segmentationPath 'complete'

& (Join-Path $scriptRoot 'run_auto.ps1') -InputMesh $InputMesh -OutputDirectory $segmentationPath -UpAxis $UpAxis -PythonExecutable $PythonExecutable -SamCheckpoint $SamCheckpoint
if ($LASTEXITCODE -ne 0) { throw 'V3 segmentation failed' }
if ($ExtraViews -gt 0) {
  & (Join-Path $scriptRoot 'run_extra_views.ps1') -SourceDirectory $finalPath -OutputDirectory $extraPath -Views $ExtraViews -PythonExecutable $PythonExecutable -SamCheckpoint $SamCheckpoint
  if ($LASTEXITCODE -ne 0) { throw 'V3 additional-view review failed' }
  $finalPath=Join-Path $extraPath 'complete'
}

$paths=[ordered]@{
  version=(Get-Content -LiteralPath (Join-Path $scriptRoot 'VERSION') -Raw).Trim()
  input_mesh=$InputMesh
  output_directory=$OutputDirectory
  requested_up_axis=$UpAxis
  extra_view_budget=$ExtraViews
  sam3_used=$false
  final_directory=$finalPath
  semantic_mesh=(Join-Path $finalPath 'semantic_mesh.ply')
  labeled_original_rgb=(Join-Path $finalPath 'labeled_original_rgb.ply')
  labels=(Join-Path $finalPath 'labels.npz')
  class_definitions=(Join-Path $finalPath 'classes.json')
  report=(Join-Path $finalPath 'report.json')
  validation=(Join-Path $finalPath 'validation.json')
  uncertainty_preview=(Join-Path $finalPath 'uncertainty_comparison.jpg')
  uncertainty_report=(Join-Path $finalPath 'uncertainty_report.json')
}
if ($ExtraViews -gt 0) {
  $paths.uncertainty_report=Join-Path $finalPath 'extra_view_report.json'
}
foreach ($key in @('semantic_mesh','labeled_original_rgb','labels','class_definitions','report','validation','uncertainty_report')) {
  if (-not (Test-Path -LiteralPath $paths[$key] -PathType Leaf)) { throw "Expected result missing: $($paths[$key])" }
}
$paths | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'result_paths.json') -Encoding UTF8
$paths.GetEnumerator() | ForEach-Object { '{0}={1}' -f $_.Key,$_.Value } | Set-Content -LiteralPath (Join-Path $OutputDirectory 'result_paths.txt') -Encoding UTF8
Write-Host "Result paths: $(Join-Path $OutputDirectory 'result_paths.txt')"
Write-Output $paths.semantic_mesh
