param(
  [Parameter(Mandatory=$true)][string]$InputMesh,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [Parameter(Mandatory=$true)][string]$Sam3Weights,
  [ValidateSet('auto','x','y','z')][string]$UpAxis='auto',
  [ValidateSet('cpu','cuda')][string]$Device='cpu',
  [string]$PythonExecutable='',
  [string]$Sam3Python='',
  [switch]$RefineLeafBoundary,
  [string]$MobileSamCheckpoint=''
)
$ErrorActionPreference='Stop'
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) { throw "Input mesh not found: $InputMesh" }
if ($RefineLeafBoundary) { $MobileSamCheckpoint=Resolve-PlantSam $MobileSamCheckpoint }
if (-not $Sam3Python) {
  $runtimeFolder=if ($Device -eq 'cuda') { '.venv-sam3-gpu' } else { '.venv-sam3' }
  $Sam3Python=Join-Path $scriptRoot "$runtimeFolder\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Sam3Python -PathType Leaf)) { throw "SAM3 Python not found: $Sam3Python" }
if (-not (Test-Path -LiteralPath (Join-Path $Sam3Weights 'model.safetensors'))) { throw 'Expected SAM3 Hugging Face checkpoint folder containing model.safetensors' }
if ($UpAxis -eq 'auto') {
  $axis=& $PythonExecutable (Join-Path $scriptRoot 'estimate_axis.py') --input $InputMesh
  if ($LASTEXITCODE -ne 0) { throw 'Axis estimation failed' }
  $UpAxis=($axis | Select-Object -Last 1).Trim()
}
$base=Join-Path $OutputDirectory 'render_global'
$detail=Join-Path $OutputDirectory 'render_detail'
$pred=Join-Path $OutputDirectory 'predictions'
$globalPred=Join-Path $pred 'global'
$detailPred=Join-Path $pred 'detail'
$complete=Join-Path $OutputDirectory 'complete'
& $PythonExecutable (Join-Path $scriptRoot 'plantseg.py') prepare --input $InputMesh --output $base --up $UpAxis
if ($LASTEXITCODE -ne 0) { throw 'Global rendering failed' }
& $Sam3Python (Join-Path $scriptRoot 'sam3_predict.py') --checkpoint $Sam3Weights --source $base --output $globalPred --device $Device
if ($LASTEXITCODE -ne 0) { throw 'SAM3 global inference failed' }
& $PythonExecutable (Join-Path $scriptRoot 'sam3_prepare_detail.py') --base $base --predictions $globalPred --output $detail
if ($LASTEXITCODE -ne 0) { throw 'Detail rendering failed' }
& $Sam3Python (Join-Path $scriptRoot 'sam3_predict.py') --checkpoint $Sam3Weights --source $detail --output $detailPred --device $Device
if ($LASTEXITCODE -ne 0) { throw 'SAM3 detail inference failed' }
& $PythonExecutable (Join-Path $scriptRoot 'sam3_fuse.py') --base $base --detail $detail --predictions $pred --output $complete
if ($LASTEXITCODE -ne 0) { throw 'SAM3 fusion failed' }
& $PythonExecutable (Join-Path $scriptRoot 'validate_complete.py') --input $InputMesh --output $complete
if ($LASTEXITCODE -ne 0) { throw 'Final geometry/labels validation failed' }
if ($RefineLeafBoundary) {
  $refined=Join-Path $OutputDirectory 'complete_leaf_refined'
  & $PythonExecutable (Join-Path $scriptRoot 'refine_leaf_boundary.py') --source $complete --output $refined --sam $MobileSamCheckpoint
  if ($LASTEXITCODE -ne 0) { throw 'SAM3 leaf boundary refinement failed' }
  & $PythonExecutable (Join-Path $scriptRoot 'validate_complete.py') --input $InputMesh --output $refined
  if ($LASTEXITCODE -ne 0) { throw 'Refined SAM3 geometry/labels validation failed' }
  Write-Host "SAM3 with MobileSAM leaf refinement: $(Join-Path $refined 'semantic_mesh.ply')"
}
Write-Host "SAM3 result: $(Join-Path $complete 'semantic_mesh.ply')"
