# Pipeline implementation; run_auto.ps1 is the supported user entry point.
param(
  [Parameter(Mandatory=$true)][string]$InputMesh,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [ValidateSet('x','y','z')][string]$UpAxis='y',
  [string]$PythonExecutable='',
  [string]$SamCheckpoint=''
)
$ErrorActionPreference='Stop'
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptRoot 'runtime.ps1')
if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) { throw "Input mesh not found: $InputMesh" }
$PythonExecutable=Resolve-PlantPython $PythonExecutable
$SamCheckpoint=Resolve-PlantSam $SamCheckpoint
$InputMesh=(Resolve-Path -LiteralPath $InputMesh).Path
$OutputDirectory=(Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $OutputDirectory)).Path
$baselinePath=Join-Path $OutputDirectory 'baseline'
$detailPath=Join-Path $OutputDirectory 'detail'
$preBoundaryPath=Join-Path $OutputDirectory 'complete_before_boundary'
$boundaryPath=Join-Path $OutputDirectory 'complete_before_rim'
$rimPath=Join-Path $OutputDirectory 'complete_before_uncertainty'
$completePath=Join-Path $OutputDirectory 'complete'
& $PythonExecutable (Join-Path $scriptRoot 'plantseg.py') run --input $InputMesh --output $baselinePath --up $UpAxis --sam $SamCheckpoint
if ($LASTEXITCODE -ne 0) { throw 'Baseline stage failed' }
foreach ($stage in @('prepare','predict','fuse')) {
  & $PythonExecutable (Join-Path $scriptRoot 'refine_detail.py') $stage --source $baselinePath --output $detailPath --sam $SamCheckpoint
  if ($LASTEXITCODE -ne 0) { throw "Detail $stage stage failed" }
}
& $PythonExecutable (Join-Path $scriptRoot 'complete_five.py') run --base $baselinePath --detail $detailPath --output $preBoundaryPath --sam $SamCheckpoint
if ($LASTEXITCODE -ne 0) { throw 'Complete five-class stage failed' }
& $PythonExecutable (Join-Path $scriptRoot 'refine_leaf_boundary.py') --source $preBoundaryPath --output $boundaryPath --sam $SamCheckpoint
if ($LASTEXITCODE -ne 0) { throw 'Leaf boundary refinement failed' }
& $PythonExecutable (Join-Path $scriptRoot 'refine_leaf_rim.py') --source $boundaryPath --output $rimPath
if ($LASTEXITCODE -ne 0) { throw 'Leaf rim completion failed' }
& $PythonExecutable (Join-Path $scriptRoot 'reassess_uncertainty.py') --source $rimPath --instances $preBoundaryPath --output $completePath
if ($LASTEXITCODE -ne 0) { throw 'Final uncertainty reassessment failed' }
& $PythonExecutable (Join-Path $scriptRoot 'validate_complete.py') --input $InputMesh --output $completePath
if ($LASTEXITCODE -ne 0) { throw 'Export validation failed' }
Write-Output (Join-Path $completePath 'semantic_mesh.ply')
