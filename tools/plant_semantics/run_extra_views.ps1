<#
.SYNOPSIS
Review an existing v3 result using additional rendered RGB camera directions.
.DESCRIPTION
Semantic labels remain frozen. Uses GroundingDINO, CLIPSeg and MobileSAM.
Results and validation are written under OutputDirectory\complete.
Views is a coverage budget; up to three spaced high views may be added for soil.
.EXAMPLE
.\run_extra_views.ps1 -SourceDirectory 'D:\results\plant\complete' -OutputDirectory 'D:\results\plant_extra'
#>
param(
  [Parameter(Mandatory=$true)][string]$SourceDirectory,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [ValidateRange(1,24)][int]$Views=12,
  [ValidateRange(32,4096)][int]$Size=896,
  [ValidateRange(1,64)][int]$Threads=8,
  [ValidateSet('auto','cpu','cuda')][string]$Device='cpu',
  [string]$PythonExecutable='',
  [string]$SamCheckpoint=''
)
$ErrorActionPreference='Stop'
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
$SamCheckpoint=Resolve-PlantSam $SamCheckpoint
$SourceDirectory=(Resolve-Path -LiteralPath $SourceDirectory).Path
$manifestPath=Join-Path $SourceDirectory 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Source manifest.json was not found' }
$manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
& $PythonExecutable (Join-Path $scriptRoot 'extra_view_review.py') run --source $SourceDirectory --output $OutputDirectory --views $Views --size $Size --threads $Threads --device $Device --sam $SamCheckpoint
if ($LASTEXITCODE -ne 0) { throw 'Extra-view review failed' }
$completePath=Join-Path $OutputDirectory 'complete'
& $PythonExecutable (Join-Path $scriptRoot 'validate_complete.py') --input $manifest.input --output $completePath
if ($LASTEXITCODE -ne 0) { throw 'Extra-view export validation failed' }
Write-Output (Join-Path $completePath 'semantic_mesh.ply')
