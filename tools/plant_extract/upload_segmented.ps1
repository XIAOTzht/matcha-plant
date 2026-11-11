$ErrorActionPreference = 'Stop'
$ServerAddr = '10.7.88.77'
$ServerUser = 'fyp24_haoteng_zhang'
$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
$Connection = "${ServerUser}@${ServerAddr}"
$ToolDir = $PSScriptRoot
$MaskDir = Join-Path $PSScriptRoot '..\..\plant_smoke\segmented_rembg_p\masks'
$MaskFiles = @(Get-ChildItem -LiteralPath $MaskDir -Filter '*_mask.png' -File)
if ($MaskFiles.Count -ne 12) {
    throw "Expected 12 masks, found $($MaskFiles.Count). Check $MaskDir"
}

ssh $Connection "mkdir -p '$RemoteRoot/tools/plant_extract' '$RemoteRoot/data/plant_smoke_segmented/masks'"
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed' }

$CodeFiles = @('auto_extract_plant.py', 'requirements.txt', 'run_extract.sh', 'extract_after_segmented.sh') |
    ForEach-Object { Join-Path $ToolDir $_ }
scp -C @CodeFiles "${Connection}:${RemoteRoot}/tools/plant_extract/"
if ($LASTEXITCODE -ne 0) { throw 'Code upload failed' }

$MaskPaths = @($MaskFiles | ForEach-Object { $_.FullName })
scp -C @MaskPaths "${Connection}:${RemoteRoot}/data/plant_smoke_segmented/masks/"
if ($LASTEXITCODE -ne 0) { throw 'Mask upload failed' }

Write-Host 'Code and 12 masks uploaded. Next: follow README.md server commands.'
