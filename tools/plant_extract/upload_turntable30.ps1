$ErrorActionPreference = 'Stop'
$ServerAddr = '10.7.88.77'
$ServerUser = 'fyp24_haoteng_zhang'
$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
$Connection = "${ServerUser}@${ServerAddr}"
$MaskDir = Join-Path $PSScriptRoot '..\..\plant_smoke\segmented_rembg_p\masks'
$Masks = @(Get-ChildItem -LiteralPath $MaskDir -Filter '*_mask.png' -File)
if ($Masks.Count -ne 12) { throw 'Expected 12 foreground masks' }
ssh $Connection "mkdir -p '$RemoteRoot/tools/plant_extract' '$RemoteRoot/data/plant_smoke_segmented/masks'"
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed' }
$Code = @('prepare_turntable30.py', 'run_fixed_pose_sfm.py', 'run_checked_stage.py', 'run_turntable30.sh') |
    ForEach-Object { Join-Path $PSScriptRoot $_ }
scp -C @Code "${Connection}:${RemoteRoot}/tools/plant_extract/"
if ($LASTEXITCODE -ne 0) { throw 'Code upload failed' }
$MaskPaths = @($Masks | ForEach-Object { $_.FullName })
scp -C @MaskPaths "${Connection}:${RemoteRoot}/data/plant_smoke_segmented/masks/"
if ($LASTEXITCODE -ne 0) { throw 'Mask upload failed' }
Write-Host '30-degree experiment scripts and masks uploaded. Start the tmux command on the server.'
