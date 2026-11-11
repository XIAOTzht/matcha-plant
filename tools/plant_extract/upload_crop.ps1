$ErrorActionPreference = 'Stop'
$ServerAddr = '10.7.88.77'
$ServerUser = 'fyp24_haoteng_zhang'
$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
$Connection = "${ServerUser}@${ServerAddr}"
$RemoteDir = "$RemoteRoot/tools/plant_extract"

# auto_crop_mesh imports load_mesh from auto_extract_plant; upload both files.
$FileNames = @(
    'auto_crop_mesh.py',
    'auto_extract_plant.py',
    'requirements.txt',
    'run_crop.sh',
    'run_crop.ps1'
)
$Sources = @($FileNames | ForEach-Object {
    $SourcePath = Join-Path $PSScriptRoot $_
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        throw "Required file missing: $SourcePath"
    }
    $SourcePath
})

ssh $Connection "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed' }

scp @Sources "${Connection}:${RemoteDir}/"
if ($LASTEXITCODE -ne 0) { throw 'Crop tool upload failed' }

ssh $Connection "ls -lh '$RemoteDir/auto_crop_mesh.py' '$RemoteDir/auto_extract_plant.py' '$RemoteDir/run_crop.sh' '$RemoteDir/requirements.txt'"
if ($LASTEXITCODE -ne 0) { throw 'Remote file check failed' }
Write-Host "Uploaded crop tools to: $RemoteDir"
Write-Host 'Next: install requirements and run run_crop.sh on the server.'
