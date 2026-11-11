param(
    [string]$RunName = 'plant_20261108_coordinated',
    [string]$Revision = 'plant_20261108_virtual_soil_v1',
    [double]$RimOffsetMm = 10,
    [double]$WallInsetMm = 6,
    [switch]$Resume,
    [switch]$RepairDisplayMesh,
    [switch]$SkipUpload,
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
foreach ($name in @($RunName,$Revision)) {
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'Invalid run/revision name.' }
}
if ($RunName -eq $Revision) { throw 'Use a new revision name to preserve the original run.' }
if ($RepairDisplayMesh -and -not $Resume) { throw 'RepairDisplayMesh requires Resume.' }
if ([double]::IsNaN($RimOffsetMm) -or [double]::IsNaN($WallInsetMm) -or
    $RimOffsetMm -lt 3 -or $RimOffsetMm -gt 30 -or $WallInsetMm -lt 3 -or $WallInsetMm -gt 20) {
    throw 'RimOffsetMm must be 3..30, WallInsetMm must be 3..20.'
}
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
if (-not $SkipUpload) {
    & (Join-Path $PSScriptRoot 'upload_plant_pipeline.ps1') -ServerAddr $ServerAddr -ServerUser $ServerUser -Port $Port -RemoteRoot $RemoteRoot
}
$connection = "${ServerUser}@${ServerAddr}"
$rim = $RimOffsetMm.ToString([Globalization.CultureInfo]::InvariantCulture)
$inset = $WallInsetMm.ToString([Globalization.CultureInfo]::InvariantCulture)
$remote = "'$RemoteRoot/envs/plant_semantics/bin/python' -u '$RemoteRoot/maintenance/continue_with_virtual_substrate.py' --root '$RemoteRoot' --run '$RunName' --revision '$Revision' --assume-pot-interior-substrate --rim-offset-mm $rim --wall-inset-mm $inset --submit"
if ($Resume) { $remote += ' --resume' }
if ($RepairDisplayMesh) {
    $remote = "'$RemoteRoot/envs/plant_semantics/bin/python' -u '$RemoteRoot/maintenance/resume_display_mesh.py' --root '$RemoteRoot' --run '$Revision' && " + $remote
}
Invoke-Checked 'ssh.exe' @('-p',"$Port",'-o','ConnectTimeout=20',$connection,$remote)
Write-Host "Server revision: $Revision"
Write-Host "Server log: $RemoteRoot/logs/pipeline_$Revision.log"
Write-Host "Server results: $RemoteRoot/outputs/pipeline/$Revision"
Write-Host 'The soil surface is an explicit simulation assumption. Wait for PHOTO_TO_PATH_SUCCESS before downloading a verified path.'
