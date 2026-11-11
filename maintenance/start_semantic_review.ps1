param(
    [string]$RunName = 'plant_20261106_new',
    [string]$Revision = '',
    [int]$Gpu = 1,
    [switch]$Plan,
    [switch]$Resume,
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
if (-not $Revision) { $Revision = $RunName + '_shape_' + (Get-Date -Format 'MMdd_HHmmss') }
foreach ($name in @($RunName,$Revision)) {
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'Invalid run/revision name (maximum 64 characters).' }
}
if ($RunName -eq $Revision -or $Gpu -lt 0) { throw 'Use a new revision name and a nonnegative GPU.' }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
$connection = "${ServerUser}@${ServerAddr}"
$remote = "'$RemoteRoot/envs/plant_semantics/bin/python' -u '$RemoteRoot/maintenance/refine_plant_semantics.py' --root '$RemoteRoot' --run '$RunName' --revision '$Revision' --gpu $Gpu --submit"
if ($Plan) { $remote += ' --plan' }
if ($Resume) { $remote += ' --resume' }
Invoke-Checked 'ssh.exe' @('-p',"$Port",'-o','ConnectTimeout=20',$connection,$remote)
Write-Host "Revision: $Revision"
Write-Host "Server log: $RemoteRoot/logs/semantic_$Revision.log"
Write-Host "Server results: $RemoteRoot/outputs/semantic_refinement/$Revision"
