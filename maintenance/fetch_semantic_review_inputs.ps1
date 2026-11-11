param(
    [string]$RunName = 'plant_20261106_new',
    [string]$OutputDirectory = '',
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'Invalid run name.' }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $project ('outputs\semantic_review_inputs\'+$RunName+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a new directory to preserve existing results.' }
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$connection = "${ServerUser}@${ServerAddr}"
$options = @('-P',"$Port",'-o','ConnectTimeout=20')
$index = Join-Path $OutputDirectory 'pipeline_result_paths.json'
Invoke-Checked 'scp.exe' ($options + @("${connection}:$RemoteRoot/outputs/pipeline/$RunName/result_paths.json",$index))
$report = Get-Content -LiteralPath $index -Raw -Encoding UTF8 | ConvertFrom-Json
if ($report.status -ne 'PHOTO_TO_PATH_SUCCESS') { throw 'Source run has not completed.' }
$remoteMesh = [string]$report.semantic_mesh
$prefix = "$RemoteRoot/outputs/pipeline/$RunName/"
if (-not $remoteMesh.StartsWith($prefix,[StringComparison]::Ordinal) -or
    $remoteMesh.Substring($prefix.Length) -notmatch '^[A-Za-z0-9_/-]+/semantic_mesh_mm\.ply$') {
    throw 'Unexpected source mesh path.'
}
$source = $remoteMesh.Substring(0,$remoteMesh.LastIndexOf('/'))
$names = @('labeled_original_rgb_mm.ply','semantic_mesh_mm.ply','labels.npz','coordinate_system.json','classes.json','report.json')
$remoteFiles = @($names | ForEach-Object { "${connection}:$source/$_" })
Invoke-Checked 'scp.exe' ($options + $remoteFiles + @($OutputDirectory))
Write-Host "SEMANTIC_REVIEW_INPUTS_DOWNLOADED: $OutputDirectory"
Write-Host 'Original RGB and its matching labels are ready; no existing model or pipeline job was changed.'
