param(
    [string]$RunName = 'plant_20261106_new',
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant',
    [switch]$DownloadOnly
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'Invalid run name.' }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$tag = [Guid]::NewGuid().ToString('N').Substring(0,8)
$localTool = Join-Path $PSScriptRoot 'export_crop_diagnostics.py'
$connection = "${ServerUser}@${ServerAddr}"
$remoteTool = "$RemoteRoot/incoming/crop_diagnostics_$tag.py"
$py = "$RemoteRoot/envs/plant_semantics/bin/python"
$sshOptions = @('-p',"$Port",'-o','ConnectTimeout=20')
$scpOptions = @('-P',"$Port",'-o','ConnectTimeout=20')
# Upload a standalone read-only helper, outside all pipeline source trees.
# This preserves the in-progress run's code signature and all stage outputs.
if (-not $DownloadOnly) {
    Invoke-Checked 'scp.exe' ($scpOptions + @($localTool,"${connection}:$remoteTool"))
    Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"'$py' -u '$remoteTool' --root '$RemoteRoot' --run '$RunName'"))
}
$latest = "$RemoteRoot/exports/crop_diagnostics/$RunName/latest.json"
$out = Join-Path $project "outputs\diagnostics\${RunName}_$tag"
[IO.Directory]::CreateDirectory($out) | Out-Null
# Parse the index locally. Windows PowerShell 5.1 strips embedded native-argv
# double quotes, so sending Python -c through SSH and Bash is unreliable.
$localIndex = Join-Path $out 'latest.json'
Invoke-Checked 'scp.exe' ($scpOptions + @("${connection}:$latest", $localIndex))
$index = Get-Content -LiteralPath $localIndex -Raw -Encoding UTF8 | ConvertFrom-Json
if ($index.status -ne 'CROP_DIAGNOSTICS_READY' -or $index.sha256 -notmatch '^[a-fA-F0-9]{64}$') {
    throw 'Diagnostics index is incomplete or invalid.'
}
$remoteArchive = [string]$index.archive
$expectedPrefix = "$RemoteRoot/exports/crop_diagnostics/$RunName/"
if (-not $remoteArchive.StartsWith($expectedPrefix, [StringComparison]::Ordinal) -or
    $remoteArchive.Substring($expectedPrefix.Length) -notmatch '^\d{8}_\d{6}_[a-f0-9]{6}/crop_diagnostics\.zip$') {
    throw 'Unexpected diagnostics archive path.'
}
Invoke-Checked 'scp.exe' ($scpOptions + @("${connection}:$remoteArchive", "${connection}:$remoteArchive.sha256", $out))
$archive = Join-Path $out 'crop_diagnostics.zip'
$expected = ((Get-Content -LiteralPath ($archive+'.sha256') -Raw).Trim() -split '\s+')[0]
if ($expected -ne $index.sha256 -or
    (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
    throw 'Diagnostics download SHA256 mismatch.'
}
Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $out 'extracted')
Write-Host "CROP_DIAGNOSTICS_DOWNLOADED: $out"
Write-Host 'Model, labels, pipeline state and completed stages were preserved. No reconstruction was started.'
