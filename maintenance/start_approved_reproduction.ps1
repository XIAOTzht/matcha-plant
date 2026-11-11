param(
    [string]$Revision = 'approved_20261107_reproduced_v1',
    [string]$Archive = '',
    [string]$PythonExe = '',
    [switch]$CheckOnly,
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
if ($Revision -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'Invalid new revision name.' }
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Get-ClientPython $project $PythonExe
$tool = Join-Path $PSScriptRoot 'approved_reproduction.py'
if (-not $Archive) {
    $verified = Join-Path $project 'maintenance\packages\approved_reproduction_20261109_v1.zip'
    if ((Test-Path -LiteralPath $verified -PathType Leaf) -and
        (Get-FileHash -LiteralPath $verified -Algorithm SHA256).Hash.ToLowerInvariant() -eq 'ba11bfd2e220b0551458268d5b43f397cfa1cb3676b92503bbfec1d60e869431') {
        $Archive = $verified
    } else {
        $tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
        $Archive = Join-Path $project "maintenance\packages\approved_reproduction_$tag.zip"
        Invoke-Checked $python @($tool,'build','--output',$Archive)
    }
}
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Missing bundle: $Archive" }
$hash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Reproduction INPUT + CODE bundle: $Archive"
Write-Host "SHA256: $hash"
if ($CheckOnly) { Write-Host 'LOCAL_REPRODUCTION_PACKAGE_READY (no server connection)'; return }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
& (Join-Path $PSScriptRoot 'upload_plant_pipeline.ps1') -ServerAddr $ServerAddr -ServerUser $ServerUser -Port $Port -RemoteRoot $RemoteRoot -PythonExe $python
$connection = "${ServerUser}@${ServerAddr}"
$stage = "$RemoteRoot/incoming/approved_reproduction_$Revision"
$remotePython = "$RemoteRoot/envs/plant_semantics/bin/python"
$remoteSim = "$RemoteRoot/envs/mujoco/bin/python"
$out = "$RemoteRoot/outputs/reproduction/$Revision"
$sshOptions = @('-p',"$Port",'-o','ConnectTimeout=20')
$scpOptions = @('-O','-P',"$Port",'-o','ConnectTimeout=20')
Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"test -x '$remotePython' && test -x '$remoteSim' && test ! -e '$out' && mkdir -p '$stage' '$RemoteRoot/logs'"))
$files = @($Archive,$tool,(Join-Path $PSScriptRoot 'reproduce_approved_semantics.py'))
Invoke-Checked 'scp.exe' ($scpOptions + $files + @("${connection}:${stage}/"))
$name = Split-Path -Leaf $Archive
# Launch from the existing project maintenance directory; originals are backed up.
$backup = "$RemoteRoot/backups/code_updates/approved_launcher_$Revision"
$install = "mkdir -p '$backup'"
foreach ($file in @('approved_reproduction.py','reproduce_approved_semantics.py')) {
    $install += " && if [ -f '$RemoteRoot/maintenance/$file' ]; then cp '$RemoteRoot/maintenance/$file' '$backup/'; fi"
    $install += " && cp '$stage/$file' '$RemoteRoot/maintenance/$file'"
}
$job = "'$remotePython' -u '$RemoteRoot/maintenance/approved_reproduction.py' run --archive '$stage/$name' --sha256 '$hash' --revision '$Revision' --sim-python '$remoteSim' --submit"
$log = "$RemoteRoot/logs/reproduction_$Revision.log"
$launch = "$install && $job"
Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,$launch))
Write-Host "APPROVED_REPRODUCTION_SUBMITTED: $Revision"
Write-Host "Log: $log"
Write-Host "Results: $out/result_paths.json"
Write-Host 'Wait for APPROVED_REPRODUCTION_SUCCESS; this replays saved upstream evidence and the approved path.'
