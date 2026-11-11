param(
    [Parameter(Mandatory=$true)][string]$ImageDirectory,
    [string]$RunName = ('plant_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [int]$GpuId = 1,
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant',
    [string]$PythonExe = '',
    [switch]$CheckOnly,
    [switch]$UploadOnly,
    [switch]$Resume
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$' -or $GpuId -lt 0) { throw 'Invalid run name or GPU index.' }
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Get-ClientPython $project $PythonExe
$tag = [Guid]::NewGuid().ToString('N').Substring(0,8)
$archive = Join-Path $project "incoming\photo_uploads\${RunName}_$tag.zip"
$tool = Join-Path $project 'scripts\plant_pipeline\photos.py'
Invoke-Checked $python @($tool,'prepare','--images',$ImageDirectory,'--run',$RunName,'--output',$archive)
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($CheckOnly) { Write-Host "PHOTO_UPLOAD_CHECK_OK: $archive"; return }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
$connection = "${ServerUser}@${ServerAddr}"
$stage = "$RemoteRoot/incoming/photo_$RunName"
$remotePython = "$RemoteRoot/envs/plant_semantics/bin/python"
$remoteTool = "$RemoteRoot/scripts/plant_pipeline/photos.py"
$sshOptions = @('-p',"$Port",'-o','ConnectTimeout=20')
$scpOptions = @('-P',"$Port",'-o','ConnectTimeout=20')
Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"test -f '$remoteTool' && mkdir -p '$stage'"))
Invoke-Checked 'scp.exe' ($scpOptions + @($archive,($archive+'.sha256'),"${connection}:${stage}/"))
$name = Split-Path -Leaf $archive
Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"'$remotePython' '$remoteTool' ingest --archive '$stage/$name' --sha256 '$hash' --root '$RemoteRoot' --run '$RunName'"))
if (-not $UploadOnly) {
    $resumeFlag = ''; if ($Resume) { $resumeFlag = ' --resume' }
    Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"'$remotePython' '$remoteTool' start --root '$RemoteRoot' --run '$RunName' --gpu $GpuId$resumeFlag"))
}
Write-Host "Run: $RunName"
Write-Host "Monitor on server: tail -f '$RemoteRoot/logs/pipeline_$RunName.log'"
Write-Host "Results: $RemoteRoot/outputs/pipeline/$RunName/result_paths.json"
