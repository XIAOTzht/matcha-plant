param(
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant',
    [string]$PythonExe = '',
    [switch]$CheckOnly
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Get-ClientPython $project $PythonExe
$tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$archive = Join-Path $project "maintenance\packages\plant_pipeline_$tag.zip"
$tool = Join-Path $project 'maintenance\plant_pipeline_bundle.py'
Invoke-Checked $python @($tool,'build','--root',$project,'--output',$archive)
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Invoke-Checked $python @($tool,'inspect','--archive',$archive,'--sha256',$hash)
Write-Host "Verified package: $archive"
if ($CheckOnly) { Write-Host 'LOCAL_DEPLOYMENT_CHECK_OK (no server connection)'; return }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
$connection = "${ServerUser}@${ServerAddr}"
$stage = "$RemoteRoot/incoming/plant_pipeline_$tag"
$remotePython = "$RemoteRoot/envs/plant_semantics/bin/python"
$sshOptions = @('-p',"$Port",'-o','ConnectTimeout=20')
$scpOptions = @('-P',"$Port",'-o','ConnectTimeout=20')
Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"test -d '$RemoteRoot/MAtCha' && test -x '$remotePython' && mkdir -p '$stage'"))
Invoke-Checked 'scp.exe' ($scpOptions + @($archive,($archive+'.sha256'),$tool,"${connection}:${stage}/"))
$name = Split-Path -Leaf $archive
Invoke-Checked 'ssh.exe' ($sshOptions + @($connection,"'$remotePython' '$stage/plant_pipeline_bundle.py' install --archive '$stage/$name' --sha256 '$hash' --root '$RemoteRoot'"))
Write-Host 'CODE_UPLOAD_INSTALLED. Run once on server:'
Write-Host "bash '$RemoteRoot/maintenance/setup_plant_pipeline.sh'"
