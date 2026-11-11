param(
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$ProjectRoot = '',
    [string]$PythonExe = '',
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant',
    [string]$RemotePython = '',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
}
if ($ServerAddr -notmatch '^[a-zA-Z0-9.-]+$' -or $ServerUser -notmatch '^[a-zA-Z0-9_.-]+$') {
    throw 'ServerAddr/ServerUser contain unsupported characters.'
}
if ($RemoteRoot -notmatch '^/[a-zA-Z0-9_./-]+$' -or $RemoteRoot.Contains('..')) {
    throw 'RemoteRoot must be an absolute Linux path without spaces or parent segments.'
}
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Invalid SSH port.' }
if ([string]::IsNullOrWhiteSpace($RemotePython)) {
    $RemotePython = "$RemoteRoot/envs/plant_semantics/bin/python"
}
if ($RemotePython -notmatch '^/[a-zA-Z0-9_./-]+$' -or $RemotePython.Contains('..')) {
    throw 'RemotePython must be an absolute Linux executable path.'
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidates = @()
    if ($env:PLANT_CONTACT_PYTHON) { $candidates += $env:PLANT_CONTACT_PYTHON }
    $candidates += (Join-Path $ProjectRoot 'envs\plant_semantics\Scripts\python.exe')
    $candidates += (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $PythonExe = $candidate; break }
    }
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        throw 'Python not found. Supply -PythonExe with an installed Python 3.10+ executable.'
    }
}
function Invoke-Native {
    param([string]$Executable, [string[]]$ArgumentList)
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
$bundleTool = Join-Path $ProjectRoot 'maintenance\plant_contact_bundle.py'
$checker = Join-Path $ProjectRoot 'scripts\plant_contact\check_environment.py'
if (-not (Test-Path -LiteralPath $bundleTool -PathType Leaf)) { throw "Missing file: $bundleTool" }
Invoke-Native -Executable $PythonExe -ArgumentList @($checker)
$tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$packageDir = Join-Path $ProjectRoot 'maintenance\packages'
[IO.Directory]::CreateDirectory($packageDir) | Out-Null
$archive = Join-Path $packageDir ("plant_contact_code_{0}.zip" -f $tag)
Write-Host 'Building source-only ZIP (no models, environments or results)...'
Invoke-Native -Executable $PythonExe -ArgumentList @($bundleTool,'build','--root',$ProjectRoot,'--output',$archive)
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Invoke-Native -Executable $PythonExe -ArgumentList @($bundleTool,'inspect','--archive',$archive,'--sha256',$hash)
Write-Host "Source archive: $archive"
if ($CheckOnly) {
    Write-Host 'PLANT_CONTACT_UPLOAD_CHECK_OK: package verified; no server connection made.'
    return
}
Get-Command ssh.exe,scp.exe -ErrorAction Stop | Out-Null
$connection = "${ServerUser}@${ServerAddr}"
$remoteStage = "$RemoteRoot/incoming/plant_contact_$tag"
$remoteArchive = "$remoteStage/$(Split-Path -Leaf $archive)"
$remoteTool = "$remoteStage/plant_contact_bundle.py"
$sshOptions = @('-p',[string]$Port,'-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
$scpOptions = @('-P',[string]$Port,'-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
Invoke-Native -Executable 'ssh.exe' -ArgumentList ($sshOptions + @($connection,"test -d '$RemoteRoot' && test -x '$RemotePython' && mkdir -p '$remoteStage'"))
$scpArguments = $scpOptions + @($archive,($archive + '.sha256'),$bundleTool,"${connection}:${remoteStage}/")
Invoke-Native -Executable 'scp.exe' -ArgumentList $scpArguments
$remoteCommand = "'$RemotePython' '$remoteTool' install --archive '$remoteArchive' --sha256 '$hash' --root '$RemoteRoot' && '$RemotePython' '$RemoteRoot/scripts/plant_contact/check_environment.py'"
Invoke-Native -Executable 'ssh.exe' -ArgumentList ($sshOptions + @($connection,$remoteCommand))
Write-Host 'PLANT_CONTACT_UPLOAD_INSTALLED: code installed and dependency imports verified.'
Write-Host 'Existing model/results were preserved; no pipeline job was started.'
Write-Host "Server guide: $RemoteRoot/docs/PLANT_CONTACT_GUIDE.md"
