param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [string]$OutputDirectory = '',
    [switch]$Extract,
    [switch]$DownloadOnly,
    [switch]$LegacyScp,
    [ValidateRange(1,5)][int]$Attempts = 2,
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [int]$Port = 22,
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant'
)
. (Join-Path $PSScriptRoot 'plant_pipeline_client.ps1')
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'Invalid run name.' }
Assert-RemoteArguments $ServerAddr $ServerUser $RemoteRoot $Port
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $PSScriptRoot "..\outputs\downloads\$RunName" }
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$archive = Join-Path $OutputDirectory 'results.zip'
$connection = "${ServerUser}@${ServerAddr}"
if (-not $DownloadOnly) {
    Invoke-Checked 'ssh.exe' @('-p',"$Port",'-o','ConnectTimeout=20',$connection,"'$RemoteRoot/envs/plant_semantics/bin/python' '$RemoteRoot/scripts/plant_pipeline/photos.py' export --root '$RemoteRoot' --run '$RunName'")
}
$scpOptions = @('-P',"$Port",'-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
if ($LegacyScp) { $scpOptions = @('-O') + $scpOptions }
function Receive-ResultFile {
    param([string]$RemoteFile, [string]$Destination)
    for ($attempt=1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-Checked 'scp.exe' ($scpOptions + @("${connection}:$RemoteFile",$Destination))
            return
        } catch {
            if ($attempt -eq $Attempts) { throw }
            Write-Host "Transfer connection failed; retry $($attempt+1)/$Attempts. Server computation is unaffected."
            Start-Sleep -Seconds 2
        }
    }
}
# Interrupted transfers stay in their own directory. Only verified files are
# published as results.zip, so the same command can be retried after failure.
$transfer = Join-Path $OutputDirectory ('.download_' + [Guid]::NewGuid().ToString('N').Substring(0,8))
[IO.Directory]::CreateDirectory($transfer) | Out-Null
$checksum = Join-Path $transfer 'results.zip.sha256'
Receive-ResultFile "$RemoteRoot/exports/$RunName/results.zip.sha256" $checksum
$checksumText = (Get-Content -LiteralPath $checksum -Raw).Trim()
if ($checksumText -notmatch '^([a-fA-F0-9]{64})\s+\*?results\.zip$') { throw 'Invalid results checksum file.' }
$expected = $Matches[1].ToLowerInvariant()
if (Test-Path -LiteralPath $archive) {
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw "Preserving existing results.zip with a different checksum. Choose a new -OutputDirectory: $archive"
    }
    Write-Host "Reusing verified existing archive: $archive"
} else {
    $partial = Join-Path $transfer 'results.zip'
    Receive-ResultFile "$RemoteRoot/exports/$RunName/results.zip" $partial
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw "Download SHA256 mismatch. Partial download preserved in $transfer. Retry the same command."
    }
    Move-Item -LiteralPath $partial -Destination $archive
}
Copy-Item -LiteralPath $checksum -Destination ($archive+'.sha256') -Force
Remove-Item -LiteralPath $checksum
Remove-Item -LiteralPath $transfer
Write-Host "RESULT_DOWNLOAD_VERIFIED: $archive"
if ($Extract) {
    $extractDirectory = Join-Path $OutputDirectory 'extracted'
    if (Test-Path -LiteralPath $extractDirectory) {
        throw "Preserving existing extraction: $extractDirectory. Choose a new OutputDirectory."
    }
    $extractStaging = Join-Path $OutputDirectory ('.extract_' + [Guid]::NewGuid().ToString('N').Substring(0,8))
    Expand-Archive -LiteralPath $archive -DestinationPath $extractStaging
    $index = Join-Path $extractStaging "outputs\pipeline\$RunName\result_paths.json"
    if (-not (Test-Path -LiteralPath $index -PathType Leaf)) {
        throw "Extracted archive is missing the result index: $index"
    }
    Move-Item -LiteralPath $extractStaging -Destination $extractDirectory
    $index = Join-Path $extractDirectory "outputs\pipeline\$RunName\result_paths.json"
    Write-Host "RESULTS_EXTRACTED: $extractDirectory"
    Write-Host "RESULT_INDEX: $index"
}
