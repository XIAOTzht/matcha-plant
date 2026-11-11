param(
    [string]$ServerAddr = '10.7.88.77',
    [string]$ServerUser = 'fyp24_haoteng_zhang',
    [string]$RemoteRoot = '/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant',
    [string]$LocalReviewRoot = 'F:\plant\github_review',
    [switch]$DownloadOnly,
    [string]$ReviewName,
    [string]$ExpectedSha256
)
$ErrorActionPreference = 'Stop'
if ($ServerAddr -notmatch '^[A-Za-z0-9.-]+$' -or
    $ServerUser -notmatch '^[A-Za-z0-9_.-]+$' -or
    $RemoteRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteRoot.Contains('..')) {
    throw 'Invalid server destination'
}
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited $LASTEXITCODE" }
}
Get-Command ssh.exe, scp.exe -ErrorAction Stop | Out-Null
$exporter = Join-Path $PSScriptRoot 'export_github_source.py'
if (-not (Test-Path -LiteralPath $exporter)) { throw 'Export tool missing' }
if ($DownloadOnly) {
    if ($ReviewName -notmatch '^github_source_review_[0-9]{8}_[0-9]{6}_[a-f0-9]{6}$' -or
        $ExpectedSha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw 'DownloadOnly requires a valid ReviewName and the SHA256 from the server export output'
    }
    $name = $ReviewName
} else {
    $name = 'github_source_review_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' +
        ([guid]::NewGuid().ToString('N').Substring(0,6))
}
$remoteParent = $RemoteRoot.Substring(0,$RemoteRoot.LastIndexOf('/'))
$destination = "$remoteParent/$name"
$target = "$ServerUser@$ServerAddr"
$local = Join-Path $LocalReviewRoot $name
New-Item -ItemType Directory -Path $local -Force -ErrorAction Stop | Out-Null

# Copy only the exporter. Existing pipeline files and results are preserved.
if (-not $DownloadOnly) {
    Invoke-Checked 'scp.exe' @('-O', $exporter, "$($target):$RemoteRoot/maintenance/export_github_source.py")
    $command = "'$RemoteRoot/envs/plant_semantics/bin/python' " +
        "'$RemoteRoot/maintenance/export_github_source.py' --root '$RemoteRoot' --destination '$destination'"
    $serverOutput = @(& ssh.exe $target $command)
    if ($LASTEXITCODE -ne 0) { throw 'Server source export failed' }
    $serverOutput | ForEach-Object { Write-Host $_ }
    $exportLine = @($serverOutput | Where-Object { $_ -match '"status":\s*"GITHUB_SOURCE_EXPORTED"' })
    if ($exportLine.Count -ne 1) { throw 'Missing server export metadata' }
    $exportMetadata = $exportLine[0] | ConvertFrom-Json
    $ExpectedSha256 = $exportMetadata.sha256
    if ($ExpectedSha256 -notmatch '^[a-fA-F0-9]{64}$' -or $exportMetadata.destination -ne $destination) {
        throw 'Invalid server export metadata'
    }
    $exportMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $local 'server_export.json') -Encoding UTF8
}
$zip = Join-Path $local "$name.zip"
$expected = $ExpectedSha256
$alreadyVerified = (Test-Path -LiteralPath $zip) -and
    ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -eq $expected)
if (-not $alreadyVerified) {
    Invoke-Checked 'scp.exe' @('-O', "$($target):$destination.zip", $local)
}
$actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
if ($expected -notmatch '^[a-fA-F0-9]{64}$' -or $actual -ne $expected) {
    throw 'Downloaded archive SHA256 mismatch'
}
"$expected  $name.zip" | Set-Content -LiteralPath "$zip.sha256" -Encoding ASCII
Add-Type -AssemblyName System.IO.Compression.FileSystem
$extracted = Join-Path $local $name
if (Test-Path -LiteralPath $extracted) {
    throw "Extracted folder already exists: $extracted. Preserve it; review latest_review.json before retrying."
}
[System.IO.Compression.ZipFile]::ExtractToDirectory($zip,$local)
$result = [ordered]@{
    status='GITHUB_SOURCE_REVIEW_DOWNLOADED'
    server_snapshot=$destination
    local_source=(Join-Path $local $name)
    archive=$zip
    sha256=$actual.ToLowerInvariant()
    git_initialized=$false
    pushed=$false
}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LocalReviewRoot 'latest_review.json') -Encoding UTF8
$result | ConvertTo-Json
Write-Host 'Source review downloaded. No Git initialization, commit or push was performed.'
