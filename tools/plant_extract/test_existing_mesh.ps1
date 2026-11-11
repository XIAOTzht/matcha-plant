$ErrorActionPreference = 'Stop'
$ServerConnection = 'fyp24_haoteng_zhang@10.7.88.77'
$WorkspaceDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$InputMesh = Join-Path $WorkspaceDir 'outputs\tetra_mesh_binary_search_7.ply'
$ImageDir = Join-Path $WorkspaceDir 'data\test'
$TestRoot = Join-Path $WorkspaceDir ('outputs\plant_auto_test_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$SfmDir = Join-Path $TestRoot 'mast3r_sfm'
$SparseDir = Join-Path $SfmDir 'sparse\0'
$ResultDir = Join-Path $TestRoot 'result'
$ExpectedHash = (Get-FileHash -LiteralPath $InputMesh -Algorithm SHA256).Hash.ToLowerInvariant()
$InputSize = (Get-Item -LiteralPath $InputMesh).Length
$RuntimePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $RuntimePython) {
    $PythonExe = $RuntimePython
} elseif (Test-Path -LiteralPath 'D:\devtools\Anaconda3-CodeTou\python.exe') {
    $PythonExe = 'D:\devtools\Anaconda3-CodeTou\python.exe'
} else {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

New-Item -ItemType Directory -Path $SparseDir -Force | Out-Null
Write-Host 'Locating the identical server mesh by size and SHA256; this does not modify server files.'
$RemoteProbe = @'
import hashlib, json
from pathlib import Path
root = Path('/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant/outputs')
matches = []
for mesh in root.rglob('tetra_mesh_binary_search_7.ply'):
    if mesh.stat().st_size != EXPECTED_SIZE:
        continue
    h = hashlib.sha256()
    with mesh.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != 'EXPECTED_HASH':
        continue
    for parent in list(mesh.parents)[:4]:
        if parent == root.parent:
            break
        sfm = parent / 'mast3r_sfm' / 'sparse' / '0'
        for suffix in ('bin', 'txt'):
            if (sfm / ('cameras.' + suffix)).is_file() and (sfm / ('images.' + suffix)).is_file():
                matches.append({'mesh': str(mesh), 'sparse': str(sfm), 'suffix': suffix})
                break
        else:
            continue
        break
if not matches:
    raise SystemExit('No identical mesh with matching mast3r_sfm was found. Do not use unrelated cameras.')
print(json.dumps(matches[0]))
'@
$RemoteProbe = $RemoteProbe.Replace('EXPECTED_SIZE', [string]$InputSize).Replace('EXPECTED_HASH', $ExpectedHash)
$RemoteInfoRaw = $RemoteProbe | ssh -o ConnectTimeout=15 $ServerConnection 'python3 -'
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate matching camera metadata. Confirm VPN/network and server access.' }
$RemoteInfo = ($RemoteInfoRaw -join "`n") | ConvertFrom-Json
if ($RemoteInfo.sparse -notmatch '^/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant/outputs/[A-Za-z0-9_./-]+$') {
    throw 'Unexpected server metadata path; stopped before copying.'
}
if ($RemoteInfo.suffix -notin @('bin','txt')) { throw 'Unexpected camera file format' }
foreach ($CameraFile in @('cameras','images')) {
    $RemoteFile = $RemoteInfo.sparse + '/' + $CameraFile + '.' + $RemoteInfo.suffix
    scp "${ServerConnection}:${RemoteFile}" $SparseDir
    if ($LASTEXITCODE -ne 0) { throw "Camera download failed: $CameraFile" }
}

# Validate that the available local photos have exactly the names and dimensions
# associated with these cameras. Auto masking will use these original photos.
$ValidatePhotos = @'
import json, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, sys.argv[1])
import auto_extract_plant as app
views, _ = app.read_colmap(Path(sys.argv[2]))
folder = Path(sys.argv[3])
for v in views:
    path = app.image_file(folder, v.name)
    with Image.open(path) as im:
        if im.size != (v.width, v.height):
            raise SystemExit(f'Photo dimensions differ from SfM: {path}; {im.size} != {(v.width,v.height)}')
print(f'Validated {len(views)} photos against matching camera metadata')
'@
$ValidatePhotos | & $PythonExe - $PSScriptRoot $SfmDir $ImageDir
if ($LASTEXITCODE -ne 0) { throw 'Local data/test photos do not match. Stop and identify the original photos.' }

$RemoteInfo | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $TestRoot 'matched_source.json') -Encoding utf8
& $PythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt') 'rembg[cpu]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed' }

Write-Host 'Generating photo masks and extracting the plant and pot on the CPU...'
$LogFile = Join-Path $TestRoot 'extract.log'
& $PythonExe (Join-Path $PSScriptRoot 'auto_extract_plant.py') `
    --mesh $InputMesh `
    --sfm-dir $SfmDir `
    --images-dir $ImageDir `
    --output-dir $ResultDir 2>&1 | Tee-Object -FilePath $LogFile
if ($LASTEXITCODE -ne 0) { throw "Extraction failed; inspect $LogFile" }
Write-Host "Completed: $ResultDir\plant.ply"
Write-Host "Preview: $ResultDir\plant_preview.jpg"
