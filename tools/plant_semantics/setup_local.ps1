param(
  [string]$PythonExecutable='',
  [switch]$InstallDependencies,
  [switch]$DownloadModels
)
$ErrorActionPreference='Stop'
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
if ($InstallDependencies) {
  & $PythonExecutable -m pip install -r (Join-Path $scriptRoot 'requirements.txt')
  if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
if ($DownloadModels) {
  & $PythonExecutable (Join-Path $scriptRoot 'fetch_models.py')
  if ($LASTEXITCODE -ne 0) { throw 'Detector or MobileSAM download failed' }
  & $PythonExecutable (Join-Path $scriptRoot 'fetch_models.py') --clipseg
  if ($LASTEXITCODE -ne 0) { throw 'CLIPSeg download failed' }
}
& $PythonExecutable (Join-Path $scriptRoot 'check_install.py')
if ($LASTEXITCODE -ne 0) { throw 'Runtime or model assets are incomplete' }
@{python=$PythonExecutable} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scriptRoot 'runtime.local.json') -Encoding UTF8
Write-Host "Saved Python runtime: $PythonExecutable"
