# Shared runtime discovery. runtime.local.json is machine-specific and is not
# distributed in the source archive. No application needs to be running.
function Resolve-PlantPython {
  param([string]$Requested='')
  $root=$PSScriptRoot
  $candidates=@()
  if ($Requested) { $candidates=@($Requested) }
  else {
    if ($env:PLANTSEG_PYTHON) { $candidates+= $env:PLANTSEG_PYTHON }
    $candidates+= (Join-Path $root '.venv\Scripts\python.exe')
    $candidates+= (Join-Path $root '..\..\.venv\Scripts\python.exe')
    $localConfig=Join-Path $root 'runtime.local.json'
    if (Test-Path -LiteralPath $localConfig -PathType Leaf) {
      $config=Get-Content -LiteralPath $localConfig -Raw | ConvertFrom-Json
      if ($config.python) { $candidates+= $config.python }
    }
    foreach ($name in @('python','python3')) {
      $command=Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
      if ($command) { $candidates+= $command.Source }
    }
    $launcher=Get-Command py -CommandType Application -ErrorAction SilentlyContinue
    if ($launcher) {
      foreach ($version in @('-3.12','-3.11','-3.10')) {
        $found=& $launcher.Source $version -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { $candidates+= $found }
      }
    }
  }
  foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (-not $candidate) { continue }
    # Ignore Windows Store launch aliases, which open an installer instead.
    if ($candidate -like '*\Microsoft\WindowsApps\*') { continue }
    $command=Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
    if (-not $command) { continue }
    $actual=& $command.Source -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12); print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $actual) { return [string]($actual | Select-Object -Last 1) }
  }
  throw 'Python 3.10-3.12 was not found. Pass -PythonExecutable C:\path\python.exe or run setup_local.ps1 with that option.'
}

function Resolve-PlantSam {
  param([string]$Requested='')
  $target=$Requested
  if (-not $target) { $target=Join-Path $PSScriptRoot 'models\mobile_sam_parallel.pt' }
  if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "MobileSAM checkpoint not found: $target. Run setup_local.ps1 -DownloadModels or pass -SamCheckpoint."
  }
  return (Resolve-Path -LiteralPath $target).Path
}
