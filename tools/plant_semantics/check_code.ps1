<#
.SYNOPSIS
Check project PowerShell syntax and run the geometry/distribution regression tests.
#>
param([string]$PythonExecutable='')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$PythonExecutable=Resolve-PlantPython $PythonExecutable
foreach ($file in (Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1' -File)) {
  $parseTokens=$null
  $parseErrors=$null
  [System.Management.Automation.Language.Parser]::ParseFile($file.FullName,[ref]$parseTokens,[ref]$parseErrors) | Out-Null
  if ($parseErrors.Count -gt 0) { throw "PowerShell syntax error in $($file.Name): $parseErrors" }
}
& $PythonExecutable -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }
Write-Host 'PowerShell syntax and regression tests passed.'
