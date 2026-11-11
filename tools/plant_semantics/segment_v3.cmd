@echo off
setlocal
if "%~1"=="" (
  echo Drag a colored triangle mesh PLY onto this file.
  echo Or run: segment_v3.cmd "D:\plants\new.ply"
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_v3.ps1" -InputMesh "%~f1" -OutputDirectory "%~dpn1_v3" -UpAxis auto
set "segment_exit=%ERRORLEVEL%"
if not "%segment_exit%"=="0" echo V3 segmentation failed. See the error above.
if "%segment_exit%"=="0" echo Result paths: %~dpn1_v3\result_paths.txt
pause
exit /b %segment_exit%
