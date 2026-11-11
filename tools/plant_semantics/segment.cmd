@echo off
setlocal
if "%~1"=="" (
  echo Drag a colored triangle mesh PLY onto this file.
  echo Or run: segment.cmd "D:\plants\new.ply"
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_auto.ps1" -InputMesh "%~f1" -OutputDirectory "%~dpn1_semantic" -UpAxis auto
set "segment_exit=%ERRORLEVEL%"
if not "%segment_exit%"=="0" echo Segmentation failed. See the error above.
if "%segment_exit%"=="0" echo Output: %~dpn1_semantic\complete\semantic_mesh.ply
pause
exit /b %segment_exit%
