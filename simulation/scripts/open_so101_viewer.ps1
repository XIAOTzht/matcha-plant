$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
$PY = Join-Path $ROOT 'envs\mujoco\Scripts\python.exe'
$SCENE = Join-Path $ROOT 'assets\robot\robotstudio_so101\scene.xml'
if (!(Test-Path -LiteralPath $PY)) { throw "找不到 MuJoCo Python：$PY" }
if (!(Test-Path -LiteralPath $SCENE)) { throw "找不到 SO-101 场景：$SCENE" }
& $PY -m mujoco.viewer --mjcf="$SCENE"
