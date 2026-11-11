#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PY="${PLANT_CONTACT_PYTHON:-$ROOT/envs/plant_semantics/bin/python}"
SCENE="$ROOT/outputs/with_car_20261021"
INPUT="$SCENE/plant_car_semantics_v2"
OUTPUT="$SCENE/contact_latest_$(date +%Y%m%d_%H%M%S)_$$"
bash "$SCRIPT_DIR/run_server.sh" "$INPUT" "$OUTPUT" "$@"
for arg in "$@"; do [[ "$arg" == "--check-only" ]] && exit 0; done
[[ -f "$OUTPUT/PLANT_CONTACT_EXPORT_SUCCESS" ]] || exit 1
"$PY" - "$SCENE" "$OUTPUT" <<'PY'
import json, os, sys
from pathlib import Path
scene, output = map(Path, sys.argv[1:])
paths = json.loads((output/"result_paths.json").read_text(encoding="utf-8"))
pointer = dict(output=str(output), relative_output=output.relative_to(scene).as_posix(), paths=paths)
tmp=scene/("latest_contact.json.tmp_"+str(os.getpid()))
tmp.write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
os.replace(tmp,scene/"latest_contact.json")
tmp=scene/("latest_contact_paths.txt.tmp_"+str(os.getpid()))
tmp.write_text((output/"result_paths.txt").read_text(encoding="utf-8"),encoding="utf-8")
os.replace(tmp,scene/"latest_contact_paths.txt")
print("LATEST_CONTACT_UPDATED:",output,flush=True)
PY
