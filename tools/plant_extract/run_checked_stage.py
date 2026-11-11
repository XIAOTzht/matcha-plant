#!/usr/bin/env python3
"""Make MAtCha stage wrappers propagate their child command failures."""
import os
from pathlib import Path
import runpy
import subprocess
import sys

script=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
sys.path.insert(0,str(script.parent.parent))

def checked_system(command):
    subprocess.run(command,shell=True,check=True)
    return 0

os.system=checked_system
runpy.run_path(str(script),run_name='__main__')
