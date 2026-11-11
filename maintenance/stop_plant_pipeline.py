"""Stop only one run's orchestrator and descendants; keep every result and log."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts/plant_pipeline'))
from common import JobLock, run_name


def processes():
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            # comm in /proc/PID/stat may contain spaces or parentheses.
            stat = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(entry.name)] = dict(ppid=int(stat[1]), state=stat[0], birth=stat[19],
                argv=[v.decode(errors='replace') for v in (entry/'cmdline').read_bytes().split(b'\0') if v])
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return rows


def run_roots(rows, root, run):
    script = str(root/'scripts/plant_pipeline/run_pipeline.py')
    return {pid for pid, row in rows.items() if script in row['argv'] and any(
        row['argv'][i:i+2] == ['--run', run] for i in range(len(row['argv'])-1))}


def descendants(rows, roots):
    selected = set(roots)
    while True:
        expanded = selected | {pid for pid, row in rows.items() if row['ppid'] in selected}
        if expanded == selected:
            return selected
        selected = expanded


def alive(tracked):
    rows = processes()
    return {pid for pid, birth in tracked.items()
            if pid in rows and rows[pid]['birth'] == birth and rows[pid]['state'] != 'Z'}


def stop(root, run):
    root = Path(root).resolve()
    rows = processes()
    roots = run_roots(rows, root, run)
    tracked = {pid: rows[pid]['birth'] for pid in descendants(rows, roots)}
    print('PIPELINE_STOP_TARGET', run, 'pids='+','.join(map(str, sorted(tracked))), flush=True)
    for sig, seconds in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 3)):
        remaining = alive(tracked)
        if not remaining:
            break
        # Signal the captured descendants too: a shell waiting for a Python
        # worker does not reliably forward SIGINT or SIGTERM.
        for pid in remaining:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline and alive(tracked):
            time.sleep(.25)
    remaining = alive(tracked)
    if remaining:
        raise RuntimeError(f'Run still has processes (possibly waiting for NFS I/O): {sorted(remaining)}; do not restart yet')
    if run_roots(processes(), root, run):
        raise RuntimeError('A new instance of this run appeared; refusing to restart automatically')
    session = 'plant_'+run
    # Only close an empty/completed pipeline pane. If it now contains an
    # unrelated foreground task, leave it alone.
    panes = subprocess.run(['tmux', 'list-panes', '-t', '='+session, '-F', '#{pane_pid}'],
                           capture_output=True, text=True)
    if panes.returncode == 0:
        current = processes()
        pane_pids = {int(value) for value in panes.stdout.split() if value.isdigit()}
        children = descendants(current, pane_pids)-pane_pids
        if any(current[pid]['state'] != 'Z' for pid in children):
            raise RuntimeError(f'Tmux session {session} still has other child processes; preserved')
        subprocess.run(['tmux', 'kill-session', '-t', '='+session], check=True)
    # Safely recovers this host's dead owner record through JobLock; never
    # remove the lock manually or override a different live run.
    with JobLock(root/'logs/plant_pipeline.lock'):
        print('PIPELINE_STOPPED', run, 'results and logs preserved; lock available', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=PROJECT)
    ap.add_argument('--run', type=run_name, required=True)
    args = ap.parse_args()
    if os.name != 'posix':
        ap.error('This server process helper requires Linux')
    stop(args.root, args.run)
