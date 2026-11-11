"""Standard-library job state, file manifests and failure-safe stage execution."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_signature(root, overrides=None, omitted=()):
    """Invalidate resumed jobs when numerical pipeline code/config changes."""
    root = Path(root)
    folders = ['scripts/plant_pipeline', 'scripts/plant_contact', 'tools/contact_weights',
               'tools/plant_organ_instances', 'tools/plant_extract_with_car', 'tools/plant_semantics_with_car',
               'simulation/scripts', 'MAtCha/scripts', 'MAtCha/configs']
    files = [p for d in folders for p in (root/d).rglob('*')
             if p.is_file() and p.suffix in {'.py','.sh','.json','.yaml'} and '__pycache__' not in p.parts
             and not p.name.startswith('test_')]
    files += [p for p in (root/'tools/plant_semantics').glob('*') if p.suffix in {'.py','.sh'}]
    h = hashlib.sha256()
    overrides = {} if overrides is None else overrides
    for p in sorted(files):
        name = p.relative_to(root).as_posix()
        if name in omitted:
            continue
        h.update(name.encode()); h.update(overrides.get(name, digest(p)).encode())
    return h.hexdigest()


def run_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Run name must be 1..64 ASCII letters/digits/_/- and begin with a letter/digit')
    return value


def confined(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    target.relative_to(root)
    if target == root:
        raise ValueError('Expected a file/directory below root')
    return target


def verify_images(data):
    data = Path(data)
    manifest = read_json(data / 'input_manifest.json')
    rows = manifest['images']
    if len(rows) < 3 or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('At least three distinct photos required')
    actual = {p.relative_to(data).as_posix() for p in (data / 'images').iterdir() if p.is_file()}
    if actual != {r['path'] for r in rows}:
        raise ValueError('Image manifest and folder disagree')
    for row in rows:
        p = confined(data, row['path'])
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'} or digest(p) != row['sha256']:
            raise ValueError(f'Image checksum/extension mismatch: {row["path"]}')
    return manifest


class JobLock:
    """NFS-safe single-job lock.

    ``flock`` is unavailable on some project NFS mounts (errno 37).  Atomic
    directory creation is the portable primitive here.  The owner record lets
    us reclaim a lock left by a crashed process while refusing to remove a
    live process's lock.
    """

    stale_after_seconds = 6 * 60 * 60

    def __init__(self, path):
        self.path = Path(path)
        self.directory = self.path.with_name(self.path.name + '.d')
        self.owner_path = self.directory / 'owner.json'
        self.token = uuid.uuid4().hex

    @staticmethod
    def _same_host(owner):
        return owner.get('hostname') == socket.gethostname()

    @staticmethod
    def _pid_alive(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        if os.name == 'nt':
            # Windows os.kill(pid, 0) broadcasts CTRL_C_EVENT; it is not the
            # harmless POSIX liveness check. Query a process handle instead.
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such PID
            try:
                code = wintypes.DWORD()
                return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _owner_is_live(self, owner):
        if self._same_host(owner):
            return self._pid_alive(owner.get('pid'))
        acquired = owner.get('acquired_epoch')
        return acquired is None or time.time() - float(acquired) < self.stale_after_seconds

    def _reclaim_if_stale(self):
        try:
            owner = json.loads(self.owner_path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            owner = None
        if owner and self._owner_is_live(owner):
            return False
        if owner is None:
            try:
                age = time.time() - self.directory.stat().st_mtime
            except FileNotFoundError:
                return True
            # A newly-created directory with no owner record may belong to a
            # process that is between mkdir and writing metadata; never reclaim it.
            if age < self.stale_after_seconds:
                return False
        if os.name == 'nt':
            try:
                shutil.rmtree(self.directory)
            except FileNotFoundError:
                return True
            return True
        stale = self.directory.with_name(self.directory.name + '.stale_' + uuid.uuid4().hex[:10])
        try:
            os.replace(self.directory, stale)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        shutil.rmtree(stale, ignore_errors=True)
        return True

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        for _ in range(3):
            try:
                self.directory.mkdir()
            except FileExistsError:
                if self._reclaim_if_stale():
                    continue
                raise RuntimeError(f'Another job holds {self.path}') from None
            owner = dict(pid=os.getpid(), hostname=socket.gethostname(),
                         acquired_utc=now(), acquired_epoch=time.time(), token=self.token)
            try:
                write_json(self.owner_path, owner)
            except BaseException:
                shutil.rmtree(self.directory, ignore_errors=True)
                raise
            return self
        raise RuntimeError(f'Another job holds {self.path}') from None

    def __exit__(self, *unused):
        try:
            owner = read_json(self.owner_path)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            owner = None
        if owner and owner.get('token') == self.token:
            self.owner_path.unlink(missing_ok=True)
            try:
                self.directory.rmdir()
            except OSError:
                # Leave unexpected contents for inspection; never delete a
                # directory that may have been replaced by another claimant.
                pass


class Stages:
    def __init__(self, root, signature, compatible_code_signatures=()):
        self.root = Path(root)
        self.path = self.root / 'pipeline_state.json'
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = read_json(self.path) if self.path.exists() else dict(signature=signature, stages={}, created_utc=now())
        if self.state['signature'] != signature:
            old = self.state['signature']
            adjusted = dict(old, code_sha256=signature.get('code_sha256'))
            if old.get('code_sha256') not in compatible_code_signatures or adjusted != signature:
                raise ValueError('Inputs/config/code changed. Use a new RunName; completed steps cannot be reused.')
            # This exception is limited to an exact, reviewed runtime-only
            # patch. Validate every successful stage before updating the state.
            for entry in self.state['stages'].values():
                if entry.get('status') == 'SUCCESS':
                    for item in entry['artifacts'].values():
                        p = Path(item['path'])
                        if not p.is_file() or digest(p) != item['sha256']:
                            raise ValueError(f'Completed stage output changed/missing: {p}')
            backup = self.path.with_name('pipeline_state.before_runtime_fix_'+uuid.uuid4().hex[:8]+'.json')
            shutil.copy2(self.path, backup)
            self.state.setdefault('runtime_code_updates', []).append(dict(
                previous_code_sha256=old['code_sha256'], code_sha256=signature['code_sha256'],
                reason='project temporary directory shortened for AF_UNIX sockets', updated_utc=now(),
                previous_state=str(backup)))
            self.state['signature'] = signature
            write_json(self.path, self.state)
            print('RUNTIME_CODE_UPDATE_VERIFIED', backup, flush=True)

    def execute(self, name, action):
        previous = self.state['stages'].get(name, {})
        if previous.get('status') == 'SUCCESS':
            for entry in previous['artifacts'].values():
                p = Path(entry['path'])
                if not p.is_file() or digest(p) != entry['sha256']:
                    raise ValueError(f'Completed stage output changed/missing: {p}')
            print(f'STAGE_REUSED {name}', flush=True)
            return {k: Path(v['path']) for k, v in previous['artifacts'].items()}
        attempt = int(previous.get('attempt', 0)) + 1
        directory = self.root / name / f'attempt_{attempt:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        entry = dict(status='RUNNING', attempt=attempt, output=str(directory), started_utc=now())
        self.state.update(status='RUNNING', current_stage=name)
        self.state['stages'][name] = entry; write_json(self.path, self.state)
        print(f'STAGE_START {name} attempt={attempt}', flush=True)
        try:
            result = action(directory)
            artifacts = {}
            for key, value in result.items():
                p = Path(value).resolve()
                if not p.is_file() or p.stat().st_size == 0:
                    raise ValueError(f'Stage {name} did not produce nonempty {key}: {p}')
                artifacts[key] = dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
            if not artifacts:
                raise ValueError('Stage returned no artifacts')
            entry.update(status='SUCCESS', artifacts=artifacts, finished_utc=now())
            write_json(directory / 'stage_success.json', entry)
            write_json(self.path, self.state)
            print(f'STAGE_SUCCESS {name}', flush=True)
            return {k: Path(v['path']) for k, v in artifacts.items()}
        except BaseException as exc:
            entry.update(status='FAILED', error=str(exc), finished_utc=now())
            self.state.update(status='FAILED'); write_json(self.path, self.state)
            raise


def command(args, directory, cwd, env=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    args = list(map(str, args))
    write_json(directory / 'command.json', dict(argv=args, cwd=str(cwd)))
    print('COMMAND ' + subprocess.list2cmdline(args), flush=True)
    with (directory / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   encoding='utf-8', errors='replace', text=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
    if code:
        raise RuntimeError(f'Command exited {code}; inspect {directory / "console.log"}')
