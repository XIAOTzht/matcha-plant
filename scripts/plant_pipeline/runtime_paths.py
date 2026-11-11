"""Keep temporary files in the project without exceeding Linux socket limits."""
from __future__ import annotations
import os
from pathlib import Path

from common import code_signature, digest


def compatible_temporary_path_signatures(root):
    """Reconstruct only the exact pre-fix signature; never bypass other changes."""
    root = Path(root)
    replacements = {
        'scripts/plant_pipeline/common.py': (
            'a9fff86f0607a0f7a6771d1e824de1728b1da6c9e55d9be5d2fbb6ca03bee1aa',
            '8bcc0699551b3edbc7ab6955bc43451552941081c46cd5fe21613b14322c3121'),
        'scripts/plant_pipeline/run_pipeline.py': (
            '29cf8f7c5d36c018843e694362a467b11d62a3ef0128e46d66991244b0bf0afd',
            'eb695bb1a9f0af43f4d55c705b0151be8faf175e6dcbfbd1ee5c2af6d0b782e6'),
    }
    if any(not (root/name).is_file() or digest(root/name) != values[0]
           for name, values in replacements.items()):
        return ()
    return (code_signature(root, overrides={name:values[1] for name,values in replacements.items()},
                           omitted=('scripts/plant_pipeline/runtime_paths.py',
                                    'scripts/plant_pipeline/check_ipc_runtime.py')),)


def validate_unix_socket_directory(directory):
    address = os.fspath(directory)+'/pymp-xxxxxxxx/listener-xxxxxxxx'
    if len(os.fsencode(address)) > 107:
        raise ValueError(f'Project path is too long for multiprocessing sockets: {address}')


def temporary_environment(root, base=None):
    env = dict(os.environ if base is None else base)
    directory = Path(root).resolve()/'t'
    # CPython adds both a pymp directory and a listener filename. Linux leaves
    # at most 107 bytes for a pathname AF_UNIX address, including the full root.
    if os.name == 'posix':
        validate_unix_socket_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('TMPDIR', 'TEMP', 'TMP'):
        env[name] = str(directory)
    return env
