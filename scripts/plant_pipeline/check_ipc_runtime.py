"""Test multiprocessing socket and FD sharing with the actual child Python."""
import faulthandler
import os
import sys
import tempfile


def main():
    if os.name != 'posix':
        raise RuntimeError('The server IPC check requires Linux')
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        from multiprocessing.connection import Listener
        from multiprocessing import resource_sharer
        print('IPC_CHECK_START', 'temp='+tempfile.gettempdir(), flush=True)
        with Listener(family='AF_UNIX') as listener:
            print('IPC_SOCKET_OK', listener.address,
                  'bytes='+str(len(os.fsencode(listener.address))), flush=True)
        read_fd, write_fd = os.pipe()
        rebuilt = None
        try:
            # PyTorch uses this exact sharing mechanism for DataLoader tensors.
            duplicate = resource_sharer.DupFd(read_fd)
            rebuilt = duplicate.detach()
            os.write(write_fd, b'ipc')
            if os.read(rebuilt, 3) != b'ipc':
                raise RuntimeError('Multiprocessing FD sharing returned invalid data')
        finally:
            if rebuilt is not None:
                os.close(rebuilt)
            os.close(read_fd)
            os.close(write_fd)
            resource_sharer.stop()
        print('PIPELINE_IPC_OK', sys.version.split()[0], flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
