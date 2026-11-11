"""Integrity checks for transporting exact approved inputs and arrays."""
from pathlib import Path
import json
import tempfile
import unittest
import zipfile

import numpy as np
from approved_reproduction import array_digest, digest, npz_digest, unpack


class ReproductionTests(unittest.TestCase):
    def test_npz_fingerprint_includes_every_field_dtype_shape_and_value(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Path(folder)/'a.npz'; b = Path(folder)/'b.npz'
            np.savez(a, x=np.arange(6,dtype='f4'), label=np.array('abc'))
            np.savez_compressed(b, label=np.array('abc'), x=np.arange(6,dtype='f4'))
            self.assertEqual(npz_digest(a),npz_digest(b))
            self.assertNotEqual(array_digest(np.arange(6,dtype='f4')),array_digest(np.arange(6,dtype='f8')))
            self.assertNotEqual(array_digest(np.arange(6)),array_digest(np.arange(6).reshape(2,3)))
            np.savez(b, x=np.arange(6,dtype='f4')+1, label=np.array('abc'))
            self.assertNotEqual(npz_digest(a),npz_digest(b))

    def test_corrupt_archive_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'a.zip';archive.write_bytes(b'not an archive')
            destination=Path(folder)/'output'
            with self.assertRaisesRegex(ValueError,'checksum'):
                unpack(archive,'0'*64,destination)
            self.assertFalse(destination.exists())

    def test_path_escape_rejected_and_valid_payload_verified(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            for index,name in enumerate(('source/input.txt','../escaped.txt')):
                archive=Path(folder)/f'{index}.zip';data=b'input'
                manifest={'files':{name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())}}
                with zipfile.ZipFile(archive,'w')as z:
                    z.writestr(name,data);z.writestr('reproduction_manifest.json',json.dumps(manifest))
                destination=Path(folder)/f'out{index}'
                if index:
                    with self.assertRaisesRegex(ValueError,'Invalid reproduction member'):
                        unpack(archive,digest(archive),destination)
                    self.assertFalse((Path(folder)/'escaped.txt').exists())
                else:
                    unpack(archive,digest(archive),destination)
                    self.assertEqual((destination/name).read_bytes(),data)


if __name__=='__main__':
    unittest.main()
