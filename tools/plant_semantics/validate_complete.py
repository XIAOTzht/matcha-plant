"""Verify exact source geometry, five-class coverage, and serialized labels."""
import argparse
from pathlib import Path
import numpy as np
import plantseg as p
from plyfile import PlyData


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    original=PlyData.read(args.input,mmap=False)
    v=original['vertex'].data
    xyz=np.column_stack([v['x'],v['y'],v['z']])
    tri=np.vstack(original['face']['vertex_indices'])
    data=np.load(args.output/'labels.npz')
    label=data['vertex_labels'];face_labels=data['face_labels']
    assert len(label)==len(xyz) and len(face_labels)==len(tri)
    assert np.isin(label,[1,2,3,4,5]).all()
    assert np.isin(face_labels,[1,2,3,4,5]).all()
    checks={}
    for name in ['semantic_mesh.ply','labeled_original_rgb.ply']:
        exported=PlyData.read(args.output/name,mmap=False)
        ev=exported['vertex'].data
        result=dict(xyz_exact=bool(np.array_equal(xyz,np.column_stack([ev['x'],ev['y'],ev['z']]))),
                    triangle_indices_exact=bool(np.array_equal(tri,np.vstack(exported['face']['vertex_indices']))),
                    vertex_labels_exact=bool(np.array_equal(label,ev['label'])),
                    face_labels_exact=bool(np.array_equal(face_labels,exported['face']['label'])),
                    uncertainty_exact=bool(np.array_equal(data['uncertain'],ev['uncertain'])),
                    inferred_exact=bool(np.array_equal(data['inferred'],ev['inferred'])))
        color=np.column_stack([ev['red'],ev['green'],ev['blue']])
        expected=p.COLORS[label] if name=='semantic_mesh.ply' else np.column_stack([v['red'],v['green'],v['blue']])
        result['colors_exact']=bool(np.array_equal(color,expected))
        assert all(result.values()),result
        checks[name]=result
    checks.update(input_sha256=p.sha256(args.input),vertex_count=len(xyz),face_count=len(tri),
                  unknown_vertices=int((label==0).sum()),unknown_faces=int((face_labels==0).sum()),
                  complete_assignment=True,semantic_accuracy_measured=False)
    p.save_json(args.output/'validation.json',checks)
    print(checks)


if __name__=='__main__':main()
