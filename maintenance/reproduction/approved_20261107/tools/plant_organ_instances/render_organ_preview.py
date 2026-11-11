"""Render actual instance colors and per-region silhouette depth, no AI imagery."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from split_plant_organs import instance_colors
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contact_weights'))
from preview_fonts import preview_font


def point_view(xyz, rgb, width, height):
    p = xyz - (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    scale = .86 * min(width / max(np.ptp(p[:, 0]), 1e-9), height / max(np.ptp(p[:, 1]), 1e-9))
    x = p[:, 0] * scale + width / 2
    y = -p[:, 1] * scale + height / 2
    im = Image.new('RGB', (width, height), (242, 244, 247))
    d = ImageDraw.Draw(im)
    for i in np.argsort(p[:, 2]):
        d.ellipse((x[i]-1, y[i]-1, x[i]+1, y[i]+1), fill=tuple(map(int, rgb[i])))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instances-dir', type=Path, required=True)
    ap.add_argument('--coordinate-system', type=Path, help='Optional board frame only for display; not used to split organs.')
    ap.add_argument('--weights-dir', type=Path, help='Optional matching contact weights, for an additional actual-weight close-up.')
    ap.add_argument('--weight-detail-output', type=Path, help='Optional separate path for the weighted close-up.')
    args = ap.parse_args()
    out = args.instances_dir
    manifest = json.loads((out / 'organ_report.json').read_text(encoding='utf-8'))
    ply = PlyData.read(str(out / 'organ_instances.ply'))
    v = ply['vertex'].data
    xyz = np.column_stack([v[k] for k in ('x', 'y', 'z')]).astype(float) * manifest['mm_per_unit']
    ids = v['organ_id']
    with np.load(out / 'organ_instances.npz', allow_pickle=False) as arrays:
        labels = arrays['vertex_labels']
    leaf_id, flower_id = manifest['leaf_id'], manifest['flower_id']
    mask = np.isin(labels, (leaf_id, flower_id))
    if args.coordinate_system:
        cs = json.loads(args.coordinate_system.read_text(encoding='utf-8'))
        frame = np.array(cs['board_frame_axes'])
        if np.linalg.det(frame) < 0:
            frame[:, 2] *= -1
        upright = (xyz - np.array(cs['board_center_mm'])) @ frame
    else:
        display_mask = mask if np.any(mask) else np.ones(len(xyz), bool)
        p = xyz[display_mask] - xyz[display_mask].mean(axis=0)
        _, basis = np.linalg.eigh(p.T @ p)
        # Long axis up, next axis horizontal. No assumption about growth direction.
        upright = xyz @ basis[:, [1, 2, 0]]
    rng = np.random.default_rng(772)
    take = np.flatnonzero(mask)
    if not len(take):
        take = np.arange(len(xyz))
    if len(take) > 260000:
        take = rng.choice(take, 260000, replace=False)
    palette = instance_colors(ids[take])
    font, chinese = preview_font(25)
    small, _ = preview_font(19)
    canvas = Image.new('RGB', (1600, 1070), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), '部件拆分候选：不同颜色表示不同编号（仍需复核）' if chinese else
              'Organ proposals: distinct colors are instance IDs (review required)', fill=(30, 30, 35), font=font)
    for j, (title, axes) in enumerate([('正视' if chinese else 'Front', (0, 1, 2)), ('侧视' if chinese else 'Side', (2, 1, 0))]):
        canvas.paste(point_view(upright[take][:, axes], palette, 800, 880), (j*800, 90))
        draw.text((24+j*800, 60), title, fill=(30, 30, 35), font=small)
    draw.text((24, 990), f"叶片候选 {manifest['per_class'][str(leaf_id)]}；花/花瓣候选 {manifest['per_class'][str(flower_id)]}。灰黑色为未解决碎片。" if chinese else
              f"Leaf proposals {manifest['per_class'][str(leaf_id)]}; flower/petal proposals {manifest['per_class'][str(flower_id)]}. Dark gray = unresolved.", fill=(30, 30, 35), font=small)
    draw.text((24, 1025), '颜色只表示部件编号，不是风险权重；不会改变叶、花、枝干等语义类别。' if chinese else
              'Colors indicate instance IDs, not risk. Original semantic classes are preserved.', fill=(30, 30, 35), font=small)
    canvas.save(out / 'organ_instances_preview.png')
    # Largest leaf/flower regions: their own PCA projection exposes center/rim.
    regions = []
    for cid in (leaf_id, flower_id):
        regions += sorted((r for r in manifest['instances'] if r['class_id'] == cid),
                          key=lambda r: -r['reconstructed_surface_area_mm2'])[:2]
    detail = Image.new('RGB', (1600, 1050), (242, 244, 247))
    d = ImageDraw.Draw(detail)
    for j, r in enumerate(regions):
        q = np.flatnonzero(ids == r['instance_id'])
        if len(q) > 120000:
            q = rng.choice(q, 120000, replace=False)
        frame = np.asarray(r['plane_axes'])
        p = (xyz[q] - r['center_xyz_mm']) @ frame
        c = v['center_score'][q]
        start = np.array([210, 244, 190]) if r['class_id'] == leaf_id else np.array([255, 244, 160])
        end = np.array([35, 143, 65]) if r['class_id'] == leaf_id else np.array([227, 136, 35])
        rgb = start + (end - start) * c[:, None]
        detail.paste(point_view(p[:, [0, 1, 2]], rgb, 800, 420), ((j%2)*800, (j//2)*520+80))
        x, y = (j%2)*800+22, (j//2)*520+20
        title = f"{'叶片' if r['class_id'] == leaf_id else '花/花瓣'}候选 #{r['instance_id']}；表面积 {r['reconstructed_surface_area_mm2']:.0f} mm²" if chinese else f"{'Leaf' if r['class_id'] == leaf_id else 'Flower/petal'} #{r['instance_id']}; area {r['reconstructed_surface_area_mm2']:.0f} mm2"
        d.text((x, y), title, fill=(30, 30, 35), font=small)
        d.text((x, y+30), '浅色边缘 → 深色内部（几何中心分数，尚非最终权重）' if chinese else
               'Pale rim -> dark interior (geometry score, not final weight)', fill=(30, 30, 35), font=small)
    detail.save(out / 'organ_center_preview.png')
    if args.weights_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contact_weights'))
        from weight_palette import risk_colors
        with np.load(args.weights_dir / 'contact_weights.npz', allow_pickle=False) as z:
            if str(z['mesh_fingerprint']) != manifest['mesh_fingerprint']:
                raise ValueError('Weights and instances have different source mesh fingerprints.')
            weights = z['vertex_effective_weight']
            heat = risk_colors(weights, z['vertex_hard_collision'])
        weighted = Image.new('RGB', (1600, 1050), (242, 244, 247))
        wd = ImageDraw.Draw(weighted)
        for j, r in enumerate(regions):
            q = np.flatnonzero(ids == r['instance_id'])
            values = weights[q]
            if len(q) > 120000:
                q = rng.choice(q, 120000, replace=False)
            p = (xyz[q] - r['center_xyz_mm']) @ np.asarray(r['plane_axes'])
            weighted.paste(point_view(p, heat[q], 800, 420), ((j%2)*800, (j//2)*520+80))
            x, y = (j%2)*800+22, (j//2)*520+20
            title = f"{'叶片' if r['class_id']==leaf_id else '花/花瓣'}候选 #{r['instance_id']}；权重 {values.min():.2f}–{values.max():.2f}" if chinese else f"{'Leaf' if r['class_id']==leaf_id else 'Flower/petal'} #{r['instance_id']}; weight {values.min():.2f}-{values.max():.2f}"
            wd.text((x,y), title, fill=(30,30,35),font=small)
            wd.text((x,y+30), '与完整权重模型共用颜色；内侧更深，仍需复核实例边界' if chinese else
                    'Shared risk palette; darker interior; inspect instance boundaries', fill=(30,30,35), font=small)
        detail_path = args.weight_detail_output or args.weights_dir / 'organ_weights_detail.png'
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        weighted.save(detail_path)
    print(out / 'organ_instances_preview.png')
    print(out / 'organ_center_preview.png')


if __name__ == '__main__':
    main()
