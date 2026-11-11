#!/usr/bin/env python3
"""Cart-frame underside trim and planar infill; synthetic geometry, not a scan.

Requires metric six-class semantics and true RGB. The automatic template is for
the blue chassis / four yellow-wheel cart. Save and reuse its JSON for the same
cart; other carts require a measured template. No plant-specific coordinates.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'plant_semantics_with_car'))
from with_car_semantics import mesh, read_json, save_json, fresh, sha, COLORS

VERSION = 'cart-underside-0.1.0'


def estimate_template(q, rgb, cart, thickness=6.):
    r,g,b = np.asarray(rgb, float).T
    blue = cart & (b>r*1.4) & (b>g*1.15) & (b>60) & (q[:,1]<-15)
    if blue.sum()<500:
        raise ValueError('No lower blue chassis; supply a measured --template')
    h,e = np.histogram(q[blue,1], np.arange(np.floor(q[blue,1].min()), -14, 1.))
    peak = (e[np.argmax(h)]+e[np.argmax(h)+1])/2
    plane = blue & (np.abs(q[:,1]-peak)<1.5)
    p = q[plane][:,[0,2]]
    if len(p)<500 or np.min(np.ptp(p,axis=0))<50:
        raise ValueError('Insufficient broad chassis plane')
    lo,hi = np.percentile(p,[.2,99.8],axis=0)
    p=p[((p>=lo)&(p<=hi)).all(1)]
    # Use a voxel envelope to suppress scan tessellation bias, then convex hull.
    p=np.unique(np.round(p/.75),axis=0)*.75
    hull=p[ConvexHull(p).vertices]
    yellow=cart&(q[:,1]<-20)&(r>70)&(g>60)&(b<r*.65)&(b<g*.7)
    p=q[yellow]
    if len(p)<1000:raise ValueError('Four yellow wheel hubs not visible; supply --template')
    vox,inv=np.unique(np.floor(p/3).astype(int),axis=0,return_inverse=True)
    edges=cKDTree(vox).query_pairs(1.75,output_type='ndarray')
    _,groups=connected_components(coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
        shape=(len(vox),len(vox))).tocsr(),directed=False)
    groups=groups[inv];counts=np.bincount(groups);wheels=[]
    for k in np.argsort(-counts)[:4]:
        if counts[k]<500:raise ValueError('Fewer than four reliable hub components')
        hub=p[groups==k];low,high=np.percentile(hub,[1,99],axis=0)
        radii=(high-low)[:2]/2
        if min(radii)<10 or max(radii)/min(radii)>1.35:
            raise ValueError('Wheel hub is not approximately circular')
        cx,cy=((low+high)/2)[:2];outer=float(np.median(hub[:,2]))
        if abs(outer)<40:raise ValueError('Wheel side ambiguous')
        # Black tire extends beyond the yellow hub; these are explicit assumptions.
        radius=float(max(radii)*1.28+1.)
        inner=max(0.,abs(outer)-28.)
        zlo,zhi=(inner,abs(outer)+5.) if outer>0 else (-abs(outer)-5.,-inner)
        wheels.append(dict(center_xy_mm=[float(cx),float(cy)], radius_mm=radius,
                           z_range_mm=[zlo,zhi], hub_samples=int(counts[k])))
    for sign in (-1,1):
        side=[w for w in wheels if np.sign(np.mean(w['z_range_mm']))==sign]
        if len(side)!=2 or abs(side[0]['center_xy_mm'][0]-side[1]['center_xy_mm'][0])<50:
            raise ValueError('Expected two separated wheels on each side')
    return dict(version=VERSION,units='mm',frame='calibrated upper plate, y up',
        chassis_top_y_mm=float(peak),bottom_y_mm=float(peak-thickness),
        assumed_chassis_thickness_mm=float(thickness),footprint_xz_mm=hull.tolist(),wheels=wheels,
        basis='Visible blue plane and yellow wheel hubs; estimated, not measured CAD',
        assumptions=['Convex lower-chassis footprint','Planar underside',
                     'Tire radius = hub radius * 1.28 + 1 mm; protected width 33 mm'])


def validate_template(t):
    y=float(t['bottom_y_mm']);top=float(t['chassis_top_y_mm']);poly=np.array(t['footprint_xz_mm'],float)
    if t.get('units')!='mm' or not np.isfinite([y,top]).all() or not y<top<0:
        raise ValueError('Invalid chassis heights or units')
    if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():
        raise ValueError('Invalid footprint')
    hull=ConvexHull(poly)
    if len(hull.vertices)!=len(poly):raise ValueError('Footprint must be a convex polygon')
    poly=poly[hull.vertices]
    if len(t['wheels'])!=4:raise ValueError('Four wheel envelopes required')
    for w in t['wheels']:
        v=np.r_[w['center_xy_mm'],w['radius_mm'],w['z_range_mm']]
        if len(v)!=5 or not np.isfinite(v).all() or v[2]<=0 or v[3]>=v[4]:
            raise ValueError('Invalid wheel envelope')
    return poly


def wheel_field(q,w):
    c=np.array(w['center_xy_mm']);lo,hi=w['z_range_mm']
    return np.minimum(w['radius_mm']-np.linalg.norm(q[:,:2]-c,axis=1),
                      np.minimum(q[:,2]-lo,hi-q[:,2]))


def half_polygon(p,axis,value,greater):
    if len(p)<3:return np.empty((0,2))
    result=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        fa=(a[axis]-value)*(1 if greater else -1);fb=(b[axis]-value)*(1 if greater else -1)
        if fa>=0:result.append(a)
        if (fa>=0)!=(fb>=0):result.append(a+(b-a)*fa/(fa-fb))
    return np.asarray(result).reshape(-1,2)


def plane_pieces(poly,t):
    """Subtract exact rectangular wheel intersections from a convex planar cap."""
    pieces=[poly]
    for w in t['wheels']:
        cx,cy=w['center_xy_mm'];r=w['radius_mm'];dy=t['bottom_y_mm']-cy
        if abs(dy)>=r:continue
        dx=np.sqrt(r*r-dy*dy);zl,zh=w['z_range_mm'];updated=[]
        for p in pieces:
            inside=p
            for axis,value,greater in ((0,cx-dx,True),(0,cx+dx,False),(1,zl,True),(1,zh,False)):
                outside=half_polygon(inside,axis,value,not greater)
                if len(outside)>=3:updated.append(outside)
                inside=half_polygon(inside,axis,value,greater)
            # The remaining inside part is occupied by the protected wheel.
        pieces=updated
    return pieces


def tessellate_cap(p,spacing=2.):
    """Dense planar vertices also support downstream nearest-surface queries."""
    p=np.unique(np.round(p,9),axis=0)
    if len(p)<3:return np.empty((0,2)),np.empty((0,3),int)
    if np.linalg.matrix_rank(p-p.mean(0),tol=1e-8)<2:
        return np.empty((0,2)),np.empty((0,3),int)
    hull=ConvexHull(p)
    if hull.volume<1e-8:return np.empty((0,2)),np.empty((0,3),int)
    p=p[hull.vertices];lo=p.min(0);hi=p.max(0);boundary=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
        boundary.extend(a+(b-a)*np.arange(n)[:,None]/n)
    xx,zz=np.meshgrid(np.arange(np.ceil(lo[0]/spacing)*spacing,hi[0],spacing),
                      np.arange(np.ceil(lo[1]/spacing)*spacing,hi[1],spacing))
    grid=np.column_stack((xx.ravel(),zz.ravel()))
    eq=hull.equations
    inside=np.all(grid@eq[:,:2].T+eq[:,2]<=1e-8,axis=1)
    points=np.unique(np.round(np.concatenate((boundary,grid[inside])),8),axis=0)
    faces=Delaunay(points).simplices
    # scipy's 2D simplices are counterclockwise: x,z -> underside normal -y.
    return points,faces


def repair_geometry(q,tri,labels,face_labels,t):
    poly=validate_template(t);height=t['bottom_y_mm'];cart=labels==6
    field=q[:,1]-height
    wheel=np.full(len(q),-np.inf)
    for w in t['wheels']:wheel=np.maximum(wheel,wheel_field(q,w))
    field=np.maximum(field,wheel)
    eligible=(cart[tri].all(1))&(face_labels==6)
    keep=field>=0
    counts=keep[tri].sum(1)
    preserved=(~eligible)|(counts==3)
    face_ids=np.flatnonzero(preserved)
    faces=[tri[face_ids]];sources=[face_ids];new=[];edge_cache={};clipped=[];clipped_source=[]
    for fi in np.flatnonzero(eligible&(counts>0)&(counts<3)):
        result=[];f=tri[fi]
        for a,b in zip(f,np.roll(f,-1)):
            if keep[a]:result.append(int(a))
            if keep[a]!=keep[b]:
                key=tuple(sorted((int(a),int(b))))
                if key not in edge_cache:
                    alpha=field[a]/(field[a]-field[b])
                    edge_cache[key]=len(q)+len(new);new.append(q[a]+alpha*(q[b]-q[a]))
                result.append(edge_cache[key])
        for i in range(1,len(result)-1):
            clipped.append([result[0],result[i],result[i+1]]);clipped_source.append(fi)
    if clipped:faces.append(np.array(clipped));sources.append(np.array(clipped_source))
    cap=[];cap_start=len(q)+len(new)
    for p in plane_pieces(poly,t):
        p,ct=tessellate_cap(p)
        if not len(ct):continue
        offset=len(q)+len(new)
        new.extend(np.column_stack((p[:,0],np.full(len(p),height),p[:,1])))
        cap.extend((ct+offset).tolist())
    if not cap:raise ValueError('No planar infill generated')
    faces.append(np.array(cap));sources.append(np.full(len(cap),-1))
    full_q=np.concatenate((q,np.array(new).reshape(-1,3)));faces=np.concatenate(faces);sources=np.concatenate(sources)
    cross=np.cross(full_q[faces[:,1]]-full_q[faces[:,0]],full_q[faces[:,2]]-full_q[faces[:,0]])
    valid=np.linalg.norm(cross,axis=1)>1e-8;faces=faces[valid];sources=sources[valid]
    used=np.unique(faces);mapping=np.full(len(full_q),-1,np.int64);mapping[used]=np.arange(len(used))
    origin=np.where(used<len(q),used,-1)
    stats=dict(removed_original_faces=int((eligible&(counts<3)).sum()),
               clipped_faces=int(len(clipped)),planar_faces=int((sources<0).sum()),
               removed_original_vertices=int(len(q)-(origin>=0).sum()),
               synthetic_vertices=int((origin<0).sum()),
               wheel_original_vertices_retained=int(((origin>=0)&(wheel[np.maximum(origin,0)]>=0)).sum()))
    return full_q[used],mapping[faces],origin,sources,used>=cap_start,stats


def remap_data(data,origin,face_origin,nv,nf):
    result={};old=origin>=0;of=face_origin>=0
    inverse=np.full(nv,-1,np.int64);inverse[origin[old]]=np.flatnonzero(old)
    for k,a in data.items():
        if k=='plant_vertex_indices':
            v=inverse[a]
            if np.any(v<0):raise ValueError('A plant vertex would be removed')
            result[k]=v
        elif a.ndim and len(a) in (nv,nf):
            ids,mask=(origin,old) if len(a)==nv else (face_origin,of)
            b=np.zeros((len(ids),)+a.shape[1:],dtype=a.dtype)
            if a.dtype.kind=='f':b.fill(np.nan)
            b[mask]=a[ids[mask]];result[k]=b
        elif k=='class_score_ids':result[k]=a.copy()
        else:raise ValueError('Unrecognized semantic array: '+k)
    result['vertex_labels'][~old]=6;result['face_labels'][~of]=6
    for k in ('uncertain','inferred'):result[k][~old]=True
    result['label_source'][~old]=5
    result['class_score_valid'][~old]=False
    result['face_uncertain'][~of]=True
    result['cart_geometry_inferred']=~old
    result['cart_face_geometry_inferred']=~of
    return result


def run(args):
    src=args.source.resolve();out=args.output.resolve()
    if out==src or out in src.parents:raise ValueError('Use separate output')
    if not np.isfinite(args.chassis_thickness_mm) or not 1<=args.chassis_thickness_mm<=20:
        raise ValueError('Chassis thickness must be 1..20 mm')
    fresh(out);print('Loading metric geometry and semantic arrays',flush=True)
    ply,xyz,tri=mesh(src/'labeled_original_rgb_mm.ply');coords=read_json(src/'coordinate_system.json')
    if coords.get('units')!='mm':raise ValueError('Metric frame required')
    frame=np.array(coords['board_frame_axes']);center=np.array(coords['board_center_mm'])
    if frame.shape!=(3,3) or center.shape!=(3,) or not np.allclose(frame.T@frame,np.eye(3),atol=1e-6):
        raise ValueError('Invalid board frame')
    with np.load(src/'labels.npz')as z:data={k:z[k]for k in z.files}
    for element,key in (('vertex','vertex_labels'),('face','face_labels')):
        if not np.array_equal(ply[element].data['label'],data[key]):raise ValueError('PLY/NPZ labels disagree')
    rgb=np.column_stack([ply['vertex'].data[k]for k in ('red','green','blue')]);q=(xyz-center)@frame
    t=read_json(args.template) if args.template else estimate_template(q,rgb,data['vertex_labels']==6,args.chassis_thickness_mm)
    save_json(out/'cart_template.json',t)
    print('Repairing below',t['bottom_y_mm'],'mm; preserving four wheel envelopes',flush=True)
    repaired,faces,origin,face_origin,is_cap,stats=repair_geometry(q,tri,data['vertex_labels'],data['face_labels'],t)
    new_xyz=repaired@frame.T+center;old=origin>=0;of=face_origin>=0
    new_xyz[old]=xyz[origin[old]] # Bit-exact original vertices, no roundtrip drift.
    nd=remap_data(data,origin,face_origin,len(xyz),len(tri))
    inferred_faces=(~of)|(~old)[faces].any(1)
    nd['cart_face_geometry_inferred']=inferred_faces
    nd['face_uncertain'][inferred_faces]=True
    nd['face_boundary'][~of]=False
    original_vertex_ids=np.flatnonzero(old);not_cart=data['vertex_labels']!=6
    inverse=np.full(len(xyz),-1,np.int64);inverse[origin[old]]=original_vertex_ids
    protected_faces=(~(data['vertex_labels'][tri]==6).all(1))|(data['face_labels']!=6)
    retained_faces=set(face_origin[of].tolist())
    checks=dict(non_cart_vertices_all_retained=bool(np.all(inverse[not_cart]>=0)),
        non_cart_positions_exact=bool(np.array_equal(new_xyz[inverse[not_cart]],xyz[not_cart])),
        protected_faces_retained=all(int(i)in retained_faces for i in np.flatnonzero(protected_faces)),
        original_retained_positions_exact=bool(np.array_equal(new_xyz[old],xyz[origin[old]])),
        all_generated_vertices_cart=bool(np.all(nd['vertex_labels'][~old]==6)),
        all_generated_faces_cart=bool(np.all(nd['face_labels'][inferred_faces]==6)),
        planar_cap=bool(np.max(abs(repaired[is_cap,1]-t['bottom_y_mm']))<1e-8),
        indices_valid=bool(faces.min()>=0 and faces.max()<len(new_xyz)),
        finite=bool(np.isfinite(new_xyz).all()))
    if not all(checks.values()):raise ValueError(str(checks))
    fields=[(k,ply['vertex'].data.dtype.fields[k][0]) for k in ply['vertex'].data.dtype.names if k!='geometry_inferred']+[('geometry_inferred','u1')]
    vertices=np.zeros(len(new_xyz),dtype=fields)
    for k in ply['vertex'].data.dtype.names:vertices[k][old]=ply['vertex'].data[k][origin[old]]
    for i,k in enumerate(('x','y','z')):vertices[k]=new_xyz[:,i]
    for k,key in [('label','vertex_labels'),('uncertain','uncertain'),('inferred','inferred'),('label_source','label_source')]:vertices[k]=nd[key]
    vertices['geometry_inferred']=~old
    for k,value in zip(('red','green','blue'),(50,56,65)):vertices[k][~old]=value
    if 'alpha' in vertices.dtype.names:vertices['alpha'][~old]=255
    fd=np.empty(len(faces),dtype=[('vertex_indices','<i4',(3,)),('label','u1'),('uncertain','u1'),('geometry_inferred','u1')])
    fd['vertex_indices']=faces;fd['label']=nd['face_labels'];fd['uncertain']=nd['face_uncertain'];fd['geometry_inferred']=inferred_faces
    print('Writing repaired semantic and RGB meshes',flush=True)
    for name,semantic in [('labeled_original_rgb_mm.ply',False),('semantic_mesh_mm.ply',True)]:
        v=vertices.copy()
        if semantic:
            for i,k in enumerate(('red','green','blue')):v[k]=COLORS[nd['vertex_labels'],i]
        PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(fd,'face')],text=False,
                comments=['Metric cart underside repair; geometry_inferred=1 is synthetic, not photographed']).write(str(out/name))
        ep,ex,et=mesh(out/name)
        checks[name+'_roundtrip']=bool(np.array_equal(et,faces) and np.array_equal(ex,new_xyz.astype('f4').astype('f8'))
            and np.array_equal(ep['vertex'].data['label'],nd['vertex_labels']) and np.array_equal(ep['face'].data['label'],nd['face_labels']))
    if not all(checks.values()):raise ValueError('Roundtrip validation failed')
    np.savez_compressed(out/'labels.npz',**nd)
    np.savez_compressed(out/'source_mapping.npz',vertex_source_index=origin,face_source_index=face_origin)
    for name in ('coordinate_system.json','classes.json'):(out/name).write_bytes((src/name).read_bytes())
    old_report=read_json(src/'report.json')
    report=dict(version=VERSION,source=str(src),units='mm',
        vertex_count=len(vertices),face_count=len(faces),vertices=len(vertices),faces=len(faces),
        geometry_unchanged=False,cart_template_registration='estimated in calibrated board frame',
        source_mesh_sha256=sha(src/'labeled_original_rgb_mm.ply'),repair=stats,
        status='REVIEW_REQUIRED',cart_template='cart_template.json',
        semantic_source_report=str(src/'report.json'),semantic_accuracy_measured=False,
        original_retained_labels_unchanged=True,non_cart_geometry_unchanged=True,
        scores_note=old_report.get('scores_note'),label_source_ids=old_report.get('label_source_ids',{}),
        class_counts={str(k):int((nd['vertex_labels']==k).sum())for k in range(7)})
    report['class_vertex_counts']={c['name']:int((nd['vertex_labels']==c['id']).sum())for c in read_json(src/'classes.json')}
    report['uncertain_vertices']=int(nd['uncertain'].sum());report['uncertain_faces']=int(nd['face_uncertain'].sum())
    report['label_source_ids']['5']='Synthetic cart underside intersection or planar infill'
    report['limitations']=['Underside is an assumed planar infill, not measured CAD.',
        'Wheel envelopes are estimates; review visible wheels and future measured template.',
        'Original mesh is open. Planar infill does not guarantee watertightness or a manifold solid.',
        'Cart remains a hard obstacle. Rebuild weight/organ indices and replan before using this revised mesh.',
        'Not registered to a real robot. No claim of verified physical underside clearance.']
    save_json(out/'report.json',report);save_json(out/'validation.json',dict(status='PASS',checks=checks))
    save_json(out/'result_paths.json',{k:str(out/v)for k,v in dict(original_rgb='labeled_original_rgb_mm.ply',semantic_mesh='semantic_mesh_mm.ply',labels='labels.npz',template='cart_template.json',validation='validation.json').items()})
    (out/'CART_UNDERSIDE_REPAIR_SUCCESS').write_text(VERSION+'\n')
    print('CART_UNDERSIDE_REPAIR_SUCCESS',stats,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--template',type=Path,help='Reuse same-cart geometry in calibrated upper-plate mm frame')
    ap.add_argument('--chassis-thickness-mm',type=float,default=6.,help='Assumed lower chassis thickness, not measured')
    run(ap.parse_args())
