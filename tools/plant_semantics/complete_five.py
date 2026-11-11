"""Closed-set five-class plant mesh segmentation with complete vertex/face labels.

Reuses frozen zero-shot detections, explicitly preserves bud instances, and uses
edge-constrained color/geometry refinement. Uncertainty is a separate field.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import plantseg as p
from PIL import Image, ImageDraw, ImageFont


def key(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sources(args):
    base = json.loads((args.base/'manifest.json').read_text(encoding='utf-8'))
    detail = json.loads((args.detail/'manifest.json').read_text(encoding='utf-8'))
    if base['input_sha256'] != detail['input_sha256'] or p.sha256(base['input']) != base['input_sha256']:
        raise ValueError('Input identity mismatch')
    return base, detail


def instances(args):
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    base, detail = sources(args)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'instances').mkdir(exist_ok=True)
    config = dict(revision=1, sam_sha256=p.sha256(args.sam),
                  input_sha256=base['input_sha256'], parent_camera_votes=True,
                  sources=[str(args.base.resolve()),str(args.detail.resolve())])
    for folder, manifest in [(args.base,base),(args.detail,detail)]:
        config[str(folder)] = {v['name']:p.sha256(folder/'views'/f"{v['name']}_detections.json")
                               for v in manifest['views']}
    config['cache_key'] = key(config)
    if (out/'instance_config.json').exists():
        old = json.loads((out/'instance_config.json').read_text(encoding='utf-8'))
        if old['cache_key'] != config['cache_key']:
            raise ValueError('Instance recipe changed; choose a new output directory')
    p.save_json(out/'instance_config.json', config)
    torch.set_num_threads(args.threads)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = SamPredictor(sam_model_registry['vit_t'](checkpoint=str(args.sam)).to(device).eval())
    for folder, manifest in [(args.base,base),(args.detail,detail)]:
        for camera in manifest['views']:
            name = camera['name']
            target = out/'instances'/f'{name}.npz'
            if target.exists():
                if str(np.load(target)['cache_key']) != config['cache_key']:
                    raise ValueError(f'Stale instance scores: {name}')
                print('Cached',name,flush=True)
                continue
            start = time.time()
            image = np.asarray(Image.open(folder/'views'/f'{name}.png').convert('RGB'))
            fid = np.load(folder/'views'/f'{name}_rays.npz')['face_id']
            foreground = fid >= 0
            detections = json.loads((folder/'views'/f'{name}_detections.json').read_text(encoding='utf-8'))
            predictor.set_image(image)
            scores = np.zeros((*fid.shape,6),np.float32)
            bud = np.zeros(fid.shape,np.float32)
            yy,xx = np.ogrid[:fid.shape[0],:fid.shape[1]]
            records = []
            for det in detections:
                # The dense model often calls dark leaves and soil "pot". Independent
                # instance proposals must survive that error rather than inherit it.
                masks, quality, _ = predictor.predict(box=np.array(det['box']),multimask_output=True)
                cls = det['class_id']
                index = int(np.argmax(quality))
                mask = masks[index]&foreground
                x1,y1,x2,y2 = det['box']
                mask &= (xx >= x1-2)&(xx <= x2+2)&(yy >= y1-2)&(yy <= y2+2)
                fraction = float(mask.sum()/max(foreground.sum(),1))
                maximum = {1:.22,2:.50,3:.35,4:1.,5:.60}[cls]
                accepted = bool(quality[index] >= .75 and mask.sum() >= 8 and fraction <= maximum)
                if det['phrase'] == 'bud':
                    accepted &= fraction < .15
                score = float(det['detector_score']*np.clip(quality[index],0,1))
                records.append(dict(phrase=det['phrase'],class_id=cls,pixels=int(mask.sum()),
                                    fraction=fraction,accepted=accepted,score=score))
                if accepted:
                    scores[...,cls][mask] = np.maximum(scores[...,cls][mask],score)
                    if det['phrase'] == 'bud':
                        bud[mask] = np.maximum(bud[mask],score)
            np.savez_compressed(target,scores=scores.astype(np.float16),bud=bud.astype(np.float16),
                                cache_key=config['cache_key'])
            label = scores.argmax(2)
            overlay = image.copy()
            overlay[foreground] = (image[foreground]*.4+p.COLORS[label[foreground]]*.6).astype(np.uint8)
            Image.fromarray(overlay).save(out/'instances'/f'{name}_overlay.jpg')
            p.save_json(out/'instances'/f'{name}_records.json',records)
            print(name,'accepted',sum(r['accepted'] for r in records),'of',len(records),
                  'seconds',round(time.time()-start,1),flush=True)


def project_instances(args, xyz, tri, base, detail):
    out = args.output
    config = json.loads((out/'instance_config.json').read_text(encoding='utf-8'))
    target = out/'instance_vertices.npz'
    if target.exists():
        cached = np.load(target)
        if str(cached['cache_key']) != config['cache_key']:
            raise ValueError('Stale projected instance scores')
        return {k:cached[k] for k in cached.files if k != 'cache_key'}
    scene = p.scene_from_arrays(xyz,tri)
    total = np.zeros((len(xyz),6),np.float32)
    support_count = np.zeros((len(xyz),6),np.uint16)
    count = np.zeros(len(xyz),np.uint16)
    bud_total = np.zeros(len(xyz),np.float32)
    bud_count = np.zeros(len(xyz),np.uint16)
    maximum = np.zeros((len(xyz),6),np.float32)
    for parent in base['views']:
        relevant = [(args.base,parent)] + [(args.detail,v) for v in detail['views'] if v['parent'] == parent['name']]
        values = np.zeros((len(xyz),6),np.float32)
        bud = np.zeros(len(xyz),np.float32)
        seen = np.zeros(len(xyz),bool)
        for folder,camera in relevant:
            cached = np.load(out/'instances'/f"{camera['name']}.npz")
            if str(cached['cache_key']) != config['cache_key']:
                raise ValueError('Stale 2D instances')
            scores = cached['scores'].astype(np.float32)
            # Project separate bud evidence with the same visibility rule.
            sampled,observed = p.sample_visible_vertices(scene,xyz,camera,scores)
            values = np.maximum(values,sampled)
            extra = np.zeros_like(scores)
            extra[...,3] = cached['bud']
            sampled_bud,_ = p.sample_visible_vertices(scene,xyz,camera,extra)
            bud = np.maximum(bud,sampled_bud[:,3])
            seen |= observed
        total += values
        support_count += values > .10
        maximum = np.maximum(maximum,values)
        count += seen
        bud_total += bud
        bud_count += bud > .10
        print('Projected',parent['name'],'bud vertices',int((bud > .1).sum()),flush=True)
    data = dict(total=total,support_count=support_count,view_count=count,
                bud_total=bud_total,bud_count=bud_count,maximum=maximum)
    np.savez_compressed(target,**data,cache_key=config['cache_key'])
    return data


def mesh_graph(xyz,tri,lab):
    """Only actual mesh edges; nearby disconnected surfaces are never linked."""
    from scipy import sparse
    a = tri[:,[0,1,2]].ravel()
    b = tri[:,[1,2,0]].ravel()
    edges = np.sort(np.column_stack([a,b]),axis=1)
    edges = np.unique(edges,axis=0)
    a,b = edges.T
    length = np.linalg.norm(xyz[a]-xyz[b],axis=1)
    color = np.linalg.norm(lab[a]-lab[b],axis=1)
    weights = (.025+np.exp(-.5*(color/13)**2))*np.clip(np.median(length)/np.maximum(length,1e-10),.2,5)
    graph = sparse.coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(len(xyz),len(xyz))).tocsr()
    rows = np.asarray(graph.sum(1)).ravel()
    normalized = sparse.diags(1/np.maximum(rows,1e-8))@graph
    return normalized.astype(np.float32)


def diffuse(graph, unary, strength, anchors, iterations=80):
    posterior = unary.copy()
    for i in range(iterations):
        posterior = (strength[:,None]*unary+graph@posterior)/(strength[:,None]+1)
        posterior[anchors] = unary[anchors]
        if (i+1)%20 == 0:
            print('Graph refinement',i+1,flush=True)
    posterior /= np.maximum(posterior.sum(1,keepdims=True),1e-8)
    return posterior


def appearance_likelihood(lab, seeds, rng):
    from sklearn.mixture import GaussianMixture
    from scipy.special import softmax
    density = np.full((len(lab),5),-25,np.float32)
    diagnostics = []
    for cls in range(1,6):
        indices = np.flatnonzero(seeds == cls)
        if len(indices) < 30:
            continue
        indices = rng.choice(indices,min(18000,len(indices)),replace=False)
        model = GaussianMixture(n_components=min(5,max(1,len(indices)//1000)),reg_covar=8,
                                random_state=0,max_iter=100,n_init=2)
        model.fit(lab[indices])
        density[:,cls-1] = model.score_samples(lab)
        diagnostics.append(dict(class_id=cls,seeds=len(indices),components=model.n_components,converged=bool(model.converged_)))
    # Soften scene color likelihoods: green buds and stems can have identical color.
    return softmax(density/2,axis=1).astype(np.float32),diagnostics


def closed_face_labels(tri,vertex_labels,scores):
    # Majority for ordinary triangles; resolve three-way ties by summed evidence.
    corner = vertex_labels[tri]
    labels = corner[:,0].copy()
    same12 = corner[:,1] == corner[:,2]
    labels[same12] = corner[same12,1]
    three = (corner[:,0] != corner[:,1])&(corner[:,0] != corner[:,2])&~same12
    if three.any():
        labels[three] = scores[tri[three]].sum(1).argmax(1)+1
    return labels.astype(np.uint8),np.any(corner != corner[:,:1],axis=1)


def fill_unobserved_components(graph,unary,observed,iterations=100):
    """Initialize invisible vertices from edge-connected semantic evidence.

    Components with no observations retain their appearance/shape fallback.
    """
    from scipy.sparse.csgraph import connected_components
    components,component = connected_components(graph,directed=False)
    seeded = np.bincount(component,weights=observed,minlength=components) > 0
    fill = ~observed & seeded[component]
    values = unary.copy()
    values[fill] = 0
    for _ in range(iterations):
        average = graph@values
        values[fill] = average[fill]
        values[observed] = unary[observed]
    mass = values.sum(1)
    reached = fill&(mass > 1e-8)
    unary = unary.copy()
    unary[reached] = values[reached]/mass[reached,None]
    return unary,reached


def refine(args):
    import cv2
    from scipy.special import softmax
    from refine_detail import pot_envelope
    base,detail = sources(args)
    xyz,tri,rgb = p.mesh_data(base['input'])
    original = np.load(args.detail/'labels.npz')
    original_labels = original['vertex_labels']
    means = original['mean_evidence'].astype(np.float32)
    feature = np.load(args.detail/'geometry.npz')['features']
    instance = project_instances(args,xyz,tri,base,detail)
    obs = instance['view_count']
    votes = instance['support_count']
    support = instance['total']/np.maximum(votes,1)*np.sqrt(votes/np.maximum(obs[:,None],1))
    support = np.clip(support/.35,0,1)
    tube = np.clip((feature[:,3]-.30)/.40,0,1)
    planar = np.clip((feature[:,4]-.40)/.40,0,1)
    # Broad "branch" proposals can include whole leaves. A cylindrical stem is
    # line-like at the larger neighbourhood, unlike a broad leaf blade.
    support[:,1] *= .20+.80*tube
    bud_strength = instance['bud_total']/np.maximum(instance['bud_count'],1)*np.sqrt(instance['bud_count']/np.maximum(obs,1))
    bud = (instance['bud_count'] >= 2)&(bud_strength > .16)
    envelope = pot_envelope(xyz,original_labels,base['views'][0]['up_axis'])
    seeds = original_labels.copy()
    # Unambiguous local instance masks recover dark leaves and thin stems.
    for cls in [1,2,5]:
        strong = (support[:,cls] > .48)&(votes[:,cls] >= 2)
        if cls == 1:
            strong &= (support[:,2] < .25)&(tube > .35)
        if cls == 5 and envelope:
            strong &= envelope['substrate']
        seeds[strong] = cls
    seeds[bud] = 0  # never fit a petal color model using green buds
    lab = cv2.cvtColor(rgb[:,None,:].astype(np.float32),cv2.COLOR_RGB2LAB)[:,0,:]
    appearance,diagnostics = appearance_likelihood(lab,seeds,np.random.default_rng(20261020))
    logits = means[:,1:]/.13 + support[:,1:]*3.0 + appearance*1.7
    # Require semantic/instance support before the scene color model can favor stem.
    stem_eligible = (means[:,1] > .14)|(support[:,1] > .20)
    logits[:,0] += tube*1.2*stem_eligible
    logits[:,0] -= (1-stem_eligible)*1.5
    leaf_eligible = (means[:,2] > .14)&(support[:,2] > .20)
    logits[:,1] += planar*1.2*leaf_eligible
    logits[:,0] -= planar*.8*(1-tube)*leaf_eligible
    if envelope:
        logits[~envelope['substrate'],4] = -20
        logits[~envelope['pot'],3] = -20
        # Container interior geometry supplements weak image evidence for soil.
        up = 'xyz'.index(envelope['up_axis'])
        horizontal = [i for i in range(3) if i != up]
        radial = np.sqrt((((xyz[:,horizontal]-envelope['center'])/envelope['axes'])**2).sum(1))
        radius = float(np.mean(envelope['axes']))
        rim = envelope['rim_height']
        interior = envelope['substrate']&(radial < .82)&(xyz[:,up] < rim-.015*radius)
        logits[interior,4] += 1.8
        shell = envelope['pot']&((radial > .90)|(xyz[:,up] < rim-.35*radius))
        logits[shell,3] += 2.0
    logits[bud,2] += 6
    unary = softmax(logits,axis=1).astype(np.float32)
    anchors = (unary.max(1) > .92)&((obs >= 2)|bud)
    # Explicit bud identity protects round green buds from color-based stem filling.
    unary[bud] = np.array([.005,.005,.98,.005,.005],np.float32)
    anchors |= bud
    strength = (.12+unary.max(1)*.6).astype(np.float32)
    strength[obs == 0] = .025
    graph = mesh_graph(xyz,tri,lab)
    unary,propagated = fill_unobserved_components(graph,unary,obs > 0)
    posterior = diffuse(graph,unary,strength,anchors,args.iterations)
    label = (posterior.argmax(1)+1).astype(np.uint8)
    face_label,boundary = closed_face_labels(tri,label,posterior)
    assert (label >= 1).all() and (label <= 5).all() and (face_label >= 1).all() and (face_label <= 5).all()
    ranking = np.sort(posterior,axis=1)
    margin = ranking[:,-1]-ranking[:,-2]
    # A diagnostic flag, deliberately independent of assigning every point a class.
    uncertain = (original_labels == 0)|(obs < 2)|(margin < .18)
    evidence = posterior.max(1)
    out = args.output
    np.savez_compressed(out/'labels.npz',vertex_labels=label,face_labels=face_label,
                        class_scores=posterior,margin=margin,evidence=evidence,
                        uncertain=uncertain,inferred=original_labels == 0,
                        propagated_unobserved=propagated,
                        view_count=obs,bud_protected=bud,face_boundary=boundary,
                        original_labels=original_labels,instance_support=support)
    p.write_labeled_ply(out/'semantic_mesh.ply',xyz,tri,rgb,label,evidence,True,
                        uncertain=uncertain,inferred=original_labels == 0,face_labels=face_label)
    p.write_labeled_ply(out/'labeled_original_rgb.ply',xyz,tri,rgb,label,evidence,False,
                        uncertain=uncertain,inferred=original_labels == 0,face_labels=face_label)
    p.save_json(out/'classes.json',p.CLASSES[1:])
    p.save_json(out/'manifest.json',dict(base,status='closed_set_all_vertices_assigned',
                base_directory=str(args.base.resolve()),detail_directory=str(args.detail.resolve()),
                labels_source_sha256=p.sha256(args.detail/'labels.npz')))
    area = np.linalg.norm(np.cross(xyz[tri[:,1]]-xyz[tri[:,0]],xyz[tri[:,2]]-xyz[tri[:,0]]),axis=1)*.5
    va = np.zeros(len(xyz),np.float64)
    for j in range(3):
        np.add.at(va,tri[:,j],area/3)
    counts = np.bincount(label,minlength=6)
    report = dict(status='complete_five_class_assignment_not_accuracy_validated',
                  method='frozen zero-shot instances + appearance pseudo-seeds + mesh-edge diffusion',
                  source_sha256=base['input_sha256'],vertex_count=len(xyz),face_count=len(tri),
                  unknown_vertices=int((label == 0).sum()),unknown_faces=int((face_label == 0).sum()),
                  class_vertex_count={c['name']:int(counts[c['id']]) for c in p.CLASSES[1:]},
                  class_surface_area_fraction={c['name']:float(va[label == c['id']].sum()/va.sum()) for c in p.CLASSES[1:]},
                  inferred_vertex_count=int((original_labels == 0).sum()),
                  uncertain_vertex_fraction=float(uncertain.mean()),bud_protected_vertices=int(bud.sum()),
                  mesh_propagated_unobserved_vertices=int(propagated.sum()),
                  manual_annotations=False,foundation_model_finetuning=False,per_scene_appearance_fitting=True,
                  semantic_accuracy_measured=False,cross_plant_validation=False,
                  scores_are_calibrated_probabilities=False,appearance_seeds=diagnostics,
                  limitations=['Every input vertex belongs to the five classes by user specification.',
                               'Complete assignment does not imply every assigned class is correct.',
                               'Uncertainty stored separately and never converted to gray labels.',
                               'Color/shape refinement can still confuse dark leaves, stems, and pot substrate.',
                               'Only actual mesh edges connect the diffusion graph; disconnected surfaces remain separate.'])
    p.save_json(out/'report.json',report)
    config = dict(recipe_revision=2,iterations=args.iterations,
                  base_labels_sha256=p.sha256(args.base/'labels.npz'),
                  detail_labels_sha256=p.sha256(args.detail/'labels.npz'),
                  instance_config_sha256=p.sha256(out/'instance_config.json'),
                  code_sha256=p.sha256(Path(__file__)),complete_closed_set=True,
                  uncertainty_separate=True,manual_annotations=False)
    p.save_json(out/'inference_config.json',config)
    preview(args,base,tri,label)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def preview(args,base,tri,label):
    old = np.load(args.detail/'labels.npz')['vertex_labels']
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    cn = font_path.exists()
    font = ImageFont.truetype(str(font_path),22) if cn else ImageFont.load_default(size=22)
    small = ImageFont.truetype(str(font_path),17) if cn else ImageFont.load_default(size=17)
    selected = [base['views'][i] for i in [0,2,4,6,8,10,12,13]]
    width = 420
    sheet = Image.new('RGB',(width*3,60+width*len(selected)+65),'white')
    draw = ImageDraw.Draw(sheet)
    titles = ['原始颜色','上一版（灰色为未分类）','本版：全部归入五类'] if cn else ['Original RGB','Previous labels (gray=unknown)','Complete five-class labels']
    for col,title in enumerate(titles):
        draw.text((col*width+10,12),title,font=font,fill='black')
    for row,camera in enumerate(selected):
        name = camera['name']
        original = Image.open(args.base/'views'/f'{name}.png').convert('RGB')
        ray = np.load(args.base/'views'/f'{name}_rays.npz')
        fid,bary = ray['face_id'],ray['barycentric']
        valid = fid >= 0
        corners = tri[fid[valid]]
        vertices = corners[np.arange(len(corners)),bary[valid].argmax(1)]
        images = [original]
        for labels in [old,label]:
            output = np.asarray(original).copy()
            output[valid] = p.COLORS[labels[vertices]]
            images.append(Image.fromarray(output))
        for col,image in enumerate(images):
            sheet.paste(image.resize((width,width)),(col*width,60+row*width))
        draw.text((8,60+row*width+5),name,font=small,fill='black')
    y = 65+len(selected)*width
    names = ['茎/枝/花梗','叶片','花/花苞','花盆/托盘','土壤/基质'] if cn else [c['name'] for c in p.CLASSES[1:]]
    for i,(cls,name) in enumerate(zip(p.CLASSES[1:],names)):
        x = i*250
        draw.rectangle((x+6,y,x+28,y+22),fill=tuple(cls['color']))
        draw.text((x+36,y-2),name,font=small,fill='black')
    sheet.save(args.output/'comparison_preview.jpg',quality=94)
    compact = Image.new('RGB',(width*3,60+width*2+65),'white')
    compact.paste(sheet.crop((0,0,width*3,60+width*2)),(0,0))
    compact.paste(sheet.crop((0,y-5,width*3,y+55)),(0,60+width*2+5))
    compact.save(args.output/'comparison_compact.jpg',quality=94)
    detail = json.loads((args.detail/'manifest.json').read_text(encoding='utf-8'))
    # Some valid plants have no detected pot opening or no high-elevation
    # linear crop.  Preview generation must remain optional and must never
    # make an otherwise successful segmentation fail with StopIteration.
    selections = []
    pot = next((v for v in detail['views'] if v.get('kind') == 'pot_opening'), None)
    linear = next((v for v in detail['views']
                   if v.get('kind') == 'linear_parts' and v.get('elevation', 0) >= 40), None)
    fallback = next(iter(detail.get('views', [])), None)
    for item in (pot, linear, fallback):
        if item is not None and item not in selections:
            selections.append(item)
        if len(selections) == 2:
            break
    if not selections:
        return
    zoom = Image.new('RGB',(600*2,50+600*len(selections)),'white')
    draw = ImageDraw.Draw(zoom)
    for col,title in enumerate(['原始颜色','完整五类结果'] if cn else ['Original RGB','Complete five-class labels']):
        draw.text((col*600+10,10),title,font=font,fill='black')
    for row,camera in enumerate(selections):
        name = camera['name']
        original = Image.open(args.detail/'views'/f'{name}.png').convert('RGB')
        ray = np.load(args.detail/'views'/f'{name}_rays.npz')
        fid,bary = ray['face_id'],ray['barycentric']
        valid = fid >= 0
        corners = tri[fid[valid]]
        vertices = corners[np.arange(len(corners)),bary[valid].argmax(1)]
        output = np.asarray(original).copy()
        output[valid] = p.COLORS[label[vertices]]
        zoom.paste(original.resize((600,600)),(0,50+row*600))
        zoom.paste(Image.fromarray(output).resize((600,600)),(600,50+row*600))
    zoom.save(args.output/'detail_preview.jpg',quality=94)


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['instances','refine','run'])
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--detail',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--sam',type=Path,default=p.ROOT/'models'/'mobile_sam_parallel.pt')
    args = parser.parse_args()
    if args.command in ['instances','run']:
        instances(args)
    if args.command in ['refine','run']:
        refine(args)


if __name__ == '__main__':
    main()
