"""Organ-scale evidence rules; geometry alone never names a leaf or flower.

Pure NumPy/SciPy helpers, independent of camera pose, RGB and plant species.
All lengths are fractions of the plant bounding-box diagonal.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label as image_components


def mask_shape(mask, positions, local_features, plant_scale):
    """Describe a connected visible mask using uniformly spaced image pixels.

    positions/features have one row per True pixel of mask.  Descriptors are
    evidence, not a botanical classifier: petals and round leaves overlap.
    """
    points = np.asarray(positions, float)
    features = np.asarray(local_features, float)
    if len(points) < 20 or not np.isfinite(points).all() or plant_scale <= 0:
        return None
    if len(points) > 12000:
        take = np.linspace(0, len(points)-1, 12000).astype(int)
        points, features = points[take], features[take]
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points-center, axis=1)
    # Include tied radial distances despite roundoff after rigid transforms.
    points = points[distances <= np.quantile(distances, .99)+plant_scale*1e-10]
    if len(points) < 12:
        return None
    eigen, frame = np.linalg.eigh(np.cov(points.T))
    eigen = np.maximum(eigen[::-1], plant_scale**2*1e-12)
    projection = (points-center) @ frame[:, ::-1]
    spans = np.quantile(projection, .98, axis=0)-np.quantile(projection, .02, axis=0)
    yy, xx = np.nonzero(mask)
    xy = np.column_stack([xx, yy]).astype(float)
    eig2 = np.maximum(np.linalg.eigvalsh(np.cov(xy.T)), 1e-9)
    return dict(
        elongation=float(np.sqrt(eigen[0]/eigen[1])),
        thickness_ratio=float(np.sqrt(eigen[2]/eigen[1])),
        width_fraction=float(spans[1]/plant_scale),
        length_fraction=float(spans[0]/plant_scale),
        pixel_elongation=float(np.sqrt(eig2[1]/eig2[0])),
        planar_fraction=float(np.mean((features[:, 4] > .4) & (features[:, 3] < .55))),
        linear_fraction=float(np.mean(features[:, 3] > .50)),
        center=center.tolist())


def shape_factor(class_id, shape):
    """Reject broad branch masks; use modest priors to compare flower/leaf masks.

    A curved stem may fail the global aspect test, so a strong local tubular
    fraction can support it. Broad planar masks cannot receive stem identity.
    """
    if shape is None:
        return 0.
    aspect, width = shape['elongation'], shape['width_fraction']
    linear, planar = shape['linear_fraction'], shape['planar_fraction']
    if class_id == 1:
        if ((aspect < 2.3 and linear < .45) or
                (width > .04 and planar > .40 and linear < .35)):
            return 0.
        return float(np.clip(.35+.5*linear+.12*(aspect-2), .35, 1.))
    if class_id == 2:
        if shape['length_fraction'] > .35 and width > .18:
            return 0.  # Reject a whole arrangement proposed as a single leaf.
        # Narrow leaves are valid. Broad circular leaves are not called flowers
        # solely for their shape; text and multiview votes decide the class.
        return float(np.clip(.75+.22*planar+.08*np.clip(aspect-1, 0, 2), .70, 1.05))
    if class_id == 3:
        if shape['length_fraction'] > .35 and width > .18:
            return 0.  # A whole arrangement is not an individual flower.
        # Whole flower heads favour compact masks; individual petals may be long.
        return float(np.clip(.85+.18*(aspect < 2)+.08*(planar > .3), .80, 1.10))
    return 0.


def coherent_mask(mask, foreground):
    """Keep the largest 2D connected region, without adding background pixels."""
    mask = np.asarray(mask, bool) & foreground
    components, count = image_components(mask)
    if not count:
        return mask
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    return components == sizes.argmax()


def mask_evidence(class_id, shape, detector_score, sam_quality, dense_mean):
    factor = shape_factor(class_id, shape)
    if factor == 0 or sam_quality < .80 or detector_score < .20:
        return 0.
    # Detector/SAM identity dominates. Dense evidence is only a weak tie-breaker,
    # since its systematic errors caused some of the original stem/flower swaps.
    dense = np.asarray(dense_mean, float)
    identity = float(dense[class_id]) if len(dense) > class_id else 0.
    competitor = max((float(dense[k]) for k in (1, 2, 3) if k != class_id), default=0.)
    modifier = np.clip(1+.25*(identity-competitor), .85, 1.15)
    return float(detector_score * np.clip(sam_quality,0,1) * factor * modifier)


def decide_relabels(labels, old_scores, total, votes, observed):
    """Multi-camera consensus overrides old soft classes, never pot/soil/cart.

    A semantic score is heuristic, not a calibrated probability. Thresholds
    below deliberately retain uncertainty on every changed vertex.
    """
    labels = np.asarray(labels)
    if not np.isfinite(total).all() or np.any(total < 0):
        raise ValueError('Evidence must be finite and nonnegative')
    if old_scores.shape != (len(labels), 6):
        raise ValueError('Expected scores ordered by class_score_ids 1..6')
    # A missed detector proposal is missing evidence, not a vote against the
    # organ. Count confident identity votes, but still require >=3 directions.
    supported = votes[:,1:4].sum(1)
    means = total/np.maximum(supported[:, None], 1)
    ranked = np.sort(means[:, 1:4], axis=1)
    winner = means[:, 1:4].argmax(1)+1
    row = np.arange(len(labels))
    wins = votes[row, winner]
    agreement = wins/np.maximum(supported, 1)
    old_scores = np.asarray(old_scores)
    current = np.nan_to_num(old_scores[row, np.clip(labels-1, 0, old_scores.shape[1]-1)], nan=0.)
    strong_prior = current > .95
    accepted = (np.isin(labels, [1, 2, 3]) & (winner != labels) &
                (wins >= np.where(strong_prior, 4, 3)) &
                (agreement >= np.where(strong_prior, .80, .75)) &
                (ranked[:, -1] >= .16) & (ranked[:, -1]-ranked[:, -2] >= .035))
    output = labels.copy(); output[accepted] = winner[accepted]
    return output, accepted, means, agreement


def extend_surface_consensus(xyz, tri, rgb, labels, total, votes, observed, features, scale, reviewed):
    """Fill short gaps on an organ along actual colored mesh edges.

    Independent multiview votes supply seeds; Euclidean neighbours are never
    connected. Linear/tubular regions block leaf/flower propagation. A short
    scale-relative geodesic limit prevents a seed from relabelling a whole plant.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    soft=np.isin(labels,[1,2,3])
    support=votes[:,1:4].sum(1)
    winner=votes[:,1:4].argmax(1)+1
    confidence=votes[:,1:4].max(1)/np.maximum(support,1)
    mean=total[:,1:4].max(1)/np.maximum(support,1)
    trusted=soft&(votes[:,1:4].max(1)>=3)&(confidence>=.80)&(mean>=.18)
    # A thin stalk should not disappear when a nearby SAM leaf mask spills out.
    tubular=(features[:,3]>.55)&(features[:,4]<.35)
    allowed=soft&~tubular
    edge=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
    a,b=edge.T
    length=np.linalg.norm(xyz[a]-xyz[b],axis=1)
    keep=allowed[a]&allowed[b]&(length<=scale*.012)
    a,b,length=a[keep],b[keep],length[keep]
    # RGB distances are scene-relative boundary evidence, not class thresholds.
    color=np.linalg.norm(np.asarray(rgb[a],float)-np.asarray(rgb[b],float),axis=1)
    cost=length*(1+(color/.18)**2)
    graph=sparse.coo_matrix((np.r_[cost,cost],(np.r_[a,b],np.r_[b,a])),shape=(len(xyz),len(xyz))).tocsr()
    distances=[]
    for cls in (2,3):
        seeds=np.flatnonzero(trusted&allowed&(winner==cls))
        if not len(seeds):distances.append(np.full(len(xyz),np.inf));continue
        distances.append(dijkstra(graph,directed=False,indices=seeds,min_only=True,limit=scale*.035))
    distances=np.stack(distances,axis=1)
    order=np.argsort(distances,axis=1);rows=np.arange(len(xyz))
    near=distances[rows,order[:,0]];far=distances[rows,order[:,1]]
    chosen=order[:,0]+2
    # Keep competing established organ evidence. Require a clear surface-distance
    # advantage and at least one independent vote at the destination itself.
    local_votes=votes[rows,chosen]
    accept=(allowed&~trusted&(chosen!=reviewed)&(near<scale*.025)&
            (far>near*1.8+scale*.002)&(local_votes>=1)&
            ((winner==chosen)|(support==0)))
    output=reviewed.copy();output[accept]=chosen[accept]
    return output,accept
