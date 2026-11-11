"""Surface Potts regularization of independent multiview organ evidence.

Contract only actual mesh edges inside a spatial cell. Disconnected surfaces
remain disconnected even when close in space. All lengths scale with the plant;
there are no species, camera orientation or hand-picked position rules.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import (connected_components, maximum_flow,
                                 breadth_first_order)

VERSION = 'surface-organ-consistency-1.2'


def stem_shape_cost(features):
    """Continuous evidence against stem identity on locally broad sheets.

    High planarity alone is insufficient: a narrow leaf or a bent branch can
    have a planar neighbourhood. Require low linearity at BOTH scales, and
    gradually increase the cost. This does not choose between leaf and flower;
    independent RGB evidence and actual surface neighbours make that decision.
    """
    features=np.asarray(features,float)
    linear=np.maximum(features[:,0],features[:,3])
    planar=np.minimum(features[:,1],features[:,4])
    sheet=np.clip((planar-.22)/.38,0,1)*np.clip((.62-linear)/.32,0,1)
    return .90*sheet


def energy(labels, unary, edges, weights):
    return float(unary[np.arange(len(labels)), labels].sum()+
                 weights[labels[edges[:,0]]!=labels[edges[:,1]]].sum())


def alpha_expansion(unary, edges, weights, max_cycles=5):
    """Minimize a three-label Potts energy by exact binary expansion moves.

    The complete multi-label solution is approximate. Each accepted move must
    lower the original floating point energy; integer flow rounding cannot
    silently accept a worse segmentation.
    """
    unary=np.asarray(unary,float);edges=np.asarray(edges,np.int32)
    weights=np.asarray(weights,float)
    n,k=unary.shape
    if not np.isfinite(unary).all() or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Finite unaries and nonnegative boundary costs required')
    labels=unary.argmin(1).astype(np.int32)
    value=energy(labels,unary,edges,weights);history=[value]
    a,b=edges.T if len(edges) else (np.empty(0,int),np.empty(0,int))
    rows=np.arange(n)
    for _ in range(max_cycles):
        moved=False
        for alpha in range(k):
            e00=weights*(labels[a]!=labels[b])
            e01=weights*(labels[a]!=alpha);e10=weights*(labels[b]!=alpha)
            pair=np.maximum((e01+e10-e00)*.5,0)
            d0=unary[rows,labels].copy();d1=unary[:,alpha].copy()
            np.add.at(d1,a,e10-e00-pair)
            np.add.at(d1,b,e01-e00-pair)
            offset=np.minimum(d0,d1);d0-=offset;d1-=offset
            positive=np.r_[d0[d0>0],d1[d1>0],pair[pair>0]]
            unit=1000/max(float(np.median(positive)) if len(positive) else 1.,1e-12)
            def integer(v):return np.rint(np.maximum(v,0)*unit).clip(0,2**30).astype(np.int64)
            pair=integer(pair)
            graph=sparse.coo_matrix((np.r_[integer(d1),integer(d0),pair,pair],
                (np.r_[np.full(n,n),rows,a,b],np.r_[rows,np.full(n,n+1),b,a])),
                shape=(n+2,n+2)).tocsr()
            flow=maximum_flow(graph,n,n+1,method='dinic')
            residual=graph-flow.flow
            residual.data[residual.data<=0]=0;residual.eliminate_zeros()
            reachable=breadth_first_order(residual,n,directed=True,return_predecessors=False)
            keep=np.zeros(n+2,bool);keep[reachable]=True
            proposal=labels.copy();proposal[~keep[:n]]=alpha
            candidate=energy(proposal,unary,edges,weights)
            if candidate<value-max(abs(value)*1e-9,1e-10):
                labels,value=proposal,candidate;moved=True;history.append(value)
        if not moved:break
    return labels,history


def regularize_surface(xyz,tri,rgb,labels,total,votes,features,scale,
                       cell_ratio=.0025,smooth_ratio=.0012):
    """Use shared organ evidence to remove unsupported soft-label islands.

    A physical surface-boundary cost competes with area-weighted class evidence.
    No nearest-space edges are created. Hard classes and unknown are immutable.
    Thin, locally tubular old stems receive extra evidence, rather than being
    overwritten just because leaves surround them in a camera image.
    """
    from organ_geometry import mesh_edges,surface_areas
    xyz=np.asarray(xyz,float);labels=np.asarray(labels)
    soft=np.isin(labels,[1,2,3]);ids=np.flatnonzero(soft)
    if not len(ids):return labels.copy(),dict(version=VERSION,changed_vertices=0)
    area=surface_areas(xyz,tri)
    edges=mesh_edges(tri)
    edges=edges[soft[edges].all(1)]
    inverse=np.full(len(xyz),-1,np.int32);inverse[ids]=np.arange(len(ids))
    local=inverse[edges]
    cells=np.floor((xyz-xyz.min(0))/(scale*cell_ratio)).astype(np.int32)
    same=(cells[edges[:,0]]==cells[edges[:,1]]).all(1)
    links=local[same]
    contract=sparse.coo_matrix((np.ones(len(links),np.uint8),(links[:,0],links[:,1])),
                              shape=(len(ids),len(ids))).tocsr()
    n,group=connected_components(contract,directed=False)
    weights=np.maximum(area[ids],scale**2*1e-14)
    evidence=np.asarray(total[ids,1:4],float).copy()
    # The original classifier is a weak prior, never a hard label. This is
    # crucial when a whole pale leaf was previously called a branch.
    evidence[np.arange(len(ids)),labels[ids]-1]+=.30
    tubular=((features[ids,0]>.60)&(features[ids,3]>.55)&
             (features[ids,1]<.35)&(features[ids,4]<.35)&(labels[ids]==1))
    evidence[tubular,0]+=2.
    probability=evidence/np.maximum(evidence.sum(1,keepdims=True),1e-12)
    confidence=np.clip(np.asarray(total[ids,1:4]).sum(1)+.30,.30,3.)
    cost=(1-probability)*confidence[:,None]
    # Old branch labels must not win by default on broad, weakly observed
    # surfaces. A shape cost removes that bias without inventing leaf/flower
    # identity, votes, or a special RGB/position rule for the current plant.
    sheet_cost=stem_shape_cost(features[ids])
    a,b=edges.T
    ga,gb=group[local[:,0]],group[local[:,1]]
    length=np.maximum(np.linalg.norm(xyz[a]-xyz[b],axis=1),scale*1e-8)
    # Vertex areas / edge length approximate dual edge lengths, making the
    # tradeoff substantially less sensitive to triangulation density.
    boundary=(area[a]+area[b])/(3*length)
    color=np.linalg.norm(np.asarray(rgb[a],float)-np.asarray(rgb[b],float),axis=1)
    boundary*=np.exp(-(color/.18)**2)*.95+.05
    diff=ga!=gb
    pair=np.sort(np.column_stack([ga[diff],gb[diff]]),axis=1)
    pair,which=np.unique(pair,axis=0,return_inverse=True)
    boundary=np.bincount(which,weights=boundary[diff],minlength=len(pair))*scale*smooth_ratio
    # A completely unobserved disconnected surface has no leaf/flower identity.
    # Do not let the equal-cost argmin choose "leaf" merely by column order.
    graph=sparse.coo_matrix((np.ones(len(pair),np.uint8),(pair[:,0],pair[:,1])),shape=(n,n)).tocsr()
    components,component=connected_components(graph,directed=False)
    observed_alternative=np.any(np.asarray(votes[ids,2:4])>0,axis=1)
    supported=np.bincount(component[group],weights=observed_alternative,minlength=components)>0
    sheet_cost*=supported[component[group]]
    cost[:,0]+=sheet_cost
    unary=np.column_stack([np.bincount(group,weights=weights*cost[:,c],minlength=n)for c in range(3)])
    coarse,history=alpha_expansion(unary,pair,boundary)
    out=labels.copy();out[ids]=coarse[group]+1
    # Report surface-area transitions as well as vertex counts; fine tessellation
    # must not be mistaken for a proportion of botanical organs.
    stats=dict(version=VERSION,cell_ratio=cell_ratio,smooth_ratio=smooth_ratio,
        surface_nodes=n,boundaries=len(pair),energy_history=history,
        changed_vertices=int((out!=labels).sum()),
        changed_surface_area=float(area[out!=labels].sum()),
        hard_labels_preserved=bool(np.array_equal(out[~soft],labels[~soft])),
        tubular_prior_vertices=int(tubular.sum()),method='area-weighted surface Potts alpha expansion',
        sheet_stem_cost_max=.90,sheet_cost_vertices=int((sheet_cost>0).sum()),
        old_stem_sheet_vertices=int(((sheet_cost>.30)&(labels[ids]==1)).sum()),
        limitation='Approximate organ consistency; fused or badly reconstructed surfaces still require review')
    return out,stats


def refine_stem_rims(xyz,tri,rgb,labels,total,votes,features,scale,
                     cell_ratio=.0025,diffusion_ratio=.006,max_distance_ratio=.015):
    """Recover narrow branch-labelled ribbons attached along a broad organ.

    A rolled leaf/petal rim can look locally tubular. Surface heat attachment
    and covariance over the SAME mesh graph supply wider context. A tube which
    extends away from a blade stays linear at that scale. Existing leaf/flower
    labels, hard classes, and unknown are immutable in this targeted step.
    """
    from organ_geometry import mesh_edges,surface_areas
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse.linalg import factorized
    xyz=np.asarray(xyz,float);labels=np.asarray(labels)
    if not np.isfinite(scale)or scale<=0:raise ValueError('Positive finite plant scale required')
    soft=np.isin(labels,[1,2,3]);ids=np.flatnonzero(soft)
    empty=dict(version=VERSION,method='surface heat attachment and surface-neighbourhood shape',
        changed_vertices=0)
    if not len(ids)or not np.any(labels==1):return labels.copy(),empty
    area=surface_areas(xyz,tri)
    edge=mesh_edges(tri);edge=edge[soft[edge].all(1)]
    inverse=np.full(len(xyz),-1,np.int32);inverse[ids]=np.arange(len(ids))
    local=inverse[edge]
    cells=np.floor((xyz-xyz.min(0))/(scale*cell_ratio)).astype(np.int32)
    same=(cells[edge[:,0]]==cells[edge[:,1]]).all(1);links=local[same]
    n,group=connected_components(sparse.coo_matrix((np.ones(len(links),np.uint8),
        (links[:,0],links[:,1])),shape=(len(ids),len(ids))).tocsr(),directed=False)
    vertex_mass=np.maximum(area[ids],scale**2*1e-14)
    mass=np.bincount(group,weights=vertex_mass,minlength=n)
    a,b=edge.T;ga,gb=group[local[:,0]],group[local[:,1]]
    length=np.maximum(np.linalg.norm(xyz[a]-xyz[b],axis=1),scale*1e-8)
    rgb_gap=np.linalg.norm(np.asarray(rgb[a],float)-np.asarray(rgb[b],float),axis=1)
    # Dimensionless conductance = dual edge length / primal edge length.
    # Together with area mass this makes diffusion time a squared length and
    # preserves behaviour when model units or triangle density change.
    conduct=(area[a]+area[b])/(3*length**2)*(.8*np.exp(-(rgb_gap/.30)**2)+.2)
    diff=ga!=gb
    graph=sparse.coo_matrix((np.r_[conduct[diff],conduct[diff]],
        (np.r_[ga[diff],gb[diff]],np.r_[gb[diff],ga[diff]])),shape=(n,n)).tocsr()
    degree=np.asarray(graph.sum(1)).ravel();time=(scale*diffusion_ratio)**2
    solve=factorized((sparse.diags(mass+time*degree)-time*graph).tocsc())
    # Broad surface seeds need some independent RGB evidence. Their current
    # identity comes from the preceding multi-view/surface optimization.
    row=np.arange(len(xyz));current=np.clip(labels,0,5)
    seen=(votes[row,current]>0)|(total[row,current]>.05)
    seeds=np.isin(labels,[2,3])&(features[:,4]>.35)&(features[:,3]<.55)&seen
    if not np.any(seeds):return labels.copy(),empty
    def coarse_average(values):
        return np.column_stack([np.bincount(group,weights=vertex_mass*values[ids,k],
            minlength=n)/mass for k in range(values.shape[1])])
    heat=coarse_average(np.column_stack([seeds&(labels==c)for c in (2,3)]))
    # Coordinates are normalized before moments to avoid cancellation when a
    # model is translated far from its world origin or uses a different unit.
    position=(xyz-(xyz.min(0)+xyz.max(0))/2)/scale
    x,y,z=position.T
    moment=coarse_average(np.column_stack([x,y,z,x*x,y*y,z*z,x*y,x*z,y*z]))
    for _ in range(3):
        heat=np.column_stack([solve(mass*heat[:,k])for k in (0,1)])
        moment=np.column_stack([solve(mass*moment[:,k])for k in range(9)])
    heat=np.maximum(heat,0)
    covariance=np.zeros((n,3,3))
    for k in range(3):covariance[:,k,k]=moment[:,3+k]
    for k,(a,b)in enumerate(((0,1),(0,2),(1,2))):
        covariance[:,a,b]=covariance[:,b,a]=moment[:,6+k]
    covariance-=moment[:,:3,None]*moment[:,None,:3]
    eigen=np.linalg.eigvalsh(covariance).clip(0)
    linearity=(eigen[:,2]-eigen[:,1])/np.maximum(eigen[:,2],1e-12)
    strength=heat.sum(1);agreement=heat.max(1)/np.maximum(strength,1e-12)
    chosen=heat.argmax(1)+2
    # Short geodesic distance is checked on original vertices, not contracted
    # cell paths. No Euclidean shortcut, other organ, or hard support is crossed.
    a,b=edge.T
    travel=length*(1+(rgb_gap/.25)**2)
    surface=sparse.coo_matrix((np.r_[travel,travel],(np.r_[a,b],np.r_[b,a])),
                              shape=(len(xyz),len(xyz))).tocsr()
    distances=[]
    for cls in (2,3):
        starts=np.flatnonzero(seeds&(labels==cls))
        distances.append(dijkstra(surface,directed=False,indices=starts,min_only=True,
            limit=scale*max_distance_ratio)if len(starts)else np.full(len(xyz),np.inf))
    distance=np.column_stack(distances)[ids,chosen[group]-2]
    strong_stem=(votes[ids,1]>=2)&(total[ids,1]>=np.maximum(total[ids,2],total[ids,3]))
    accept=(labels[ids]==1)&(strength[group]>.16)&(agreement[group]>.80)&\
        (linearity[group]<.65)&(distance<scale*max_distance_ratio)&~strong_stem
    out=labels.copy();out[ids[accept]]=chosen[group[accept]]
    report=dict(version=VERSION,method=empty['method'],cell_ratio=cell_ratio,
        diffusion_ratio=diffusion_ratio,diffusion_steps=3,max_distance_ratio=max_distance_ratio,
        minimum_attachment=.16,minimum_agreement=.80,maximum_surface_linearity=.65,
        seed_vertices=int(seeds.sum()),changed_vertices=int(accept.sum()),
        changed_surface_area=float(area[ids[accept]].sum()),
        stem_to_leaf=int(((labels==1)&(out==2)).sum()),
        stem_to_flower=int(((labels==1)&(out==3)).sum()),
        nonstem_exact=bool(np.array_equal(out[labels!=1],labels[labels!=1])),
        strong_stem_vote_protected_vertices=int(strong_stem.sum()),
        limitation='Heuristic surface attachment; fused geometry and real stalk junctions still require review')
    return out,report
