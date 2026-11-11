"""Finite, two-sided spray-ray intersection against original mesh triangles."""
import numpy as np


class SprayRayQuery:
    def __init__(self,xyz,faces,labels):
        self.xyz=xyz;self.faces=faces;self.labels=labels
        triangles=xyz[faces]
        self.lower=triangles.min(axis=1);self.upper=triangles.max(axis=1)

    def first_hit(self,origin,direction,max_distance=.04):
        origin=np.asarray(origin,dtype=float);direction=np.asarray(direction,dtype=float)
        norm=np.linalg.norm(direction)
        if norm<1e-12:raise ValueError('Spray direction must be nonzero')
        direction=direction/norm;end=origin+max_distance*direction
        lower=np.minimum(origin,end)-1e-10;upper=np.maximum(origin,end)+1e-10
        ids=np.flatnonzero(np.all(self.lower<=upper,axis=1)&np.all(self.upper>=lower,axis=1))
        if not len(ids):return dict(first_hit_class=None,range_mm=None)
        t=self.xyz[self.faces[ids]]
        e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0]
        h=np.cross(direction,e2);a=np.einsum('ij,ij->i',e1,h)
        keep=abs(a)>1e-16
        ids,t,e1,e2,h,a=ids[keep],t[keep],e1[keep],e2[keep],h[keep],a[keep]
        inverse=1/a;s=origin-t[:,0]
        u=inverse*np.einsum('ij,ij->i',s,h)
        q=np.cross(s,e1);v=inverse*(q@direction)
        distance=inverse*np.einsum('ij,ij->i',e2,q)
        hits=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(distance>1e-7)&(distance<=max_distance)
        valid=np.flatnonzero(hits)
        if not len(valid):return dict(first_hit_class=None,range_mm=None)
        k=valid[np.argmin(distance[valid])];face=int(ids[k])
        return dict(first_hit_class=int(self.labels[face]),range_mm=float(distance[k]*1000),source_face_index=face)
