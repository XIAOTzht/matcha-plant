"""Plot actual saved planned joint profiles; no added motion or interpolation art."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plant_scene import read_json
from trajectory_tools import time_parameterize


def font(size):
    for path in ('C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def plot(path):
    root=Path(__file__).resolve().parents[1];path=Path(path).resolve()
    report=read_json(path/'path_report.json');source=root/report['source_path']
    traces=[]
    for directory in (source,path):
        with np.load(directory/'trajectory_q.npz')as data:q=data['q_arm_rad'].copy()
        own=read_json(directory/'path_report.json')
        plan=read_json(directory/own['timing_plan_relative'])if own.get('timing_plan_relative')else None
        traces.append(time_parameterize(q,own.get('timing_config',report['timing_config']),.005,plan))
    width,height=1160,1030
    im=Image.new('RGB',(width,height),'#111c2c');draw=ImageDraw.Draw(im)
    draw.text((35,18),'SO-101 | planned joint coordination',font=font(28),fill='white')
    draw.text((35,57),'Gray: baseline    Cyan: selected    Five arm joints; gripper stays closed',font=font(19),fill='#b8e4f4')
    names=['Shoulder pan','Shoulder lift','Elbow flex','Wrist flex','Wrist roll']
    left,right=110,1110
    for j,name in enumerate(names):
        top=110+j*158;bottom=top+117
        values=[np.rad2deg(t['q_ref'][:,j])for t in traces]
        low=min(v.min()for v in values);high=max(v.max()for v in values)
        pad=max(2.,(high-low)*.14);low-=pad;high+=pad
        draw.text((left,top-25),name,font=font(19),fill='white')
        for frac in (0.,.5,1.):
            y=bottom-frac*(bottom-top)
            draw.line((left,y,right,y),fill='#2e4154',width=1)
            draw.text((20,y-8),f'{low+frac*(high-low):.0f} deg',font=font(14),fill='#acc3d4')
        for k,(t,v)in enumerate(zip(traces,values)):
            points=np.c_[left+(right-left)*t['time_s']/t['time_s'][-1],bottom-(v-low)/(high-low)*(bottom-top)]
            draw.line([tuple(x)for x in points],fill=('#8492a3','#23d6ec')[k],width=(3,3)[k])
        spans=[np.ptp(v)for v in values]
        draw.text((right-310,top-25),f'Span: {spans[0]:.1f} -> {spans[1]:.1f} deg',font=font(17),fill='#a1e6ec')
    for pct in (0,25,50,75,100):
        draw.text((left+(right-left)*pct/100-14,898),str(pct),font=font(15),fill='#acc3d4')
    draw.text((440,925),'Normalized motion time (%)',font=font(18),fill='#d5e3ee')
    baseline=(report.get('coordination')or report.get('natural_search'))['baseline_score'];selected=report['selected_score']
    baseline_duration=baseline.get('duration_s',baseline.get('motion',{}).get('duration_s'))
    draw.text((35,965),f"Duration: {baseline_duration:.2f} -> {selected['duration_s']:.2f} s     Weighted exposure cost: {baseline['persistence_cost']:.2f} -> {selected['persistence_cost']:.2f}",font=font(20),fill='white')
    draw.text((35,998),'Reference trajectory comparison. Actual simulated motion is checked separately in validation_report.json.',font=font(15),fill='#c2cdd9')
    out=path/'joint_coordination.png';im.save(out)
    # Show velocities against physical seconds, exposing interior hesitation.
    speed_image=Image.new('RGB',(width,height),'#111c2c');d=ImageDraw.Draw(speed_image)
    d.text((35,18),'SO-101 | joint velocity and motion flow',font=font(28),fill='white')
    d.text((35,57),'Gray: baseline    Cyan: selected    Same playback speed; holds included',font=font(19),fill='#b8e4f4')
    total=max(t['time_s'][-1]for t in traces)
    for j,name in enumerate(names):
        top=110+j*158;bottom=top+117
        values=[np.rad2deg(t['qd_ref'][:,j])for t in traces];bound=max(1.,max(abs(v).max()for v in values)*1.15)
        d.text((left,top-25),name,font=font(19),fill='white')
        for value in (-bound,0,bound):
            y=(top+bottom)/2-value/bound*(bottom-top)/2
            d.line((left,y,right,y),fill='#2e4154');d.text((14,y-8),f'{value:.0f} deg/s',font=font(13),fill='#acc3d4')
        for k,(t,v)in enumerate(zip(traces,values)):
            pts=np.c_[left+(right-left)*t['time_s']/total,(top+bottom)/2-v/bound*(bottom-top)/2]
            d.line([tuple(x)for x in pts],fill=('#8492a3','#23d6ec')[k],width=3)
    d.text((420,925),'Elapsed time in seconds',font=font(18),fill='#d5e3ee')
    for t in np.linspace(0,total,6):d.text((left+(right-left)*t/total-12,898),f'{t:.1f}',font=font(15),fill='#acc3d4')
    d.text((35,977),'Saved C3 reference profiles. Dynamics and actual contact penalties are checked separately.',font=font(18),fill='white')
    speed_image.save(path/'joint_velocity_comparison.png')
    print('JOINT_COORDINATION_PLOT',out,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--path',type=Path,required=True)
    plot(ap.parse_args().path)
