"""Render true final poses side by side and audit class-level exposure."""
from pathlib import Path
import argparse
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plant_scene import read_json,write_json
from replay_watering_viewer import load_replay,apply_frame


def class_costs(report):
    totals={}
    for e in report['contact_risk']['simulated']['episodes']:
        cid=e['class_id'];totals[cid]=totals.get(cid,0.)+e['exposure_integral']*e['persistence_factor']
    return totals


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True);ap.add_argument('--after',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),22) if fontpath.exists() else ImageFont.load_default()
    sheet=Image.new('RGB',(1920,920),'#142133');reports=[]
    for k,path in enumerate((args.before,args.after)):
        directory,report,scene,a=load_replay(root,path)
        model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
        apply_frame(model,data,a,len(a['time_s'])-1)
        cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
        cam.lookat[:]=[.28,.20,.245];cam.distance=.43;cam.azimuth=315;cam.elevation=-18
        opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.geomgroup[3:5]=0;opt.sitegroup[:]=0
        with mujoco.Renderer(model,width=960,height=770) as renderer:
            renderer.update_scene(data,camera=cam,scene_option=opt)
            sheet.paste(Image.fromarray(renderer.render()),(k*960,150))
        cost=report['contact_risk']['simulated']['persistence_cost']
        excursion=np.rad2deg(np.ptp(a['q_actual_rad'][:,:5],axis=0))
        d=ImageDraw.Draw(sheet)
        title='Previous restricted attitude' if k==0 else 'New side approach + free wrist attitude'
        d.text((k*960+18,12),title,font=font,fill='white')
        d.text((k*960+18,45),f'Soft persistence score: {cost:.2f} | soil height: {report["final_height_above_soil_mm"]:.2f} mm',font=font,fill='#d9e9f4')
        d.text((k*960+18,78),f'Wrist roll travel range: {excursion[4]:.1f} deg | wrist pitch: {excursion[3]:.1f} deg',font=font,fill='#ffcd78')
        d.text((k*960+18,111),'Saved simulated pose; geometric contact score, not measured damage',font=font,fill='#d9e9f4')
        reports.append(dict(output=directory.relative_to(root).as_posix(),duration_s=report['duration_s'],
            persistence_cost=cost,exposure_integral=report['contact_risk']['simulated']['exposure_integral'],
            class_persistence_costs=class_costs(report),joint_excursion_deg=excursion.tolist(),
            minimum_hard_gap_mm=report['simulated_min_hard_clearance_mm']))
    args.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(args.output)
    write_json(args.output.with_suffix('.json'),dict(before=reports[0],after=reports[1],
        reduction_fraction=1-reports[1]['persistence_cost']/reports[0]['persistence_cost']))
    print(f'CONTACT_COMPARISON: {args.output}',flush=True)


if __name__=='__main__':main()
