"""Local SAM3 text segmentation. Runs separately from the legacy vendor stack.

Uses the exact detector tensors from the downloaded official HF checkpoint.
No manual points/boxes, color words, training, or previous semantic labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

import numpy as np
from PIL import Image

PROMPTS = [('plant stem', 1), ('branch', 1), ('leaf', 2),
           ('flower', 3), ('petal', 3), ('flower bud', 3),
           ('flower pot', 4), ('plant saucer', 4),
           ('soil', 5), ('moss', 5), ('potting substrate', 5)]
COLORS = np.array([[140,140,140],[245,155,35],[40,190,85],
                   [235,75,165],[55,130,240],[140,95,55]], np.uint8)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint, device):
    import torch
    from safetensors import safe_open
    from transformers import Sam3Config, Sam3Model, Sam3Processor, Sam3ImageProcessor, CLIPTokenizer
    raw = json.loads((checkpoint/'config.json').read_text(encoding='utf8'))
    config = Sam3Config(**raw.get('detector_config', raw))
    config._attn_implementation = 'sdpa'
    print('Constructing SAM3 detector', flush=True)
    model = Sam3Model(config)
    with safe_open(str(checkpoint/'model.safetensors'), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        prefix = 'detector_model.' if any(k.startswith('detector_model.') for k in keys) else ''
        state = {k[len(prefix):]:f.get_tensor(k) for k in keys if not prefix or k.startswith(prefix)}
    # Never accept an incompatible or partially initialized network.
    result = model.load_state_dict(state, strict=True, assign=True)
    del state
    print('Strict checkpoint loading:', result, flush=True)
    model = model.to(dtype=torch.bfloat16 if device == 'cuda' else torch.float32).to(device).eval()
    processor = Sam3Processor(image_processor=Sam3ImageProcessor(),
                              tokenizer=CLIPTokenizer.from_pretrained(str(checkpoint), local_files_only=True))
    return model, processor


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Prepared RGB camera directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=.25)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import transformers
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA PyTorch runtime is required for --device cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'views').mkdir(exist_ok=True)
    manifest = json.loads((args.source/'manifest.json').read_text(encoding='utf8'))
    config = dict(recipe_revision=1, checkpoint=str(args.checkpoint.resolve()),
                  weight_sha256=sha256(args.checkpoint/'model.safetensors'),
                  input_sha256=manifest['input_sha256'],
                  source_manifest_sha256=sha256(args.source/'manifest.json'),
                  prompts=PROMPTS, threshold=args.threshold,
                  mask_threshold=.5, precision='bfloat16' if args.device=='cuda' else 'float32',
                  torch=torch.__version__, transformers=transformers.__version__,
                  manual_annotations=False, prior_semantic_labels_used=False,
                  code_sha256=sha256(__file__))
    config['cache_key'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    previous = args.output/'inference_config.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf8'))['cache_key'] != config['cache_key']:
        raise ValueError('Inference recipe changed: use a new output directory')
    save_json(previous, config)
    save_json(args.output/'manifest.json', dict(manifest, rgb_source=str(args.source.resolve()),
              backend='SAM3 text masks', status='sam3_prediction'))
    model, processor = load_model(args.checkpoint, args.device)
    prompts = [processor(text=text, return_tensors='pt').to(args.device) for text,_ in PROMPTS]
    cameras = manifest['views'][args.start:]
    if args.limit:
        cameras = cameras[:args.limit]
    for camera in cameras:
        name = camera['name']
        target = args.output/'views'/f'{name}_scores.npz'
        if target.exists():
            with np.load(target) as cached:
                if str(cached['cache_key']) != config['cache_key']:
                    raise ValueError(f'Stale cache: {target}')
            print('Cached', name, flush=True)
            continue
        start = time.time()
        picture = Image.open(args.source/'views'/f'{name}.png').convert('RGB')
        with np.load(args.source/'views'/f'{name}_rays.npz') as ray:
            foreground = ray['face_id'] >= 0
        h,w = foreground.shape
        scores = np.zeros((h,w,6), np.float32)
        bud = np.zeros((h,w), np.float32)
        records = []
        img = processor(images=picture, return_tensors='pt')['pixel_values'].to(args.device, dtype=model.dtype)
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=img)
            print(name, 'vision encoded', round(time.time()-start, 1), flush=True)
            for (phrase,cls), text_inputs in zip(PROMPTS, prompts):
                output = model(vision_embeds=vision, **text_inputs)
                confidence = (output.pred_logits.sigmoid()*output.presence_logits.sigmoid()).flatten()
                keep = torch.nonzero(confidence >= args.threshold).flatten()
                for idx in keep.tolist():
                    # Resize one mask at a time to keep 8GB GPU memory bounded.
                    prob = F.interpolate(output.pred_masks[:,idx:idx+1].float(),(h,w),
                                         mode='bilinear',align_corners=False)[0,0].sigmoid().cpu().numpy()
                    mask = (prob >= .5)&foreground
                    if mask.sum() < 8:
                        continue
                    score = float(confidence[idx])
                    evidence = prob*score*mask
                    scores[...,cls] = np.maximum(scores[...,cls], evidence)
                    if phrase == 'flower bud':
                        bud = np.maximum(bud, evidence)
                    yy,xx = np.where(mask)
                    records.append(dict(phrase=phrase,class_id=cls,score=score,pixels=int(mask.sum()),
                                        box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
                print(name, phrase, 'instances', len(keep), flush=True)
                del output
            del vision,img
        np.savez_compressed(target, scores=scores.astype(np.float16),bud=bud.astype(np.float16),
                            cache_key=config['cache_key'])
        label = scores.argmax(2)
        overlay = np.asarray(picture).copy()
        selected = foreground&(scores.max(2)>0)
        overlay[selected] = (overlay[selected]*.3+COLORS[label[selected]]*.7).astype(np.uint8)
        Image.fromarray(overlay).save(args.output/'views'/f'{name}_overlay.jpg',quality=92)
        save_json(args.output/'views'/f'{name}_records.json',records)
        print(name, 'completed', round(time.time()-start,1),'seconds',
              'foreground_mask_coverage',round(float(selected.sum()/max(foreground.sum(),1)),3),
              'peak_gpu_GB',round(torch.cuda.max_memory_allocated()/1e9,2) if args.device=='cuda' else 0,
              flush=True)


if __name__ == '__main__':
    main()
