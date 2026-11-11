"""Download public GroundingDINO weights and official MobileSAM sources.

Does not request or bypass access to gated SAM3 checkpoints.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models' / 'grounding-dino-tiny'
SOURCE = ROOT / 'third_party' / 'MobileSAM'


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return {'path': str(path), 'bytes': path.stat().st_size, 'cached': True}
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=40) as response:
        expected = response.headers.get('Content-Length')
        sha = hashlib.sha256()
        total = 0
        partial = path.with_suffix(path.suffix + '.partial')
        with partial.open('wb') as f:
            while block := response.read(1024 * 1024):
                f.write(block)
                sha.update(block)
                total += len(block)
        if expected and total != int(expected):
            raise RuntimeError(f'Incomplete download: {path}')
        partial.replace(path)
        return {'url': url, 'path': str(path), 'bytes': total,
                'sha256': sha.hexdigest(), 'revision': response.headers.get('x-repo-commit')}


def download_weights(path=None, url=None, size=689359096):
    path = path or DEST / 'model.safetensors'
    if path.exists(): return {'path': str(path), 'bytes': path.stat().st_size, 'cached': True}
    url = url or 'https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/model.safetensors?download=true'
    step = 4*1024*1024
    parts = path.parent/'parts'; parts.mkdir(parents=True, exist_ok=True)
    def part(i):
        start, end = i*step, min((i+1)*step, size)-1
        target = parts/f'{i:04d}.part'
        if target.exists() and target.stat().st_size == end-start+1: return target
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={'Range': f'bytes={start}-{end}', 'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(request,timeout=25) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Incorrect range response')
                    content = r.read(end-start+2)
                if len(content) != end-start+1: raise ValueError('Short range response')
                target.write_bytes(content)
                return target
            except Exception:
                if attempt == 3: raise
                time.sleep(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        completed=0
        futures=[pool.submit(part,i) for i in range((size+step-1)//step)]
        for future in concurrent.futures.as_completed(futures):
            future.result();completed+=1
            if completed%10==0:print('Weight ranges',completed,'/',len(futures),flush=True)
    checksum=hashlib.sha256()
    temp=path.with_suffix('.assembling')
    with temp.open('wb') as f:
        for i in range((size+step-1)//step):
            content=(parts/f'{i:04d}.part').read_bytes();f.write(content);checksum.update(content)
    if temp.stat().st_size!=size: raise ValueError('Incomplete assembled weights')
    temp.replace(path)
    return {'url':url,'path':str(path),'bytes':size,'sha256':checksum.hexdigest()}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    files = ['config.json', 'preprocessor_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json']
    jobs = [(f'https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/main/{f}', DEST / f) for f in files]
    source_files = ['mobile_sam/__init__.py', 'mobile_sam/build_sam.py', 'mobile_sam/predictor.py',
                    'mobile_sam/automatic_mask_generator.py', 'mobile_sam/modeling/__init__.py',
                    'mobile_sam/modeling/sam.py', 'mobile_sam/modeling/tiny_vit_sam.py',
                    'mobile_sam/modeling/image_encoder.py', 'mobile_sam/modeling/mask_decoder.py',
                    'mobile_sam/modeling/prompt_encoder.py', 'mobile_sam/modeling/transformer.py',
                    'mobile_sam/modeling/common.py', 'mobile_sam/utils/__init__.py',
                    'mobile_sam/utils/transforms.py', 'mobile_sam/utils/amg.py', 'LICENSE']
    jobs += [(f'https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/{f}', SOURCE / f) for f in source_files]
    jobs.append(('https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt',
                 ROOT / 'models' / 'mobile_sam_parallel.pt'))
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        def retry_fetch(u,p):
            for attempt in range(3):
                try:return fetch(u,p)
                except Exception:
                    if attempt==2:raise
        futures = {pool.submit(retry_fetch, u, p): (u, p) for u, p in jobs}
        for future in concurrent.futures.as_completed(futures):
            u, p = futures[future]
            try:
                r = future.result()
                records.append(r)
                print(p.name, r['bytes'], flush=True)
            except Exception as exc:
                records.append({'url': u, 'path': str(p), 'error': str(exc)})
                print('ERROR', p.name, str(exc), flush=True)
    records.append(download_weights())
    (ROOT / 'models' / 'downloads.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    if any('error' in r for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    if '--clipseg' in sys.argv:
        sys.stdout.reconfigure(encoding='utf-8')
        dest=ROOT/'models'/'clipseg-rd64-refined'
        base='https://hf-mirror.com/CIDAS/clipseg-rd64-refined/resolve/main/'
        names=['config.json','preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json']
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            records=list(pool.map(lambda n:fetch(base+n,dest/n),names))
        records.append(download_weights(dest/'model.safetensors',base+'model.safetensors?download=true',603049496))
        (dest/'downloads.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        main()
