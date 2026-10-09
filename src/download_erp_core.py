"""Fetch a complete, fixed two-task ERP CORE cohort with source checksums."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests
from common import ROOT,sha256,write_json

DEST=ROOT/'data'/'erp_core'
BASE='https://data.nemar.org/nm000132/v1.1.1/'

def checksum(path,algorithm):
    if algorithm=='sha256':return sha256(path)
    if algorithm=='git':
        h=hashlib.sha1();h.update(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    raise ValueError(algorithm)

def fetch(row):
    p=DEST/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==row['size'] and checksum(p,row['checksum_algorithm'])==row['checksum']:
        return dict(path=row['path'],bytes=row['size'],sha256=sha256(p),cached=True)
    partial=p.with_name(p.name+'.partial')
    for attempt in range(10):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>=row['size']:
                if offset==row['size'] and checksum(partial,row['checksum_algorithm'])==row['checksum']:
                    partial.replace(p)
                    return dict(path=row['path'],bytes=row['size'],sha256=sha256(p),cached=False)
                partial.unlink();offset=0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(row['bytes_url'],headers=headers,stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if offset and r.status_code==206:
                    assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-'),'Incorrect range response'
                    mode='ab'
                else:mode='wb' # A server ignoring Range must supply a new complete file.
                with partial.open(mode) as f:
                    for block in r.iter_content(512*1024):f.write(block)
            assert partial.stat().st_size==row['size'],f'Wrong byte count: {p}'
            assert checksum(partial,row['checksum_algorithm'])==row['checksum'],f'Checksum mismatch: {p}'
            partial.replace(p)
            return dict(path=row['path'],bytes=row['size'],sha256=sha256(p),cached=False)
        except Exception as exc:
            if attempt==9:raise
            print(f'Retrying {row["path"]}: {type(exc).__name__}; retaining partial bytes',flush=True)
            time.sleep(min(2*(attempt+1),10))

def main():
    manifest=json.loads((DEST/'manifest.json').read_text())
    rows=[r for r in manifest if
          (r['path'].startswith('sub-') and any(f'task-{t}_' in r['path'] for t in ['N170','P3']))
          or r['path'] in ['README.md','LICENSE','dataset_description.json','participants.tsv','task-N170_events.json','task-P3_events.json']]
    print(f'Fixed selection: 40 participants, N170 and P3; {len(rows)} files, {sum(r["size"] for r in rows)/1e9:.2f} GB',flush=True)
    completed=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fetch,r) for r in rows]
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed)%20==0:print(f'Verified {len(completed)}/{len(rows)} files',flush=True)
    write_json(DEST/'download_manifest.json',dict(dataset='ERP CORE',nemar_version='v1.1.1',
        tasks=['N170','P3'],participants=list(range(1,41)),base_url=BASE,
        selection='Complete two-task cohort fixed before EEG scoring; no outcome-based file selection',
        published_manifest_sha256=sha256(DEST/'manifest.json'),files=sorted(completed,key=lambda r:r['path'])))
    print('Complete ERP CORE two-task cohort downloaded and source checksums verified.',flush=True)

if __name__=='__main__':main()
