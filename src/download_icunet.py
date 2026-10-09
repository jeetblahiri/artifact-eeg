"""Download two pinned author files; verify hashes before local inference."""
import argparse
import urllib.request
from pathlib import Path
from common import ROOT,sha256,write_json

COMMIT='7f4f27dbf79c0909a0993f680209cf24c32f7791'
FILES={
 'model/cumbersome_model2.py':'be5a7a9350f36f06d35f76d8ab1d84d535d2a85441d584413785bb1044c427b0',
 'model/ICUNet/modelsave/checkpoint.pth.tar':'591d2a7819c4ddcd6280198645aae59c80e40c280a4ec7ac368068c33c2de75e'}

def download(destination):
    rows=[]
    for relative,digest in FILES.items():
        path=destination/relative;url=f'https://raw.githubusercontent.com/roseDwayane/AIEEG/{COMMIT}/{relative}'
        if path.exists():
            if sha256(path)!=digest:raise ValueError(f'Existing file differs from pinned source: {relative}')
        else:
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.download')
            request=urllib.request.Request(url,headers={'User-Agent':'artifact-eeg-reproduction'})
            with urllib.request.urlopen(request,timeout=90) as response,temporary.open('wb') as target:
                while chunk:=response.read(1048576):target.write(chunk)
            if sha256(temporary)!=digest:
                temporary.unlink();raise ValueError(f'Pinned SHA-256 mismatch: {relative}')
            temporary.replace(path)
        rows.append(dict(path=relative,url=url,sha256=digest,bytes=path.stat().st_size))
        print('Verified',relative,flush=True)
    write_json(destination/'download_manifest.json',dict(repository='https://github.com/roseDwayane/AIEEG',commit=COMMIT,
        paper_doi='10.1016/j.neuroimage.2022.119586',files=rows,
        scope='Files fetched from the original authors; not redistributed in this code release. Consult the source terms before reuse or redistribution.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,default=ROOT/'vendor/AIEEG')
    download(parser.parse_args().destination)
