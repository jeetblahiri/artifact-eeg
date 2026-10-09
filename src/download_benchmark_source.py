"""Fetch hash-pinned published architecture specifications and original license."""
from pathlib import Path
import urllib.request
from common import ROOT,sha256,write_json

COMMIT='8d290661146c7189c98cc04812d37371d4b9426c'
FILES={
 'Network_structure.py':('code/benchmark_networks/Network_structure.py','6a49fbc7e8ca7e376c8cd5704240392b99f9789340958a695f8a727d413270ba'),
 'main.py':('code/benchmark_networks/main.py','c5663c5d18cc0a4703b8af27117b883552da22bbd542d423b66cd9ee114add8e'),
 'train_method.py':('code/benchmark_networks/train_method.py','3c8c0774456cb2febaee8693b062703bf54cad3ef8153447c5f3001ae64ec26e'),
 'LICENSE':('LICENSE','9edecc3e64eff42ce4c7381349279fc9cdbc760b05384ae499c044bf8c40d010')}

def main():
    destination=ROOT/'vendor/EEGdenoiseNet';destination.mkdir(parents=True,exist_ok=True);rows=[]
    for name,(relative,digest) in FILES.items():
        p=destination/name;url=f'https://raw.githubusercontent.com/ncclabsustech/EEGdenoiseNet/{COMMIT}/{relative}'
        if not p.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'artifact-eeg-reproduction'})
            with urllib.request.urlopen(request,timeout=90) as response:p.write_bytes(response.read())
        if sha256(p)!=digest:raise ValueError('Pinned source differs: '+name)
        rows.append(dict(path=name,url=url,sha256=digest));print('Verified',name,flush=True)
    write_json(destination/'download_manifest.json',dict(commit=COMMIT,files=rows,paper_doi='10.1088/1741-2552/ac2bf8',
        scope='Architecture and original training specification, not pretrained checkpoints. Public model reproductions require retraining.'))

if __name__=='__main__':main()
