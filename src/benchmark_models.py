"""Architecture reproductions of the official EEGdenoiseNet CNN baselines.

These are retrained models, not unpublished or official pretrained weights.
The pinned TensorFlow source is the specification; protocol changes are recorded.
"""
import json,copy,time
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
import torch
from torch import nn
from erp_validation import baseline,stem

C=json.loads((ROOT/'causal_revision_config.json').read_text())
OUT=ROOT/'results/causal_revision';CACHE=ROOT/'data/causal_revision'
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_num_threads(2)

def block(i,o,k):
    # Keras epsilon=.001; moving-stat decay=.99 corresponds to momentum=.01.
    return nn.Sequential(nn.Conv1d(i,o,k,padding=k//2),nn.BatchNorm1d(o,eps=.001,momentum=.01),nn.ReLU())

class Residual(nn.Module):
    def __init__(self,k):
        super().__init__();self.layers=nn.Sequential(block(32,32,k),block(32,16,k),block(16,32,k))
    def forward(self,x):return x+self.layers(x)

class BenchmarkCNN(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.architecture=architecture
        if architecture=='simple_cnn':
            self.conv=nn.Sequential(*[nn.Sequential(block(1 if j==0 else 64,64,3),nn.Dropout(.3)) for j in range(4)])
            width=64
        elif architecture=='complex_cnn':
            self.first=block(1,32,5)
            self.branches=nn.ModuleList([nn.Sequential(Residual(k),Residual(k)) for k in [3,5,7]])
            self.last=block(96,32,1);width=32
        else:raise ValueError(architecture)
        self.dense=nn.Linear(512*width,512)
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):
        if self.architecture=='simple_cnn':x=self.conv(x)
        else:
            x=self.first(x);x=self.last(torch.cat([b(x) for b in self.branches],1))
        # Keras Flatten sees [time,channel], not PyTorch [channel,time].
        return self.dense(x.transpose(1,2).reshape(len(x),-1))

def predict(model,x,batch=256):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(x),batch):
            a=torch.from_numpy(np.asarray(x[i:i+batch],dtype=np.float32)).to(DEVICE)
            out.append(model(a[:,None]).cpu().numpy())
    return np.concatenate(out)

def train(architecture,seed):
    dst=OUT/f'{architecture}_{seed}.pth';report=dst.with_suffix('.json')
    if dst.exists() and report.exists():return json.loads(report.read_text())
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=BenchmarkCNN(architecture).to(DEVICE);tc=C['models']
    opt=torch.optim.RMSprop(model.parameters(),lr=tc['learning_rate'],alpha=tc['rho'],eps=tc['epsilon'])
    tr=dict(np.load(ROOT/'data/mixtures/EOG_train.npz'));va=dict(np.load(ROOT/'data/mixtures/EOG_val.npz'))
    history=[];best=float('inf');state=None;start=time.monotonic()
    for ep in range(1,tc['epochs']+1):
        model.train();loss_sum=0.
        for idx in np.array_split(rng.permutation(len(tr['x'])),int(np.ceil(len(tr['x'])/tc['batch_size']))):
            x=torch.from_numpy(tr['x'][idx]).to(DEVICE);y=torch.from_numpy(tr['s'][idx]).to(DEVICE)
            opt.zero_grad(set_to_none=True);loss=((model(x[:,None])-y)**2).mean();loss.backward();opt.step()
            loss_sum+=float(loss.detach().cpu())*len(idx)
        val=float(np.mean((predict(model,va['x'],batch=40)-va['s'])**2));history.append(dict(epoch=ep,train_mse=loss_sum/len(tr['x']),validation_mse=val))
        if val<best:best=val;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=ep
        print(architecture,seed,'epoch',ep,'validation',round(val,7),'seconds',round(time.monotonic()-start),flush=True)
    torch.save(state,dst)
    r=dict(architecture=architecture,seed=seed,parameters=sum(p.numel() for p in model.parameters()),selected_epoch=best_epoch,
        validation_mse=best,history=history,device=DEVICE,checkpoint_sha256=sha256(dst),protocol=C['models'],
        config_sha256=sha256(ROOT/'causal_revision_config.json'),code_sha256=sha256(Path(__file__)),
        source_sha256=sha256(ROOT/'vendor/EEGdenoiseNet/Network_structure.py'),
        train_sha256=sha256(ROOT/'data/mixtures/EOG_train.npz'),validation_sha256=sha256(ROOT/'data/mixtures/EOG_val.npz'))
    write_json(report,r);return r

def load(architecture,seed):
    model=BenchmarkCNN(architecture);model.load_state_dict(torch.load(OUT/f'{architecture}_{seed}.pth',map_location='cpu',weights_only=True),strict=True)
    return model.to(DEVICE).eval()

def apply(x,models,return_seeds=False):
    shape=x.shape;a=x.reshape(-1,512);scale=a.std(1,keepdims=True)
    if np.any(scale<1e-9):raise ValueError('Degenerate input')
    seeds=np.stack([baseline((predict(m,a/scale)*scale).reshape(shape)) for m in models])
    return seeds if return_seeds else seeds.mean(0)

def main(force_inference=False):
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);report=[]
    for architecture in C['models']['architectures']:
        for seed in C['models']['seeds']:report.append(train(architecture,seed))
        models=[load(architecture,s) for s in C['models']['seeds']]
        te=dict(np.load(ROOT/'data/mixtures/EOG_test.npz'))
        pred=np.stack([predict(m,te['x']) for m in models]);np.save(OUT/f'{architecture}_test_predictions.npy',pred)
        losses=np.mean((pred-te['s'])**2,axis=(1,2))
        for task in ['N170','P3']:
            for s in range(1,41):
                dst=CACHE/f'{stem(task,s)}_{architecture}.npy'
                if dst.exists() and not force_inference:continue
                x=np.load(ROOT/f'data/erp_validation/{stem(task,s)}_parent.npy')
                seeds=apply(x,models,return_seeds=True)
                np.save(dst,seeds.mean(0).astype(np.float32))
                if s in C['evaluation']:
                    for seed,z in zip(C['models']['seeds'],seeds):np.save(CACHE/f'{stem(task,s)}_{architecture}_{seed}.npy',z.astype(np.float32))
                print('ERP',architecture,task,s,flush=True)
        write_json(OUT/f'{architecture}_deployment.json',dict(test_mse_each_seed=losses.tolist(),mean_seed_test_mse=float(losses.mean()),
            ensemble_test_mse=float(np.mean((pred.mean(0)-te['s'])**2)),identity_test_mse=float(np.mean((te['x']-te['s'])**2)),
            seeds=C['models']['seeds'],erp_input='Per-channel window-SD normalized 0.1-30-Hz parent, scale restored, baseline reapplied',
            ensemble='Equal seed average declared before new ERP outcomes; individual seed outputs also assessed',inference_batch_size=256,erp_fine_tuning=False))
    write_json(OUT/'benchmark_training.json',dict(models=report,config_sha256=sha256(ROOT/'causal_revision_config.json')))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--force-inference',action='store_true');args=parser.parse_args()
    main(force_inference=args.force_inference)
