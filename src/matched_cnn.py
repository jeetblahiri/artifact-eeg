"""Paired-target nonlinear fits: shared initialization, inputs and fixed duration."""
import json,time,hashlib
from pathlib import Path
from common import ROOT,sha256,write_json
import numpy as np
import torch
from benchmark_models import BenchmarkCNN,DEVICE,predict

OUT=ROOT/'results/nonlinear_revision'
CFG=json.loads((ROOT/'nonlinear_revision_config.json').read_text())


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare_design():
    arrays={key:[] for key in ['x','vx',*CFG['targets'],*['validation_'+t for t in CFG['targets']]]};sources={}
    for task in CFG['tasks']:
        source=ROOT/f'results/causal_revision/{task}_target_design.npz';sources[task]=sha256(source)
        data=np.load(source)
        for key in arrays:
            n=CFG['validation_rows_per_task'] if key=='vx' or key.startswith('validation_') else CFG['fit_rows_per_task']
            arrays[key].append(data[key][:n].astype(np.float32))
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    dst=OUT/'matched_design.npz'
    if dst.exists():
        old=np.load(dst)
        for k,v in arrays.items():np.testing.assert_array_equal(old[k],v)
    else:np.savez_compressed(dst,**arrays)
    write_json(OUT/'design_provenance.json',dict(source_design_sha256=sources,source_causal_config_sha256=sha256(ROOT/'causal_revision_config.json'),
        first_fixed_rows_per_task=CFG['fit_rows_per_task'],validation_rows_per_task=CFG['validation_rows_per_task'],
        tasks_equal_weighted=True,training_people=json.loads((ROOT/'causal_revision_config.json').read_text())['development'],
        input_design_sha256=sha256(dst),configuration_sha256=sha256(ROOT/'nonlinear_revision_config.json')))
    return arrays


def train(architecture,seed,target,data):
    tag=f'{architecture}_{seed}_{target}';dst=OUT/f'{tag}.pth';report=dst.with_suffix('.json')
    if dst.exists() and report.exists():
        r=json.loads(report.read_text());assert r['config_sha256']==sha256(ROOT/'nonlinear_revision_config.json');assert r['checkpoint_sha256']==sha256(dst);return r
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=BenchmarkCNN(architecture).to(DEVICE);initial=state_hash(model.state_dict())
    opt=torch.optim.RMSprop(model.parameters(),lr=CFG['learning_rate'],alpha=CFG['rho'],eps=CFG['epsilon'],weight_decay=CFG['weight_decay'])
    history=[];start=time.monotonic();batch_hash=hashlib.sha256()
    x,y=data['x'],data[target];vx,vy=data['vx'],data['validation_'+target]
    # Same data ordering and random-number consumption in every paired target arm.
    for epoch in range(1,CFG['epochs']+1):
        model.train();loss_sum=0.;order=rng.permutation(len(x));batch_hash.update(order.astype('<i8').tobytes())
        for at in range(0,len(order),CFG['batch_size']):
            idx=order[at:at+CFG['batch_size']]
            a=torch.from_numpy(x[idx]).to(DEVICE);b=torch.from_numpy(y[idx]).to(DEVICE)
            opt.zero_grad(set_to_none=True);loss=((model(a[:,None])-b)**2).mean();loss.backward();opt.step();loss_sum+=float(loss.detach().cpu())*len(idx)
        val=float(np.mean((predict(model,vx,batch=CFG['inference_batch_size'])-vy)**2))
        history.append(dict(epoch=epoch,train_mse=loss_sum/len(x),validation_mse=val))
        print(tag,'epoch',epoch,'validation',round(val,6),'seconds',round(time.monotonic()-start),flush=True)
        if epoch%10==0:write_json(OUT/f'{tag}_progress.json',dict(history=history,fixed_primary_epoch=CFG['epochs'],code_sha256=sha256(Path(__file__))))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save(state,dst)
    pred=predict(model,vx,batch=CFG['inference_batch_size']);np.save(OUT/f'{tag}_validation_predictions.npy',pred)
    cross_loss={t:float(np.mean((pred-data['validation_'+t])**2)) for t in CFG['targets']}
    r=dict(architecture=architecture,seed=seed,target=target,parameters=sum(p.numel() for p in model.parameters()),selected_epoch=CFG['epochs'],
        selection='Fixed final epoch, validation diagnostic only',initial_state_sha256=initial,minibatch_order_sha256=batch_hash.hexdigest(),
        history=history,cross_target_validation_mse=cross_loss,checkpoint_sha256=sha256(dst),device=DEVICE,
        input_design_sha256=sha256(OUT/'matched_design.npz'),config_sha256=sha256(ROOT/'nonlinear_revision_config.json'),
        architecture_code_sha256=sha256(ROOT/'src/benchmark_models.py'),code_sha256=sha256(Path(__file__)),elapsed_seconds=time.monotonic()-start)
    write_json(report,r);return r


def load(architecture,seed,target):
    tag=f'{architecture}_{seed}_{target}';m=BenchmarkCNN(architecture)
    m.load_state_dict(torch.load(OUT/f'{tag}.pth',map_location='cpu',weights_only=True),strict=True)
    return m.to(DEVICE).eval()


def main():
    OUT.mkdir(exist_ok=True);data=prepare_design();reports=[]
    for architecture in CFG['architectures']:
        for seed in CFG['seeds']:
            group=[train(architecture,seed,target,data) for target in CFG['targets']]
            for field in ['initial_state_sha256','minibatch_order_sha256','input_design_sha256','selected_epoch']:
                assert len({r[field] for r in group})==1,(architecture,seed,field)
            reports.extend(group)
    write_json(OUT/'matched_training.json',dict(models=reports,config_sha256=sha256(ROOT/'nonlinear_revision_config.json'),
        common_primary_training_duration=True,no_validation_or_erp_checkpoint_selection=True,paired_initialization_and_batches_verified=True))


if __name__=='__main__':main()
