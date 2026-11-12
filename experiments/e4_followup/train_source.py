"""Single prespecified class-weight intervention; never opens an external test split."""
from __future__ import annotations
import gc
import json
import math
import os
import time
from pathlib import Path

from experiments.e4.common import (ROOT, RESULTS as ORIGINAL, base_model, load_frame, config as original_config,
    seed_all, setup, tokenizer, metrics, score_texts, sha256, write_json, write_csv, now, environment, check_space)
from experiments.e4.train import optimizer_for, encoded_batch
import numpy as np
import pandas as pd
import torch

OUT=ROOT/'results/e4_followup'
CP=ROOT/'models/e4_followup/benign_weight2'
CFG=ROOT/'configs/e4_followup.json'


def candidate_metrics(y,s):
    # Efficient confusion counts at all unique score thresholds, using score >= threshold.
    values,inv=np.unique(s,return_inverse=True)
    pos=np.bincount(inv,weights=y,minlength=len(values))
    neg=np.bincount(inv,weights=1-y,minlength=len(values))
    tp=np.cumsum(pos[::-1])[::-1];fp=np.cumsum(neg[::-1])[::-1]
    tp=np.r_[tp,0];fp=np.r_[fp,0];threshold=np.r_[values,np.nextafter(values[-1],np.inf)]
    fn=sum(y)-tp;tn=len(y)-sum(y)-fp
    f1p=2*tp/np.maximum(2*tp+fp+fn,1)
    f1n=2*tn/np.maximum(2*tn+fp+fn,1)
    return pd.DataFrame({'threshold':threshold,'tp':tp.astype(int),'fp':fp.astype(int),
        'tn':tn.astype(int),'fn':fn.astype(int),'macro_f1':(f1p+f1n)/2,
        'attack_recall':tp/sum(y),'fpr':fp/(len(y)-sum(y))})


def policies(y,s,cfg):
    candidates=candidate_metrics(y,s)
    tf1=candidates.sort_values(['macro_f1','threshold'],ascending=[False,False]).iloc[0]
    t5=candidates[candidates.fpr<=.05].sort_values(['attack_recall','threshold'],ascending=[False,False]).iloc[0]
    r95=candidates[candidates.attack_recall>=cfg['R95_recall_floor']].sort_values('threshold',ascending=False).iloc[0]
    thresholds={'default_0.5':.5,'tF1_reference':float(tf1.threshold),'t5_reference':float(t5.threshold),'R95_exploratory':float(r95.threshold)}
    return [{'policy':k,'threshold':v,**metrics(y,s,v)} for k,v in thresholds.items()]


def main():
    setup();check_space(5)
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CFG.read_text())
    if (OUT/'selection_frozen.json').exists():
        raise RuntimeError('Follow-up already frozen; use saved outputs, do not retrain after external evaluation.')
    provenance={'time_utc':now(),'config_sha256':sha256(CFG),'config':cfg,
        'training_code_sha256':sha256(Path(__file__)),'original_training_config_sha256':sha256(ROOT/'configs/e4.json'),
        'environment':environment()}
    frozen=OUT/'protocol_frozen.json'
    if frozen.exists():
        old=json.loads(frozen.read_text())
        assert old['config_sha256']==provenance['config_sha256'] and old['training_code_sha256']==provenance['training_code_sha256']
    else:write_json(frozen,provenance)
    train,dev=load_frame('d1_train'),load_frame('d1_validation')
    tok=tokenizer();enc=tok(train.text.tolist(),truncation=True,max_length=256,padding=False)
    labels=train.label.to_numpy(dtype=np.int64)
    seed_all(cfg['seed']);model=base_model().cuda()
    optimizer=optimizer_for(model,cfg['learning_rate'])
    steps=math.ceil(len(train)/cfg['batch_size']);total=steps*cfg['epochs'];warmup=round(.1*total)
    schedule=lambda step:step/max(1,warmup) if step<warmup else max(0.,(total-step)/max(1,total-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    weights=torch.tensor([2.,1.],device='cuda')
    records=[];best=-1.;best_epoch=None
    for epoch in range(1,cfg['epochs']+1):
        model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter();total_ce=0.;total_weighted=0.
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train))
        for step,start in enumerate(range(0,len(train),cfg['batch_size']),1):
            ids=order[start:start+cfg['batch_size']]
            inputs,y=encoded_batch(enc,labels,ids,tok)
            optimizer.zero_grad(set_to_none=True)
            logits=model(**inputs).logits
            ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
            loss=(ce*weights[y]).sum()/weights[y].sum()
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm)
            optimizer.step();scheduler.step()
            total_ce+=ce.sum().item();total_weighted+=loss.item()*len(ids)
            if step%100==0:print(json.dumps({'epoch':epoch,'update':step,'total':steps,'seconds':time.perf_counter()-started}),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        score,devloss=score_texts(model,tok,dev.text.tolist(),return_loss=True,labels=dev.label.tolist())
        met=metrics(dev.label.to_numpy(),score)
        rec={'epoch':epoch,'learning_rate':cfg['learning_rate'],'benign_weight':2.,'attack_weight':1.,
            'train_unweighted_loss':total_ce/len(train),'train_weighted_loss':total_weighted/len(train),
            'dev_loss':devloss,**{f'dev_{k}':v for k,v in met.items()},'train_seconds':seconds,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
        records.append(rec);write_csv(OUT/'tuning.csv',records)
        if met['macro_f1']>best:
            best=met['macro_f1'];best_epoch=epoch
            model.save_pretrained(CP,safe_serialization=True);tok.save_pretrained(CP)
            write_csv(OUT/'weighted_validation_scores.csv',pd.DataFrame({'sample_id':dev.sample_id,'gold':dev.label,'score_attack':score}))
        print('epoch_complete',json.dumps(rec),flush=True)
    del optimizer,scheduler,model;gc.collect();torch.cuda.empty_cache()
    # This filter reads only the already frozen baseline validation scores, not held-out outcomes.
    baseline=pd.read_parquet(ORIGINAL/'handoff/roberta_scores.parquet',filters=[('dataset','==','d1'),('split','==','validation')])
    weighted=pd.read_csv(OUT/'weighted_validation_scores.csv')
    assert baseline.sample_id.tolist()==weighted.sample_id.tolist()==dev.sample_id.tolist()
    original_f1=metrics(baseline.gold,baseline.score_attack)['macro_f1']
    winner='benign_weight2' if best>original_f1 else 'original'
    rows=[]
    for name,frame in [('original',baseline),('benign_weight2',weighted)]:
        for r in policies(frame.gold.to_numpy(dtype=int),frame.score_attack.to_numpy(),cfg):rows.append({'model':name,**r})
    write_csv(OUT/'thresholds_source_only.csv',rows)
    selection={'time_utc':now(),'config_sha256':sha256(CFG),'weighted_best_epoch':best_epoch,
        'weighted_dev_macro_f1':best,'original_dev_macro_f1':original_f1,'source_selected_model':winner,
        'weighted_checkpoint':CP.relative_to(ROOT).as_posix(),'weighted_checkpoint_sha256':sha256(CP/'model.safetensors'),
        'thresholds_sha256':sha256(OUT/'thresholds_source_only.csv'),
        'selection_data':'D1 validation only','external_results_read_for_this_selection':False,
        'scientific_status':'Exploratory because method was motivated by previously observed test behavior'}
    write_json(OUT/'selection_frozen.json',selection)
    print('FROZEN',json.dumps(selection),flush=True)


if __name__=='__main__':main()
