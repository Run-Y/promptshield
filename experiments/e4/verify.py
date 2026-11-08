"""Verify accumulated gradients, handoff integrity and saved-model reproducibility."""
from __future__ import annotations
import argparse
import json
from types import SimpleNamespace

from .common import (HANDOFF_COLUMNS, RESULTS, ROOT, config, config_hash, load_frame,
                     metrics, now, read_json, score_texts, selected, selected_model,
                     setup, sha256, write_json)
import numpy as np
import pandas as pd
import torch


def preflight():
    """Compare the actual update routine against full-batch SGD, including a short tail."""
    from .train import update
    setup()
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(3,2)
        def forward(self,input_ids,attention_mask):
            return SimpleNamespace(logits=self.linear(input_ids.float()))
    class Pad:
        def pad(self,rows,**kwargs):
            class Batch(dict):
                def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
            return Batch({k:torch.tensor([r[k] for r in rows]) for k in rows[0]})
    checks=[]
    for n in [16,10,1]:
        torch.manual_seed(6405)
        a,b=Tiny().cuda(),Tiny().cuda();b.load_state_dict(a.state_dict())
        x=np.arange(n*3).reshape(n,3)%7
        enc={"input_ids":x.tolist(),"attention_mask":np.ones_like(x).tolist()}
        y=np.arange(n)%2
        oa,ob=torch.optim.SGD(a.parameters(),lr=.01),torch.optim.SGD(b.parameters(),lr=.01)
        logits=a(torch.tensor(x,device="cuda"),None).logits
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor(y,device="cuda",dtype=torch.long))
        loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.0);oa.step()
        actual=update(b,ob,torch.amp.GradScaler("cuda",enabled=False),enc,y,np.arange(n),Pad(),4,"fp32")
        maxdiff=max((pa-pb).abs().max().item() for pa,pb in zip(a.parameters(),b.parameters()))
        assert maxdiff<2e-7 and abs(actual-loss.item())<1e-6,(n,maxdiff)
        checks.append({"effective_batch":n,"max_parameter_error":maxdiff})
    write_json(RESULTS/"preflight_verification.json",{"status":"passed","accumulation_matches_full_batch_sgd":checks,"time_utc":now()})
    print(json.dumps(checks),flush=True)


def verify():
    setup()
    s,cp=selected()
    checks={}
    def check(name,condition,detail=None):
        checks[name]={"passed":bool(condition),"detail":detail}
        if not condition:
            write_json(RESULTS/"verification.json",{"status":"failed","checks":checks,"time_utc":now()})
            raise AssertionError(name)
    raw=pd.read_parquet(RESULTS/"handoff/roberta_scores.parquet")
    check("schema",list(raw.columns)==HANDOFF_COLUMNS,list(raw.columns))
    check("unique_rows",not raw.duplicated(["model_id","dataset","split","sample_id","transform_id"]).any())
    check("finite_unit_scores",np.isfinite(raw.score_attack).all() and raw.score_attack.between(0,1).all())
    check("clean_parent_contract",raw.transform_id.eq("clean").all() and raw.parent_id.isna().all())
    check("checkpoint_and_seed",raw.checkpoint_rev.eq(s["checkpoint_rev"]).all() and raw.seed.eq(s["seed"]).all()
          and raw.model_id.eq(s["model_id"]).all())
    expected=config()["evaluation"]["expected_rows"]
    check("total_rows",len(raw)==sum(expected.values()),len(raw))
    tune=pd.read_csv(RESULTS/"tuning.csv")
    check("six_candidates",set(zip(tune.learning_rate,tune.epoch))=={(lr,ep) for lr in [2e-5,5e-5] for ep in [1,2,3]})
    check("same_epoch_permutations",tune.groupby("epoch").permutation_sha256.nunique().eq(1).all())
    check("all_training_rows_used",tune.train_examples_seen.eq(11046).all() and tune.optimizer_updates.eq(691).all())
    winner=tune.sort_values(["dev_macro_f1","epoch","learning_rate"],ascending=[False,True,True]).iloc[0]
    check("selection_rule",winner.learning_rate==s["learning_rate"] and int(winner.epoch)==s["epoch"])
    events=[json.loads(line) for line in (RESULTS/"events.jsonl").read_text().splitlines()]
    names=[e["event"] for e in events]
    check("freeze_precedes_test",names.index("checkpoint_frozen")<names.index("test_scoring_started"))
    model,tok,_=selected_model()
    metric_table=pd.read_csv(RESULTS/"metrics_at_0.5.csv").set_index("dataset_split")
    for name,count in expected.items():
        frame=load_frame(name)
        dataset,split=name.split("_",1)
        part=raw[(raw.dataset==dataset)&(raw.split==split)]
        check(f"{name}_identity",len(part)==count and part.sample_id.tolist()==frame.sample_id.tolist()
              and np.array_equal(part.gold.to_numpy(),frame.label.to_numpy()))
        scores=score_texts(model,tok,frame.text.tolist())
        error=float(np.max(np.abs(scores-part.score_attack.to_numpy())))
        check(f"{name}_saved_model_rescore",error<2e-6,{"max_abs_error":error})
        m=metrics(part.gold,part.score_attack)
        check(f"{name}_metric_recompute",all(abs(float(metric_table.loc[name,k])-v)<1e-10 for k,v in m.items()))
        if name=="d1_validation":
            check("selected_validation_f1_matches",abs(m["macro_f1"]-s["dev_macro_f1"])<1e-10)
        # A second batch shape detects score/order problems hidden by matching batch code.
        independent=score_texts(model,tok,frame.text.iloc[:10].tolist(),batch_size=1)
        check(f"{name}_batch1_agreement",np.allclose(independent,scores[:10],atol=2e-5,rtol=0),
              float(np.max(np.abs(independent-scores[:10]))))
    lat=pd.read_csv(RESULTS/"cost_comparison.csv")
    trials=pd.read_csv(RESULTS/"latency_trials.csv")
    for row in lat.itertuples():
        values=trials[(trials.model_id==row.model_id)&(trials.device==row.device)].latency_ms.to_numpy()
        check(f"latency_{row.model_id}_{row.device}",len(values)==100 and
              abs(np.median(values)-row.median_ms)<1e-10 and abs(np.percentile(values,95)-row.p95_ms)<1e-10)
    spans=pd.read_csv(RESULTS/"occlusion_spans.csv")
    cases=pd.read_csv(RESULTS/"occlusion_cases.csv")
    check("fixed_six_cases",cases.sample_id.tolist()==config()["occlusion"]["case_ids"])
    check("occlusion_deltas",np.allclose(spans.delta_score,spans.original_score-spans.masked_score,atol=1e-12))
    check("occlusion_same_checkpoint",spans.checkpoint_rev.eq(s["checkpoint_rev"]).all())
    with torch.inference_mode():
        for case in cases.itertuples():
            encoded=tok(case.text,truncation=True,max_length=256,return_tensors="pt").to("cuda")
            for row in spans[spans.sample_id==case.sample_id].itertuples():
                mask={k:v.clone() for k,v in encoded.items()}
                positions=[int(x) for x in str(row.token_positions).split()]
                mask["input_ids"][0,positions]=tok.mask_token_id
                score=float(torch.softmax(model(**mask).logits.float(),dim=-1)[0,1].item())
                check(f"occlusion_{case.sample_id}_{row.span_index}",abs(score-row.masked_score)<2e-6)
    report={"status":"passed","checks":checks,"check_count":len(checks),"time_utc":now(),
            "checkpoint_rev":s["checkpoint_rev"],"config_sha256":config_hash()}
    write_json(RESULTS/"verification.json",report)
    print(f"VERIFIED {len(checks)} checks",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    preflight() if args.preflight else verify()
