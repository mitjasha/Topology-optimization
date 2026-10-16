"""3D U-Net/FNO/GNN training and paired FEM/OC comparison.

3D supervised models share the same volumetric dataset and are never assessed
as image classifiers: engineering selection is based on actual FEM after OC.
"""
import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from topopt.learning import RidgeSurrogate
from topopt.solver import optimize
from topopt.solver3d import Model3D, write_vtk
from .data3d import case_from_record, features3d, load3d
from .models3d import build_3d_model


ARCHITECTURES = ("unet3d","gnn3d","fno3d")


def train(dataset, architecture, output, epochs=400,patience=90,lr=.001,
          batch_size=4,seed=17,device="cpu"):
    if architecture not in ARCHITECTURES or min(epochs,patience,batch_size)<=0 or lr<=0:
        raise ValueError("Invalid architecture or training parameters")
    if device not in ("cpu","cuda"):
        raise ValueError("Unknown device")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    a=load3d(dataset)
    indices={s:np.flatnonzero(a["split"]==s)
             for s in ("train","validation","test")}
    if any(not len(v) for v in indices.values()):
        raise ValueError("Each split must have at least one converged FEM case")
    x=torch.from_numpy(a["features"]).to(device)
    y=torch.from_numpy(a["targets"][:,None]).to(device)
    model=build_3d_model(architecture,x.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=20,factor=.5)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    best,stale,history=float("inf"),0,[]
    start=perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        rng=np.random.permutation(indices["train"])
        loss_sum=0.
        for ids in np.array_split(rng,max(1,int(np.ceil(len(rng)/batch_size)))):
            optimizer.zero_grad()
            estimate=model(x[ids]).sigmoid()
            loss=F.mse_loss(estimate,y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            score=0.
            for ids in np.array_split(indices["validation"],
                                      max(1,int(np.ceil(len(indices["validation"])/batch_size)))):
                score+=float(F.mse_loss(model(x[ids]).sigmoid(),y[ids],reduction="sum"))
            score/=len(indices["validation"])*np.prod(a["targets"].shape[1:])
        scheduler.step(score)
        history.append({"epoch":epoch,"train_loss":loss_sum/len(indices["train"]),
                        "val_mse":score,"lr":optimizer.param_groups[0]["lr"]})
        if score<best-1e-8:
            best,stale=score,0
            torch.save({"architecture":architecture,"channels":x.shape[1],
                        "seed":seed,"state_dict":model.state_dict(),
                        "dimension":3,"epoch":epoch},output/"best.pt")
        else:
            stale+=1
        if epoch%25==0 or epoch==1:
            print(f"{architecture} epoch={epoch} val_mse={score:.6g}",flush=True)
        if stale>=patience:
            break
    report={"architecture":architecture,"dimension":3,"best_validation_mse":best,
            "train_cases":len(indices["train"]),"validation_cases":len(indices["validation"]),
            "test_cases_reserved":len(indices["test"]),"trained_epochs":len(history),
            "train_seconds":perf_counter()-start,"history":history,
            "selection":"validation density MSE, no test-based checkpoint selection"}
    (output/"training.json").write_text(json.dumps(report,indent=2))
    return report


def load_checkpoint(path,device="cpu"):
    state=torch.load(path,map_location=device,weights_only=True)
    if int(state["dimension"])!=3:
        raise ValueError("This is not a 3D model checkpoint")
    model=build_3d_model(state["architecture"],state["channels"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model,state["architecture"]


def evaluate(dataset,checkpoints,output,max_iter=400,tolerance=.02,seed=17):
    data=load3d(dataset)
    indices={s:np.flatnonzero(data["split"]==s)
             for s in ("train","validation","test")}
    flat_x=data["features"].reshape(len(data["features"]),-1)
    flat_y=data["targets"].reshape(len(data["targets"]),-1)
    validation=[]
    train_time=perf_counter()
    for alpha in (.0001,.001,.01,.1,1,10):
        ridge=RidgeSurrogate(alpha).fit(flat_x[indices["train"]],flat_y[indices["train"]])
        ratios=[]
        for i,pred in zip(indices["validation"],ridge.predict(flat_x[indices["validation"]])):
            model=Model3D(case_from_record(data["cases"][i]))
            guess=model.project_volume(pred)
            ratios.append(model.analyze(guess)["compliance"]/
                          float(data["reference_compliance"][i]))
        validation.append((np.mean(ratios),alpha))
    chosen_alpha=min(validation)[1]
    ridge=RidgeSurrogate(chosen_alpha).fit(flat_x[indices["train"]],flat_y[indices["train"]])
    ridge_train_time=perf_counter()-train_time
    networks={}
    training_times={"ridge":ridge_train_time}
    for name, path in checkpoints.items():
        network, architecture=load_checkpoint(path)
        if architecture!=name:
            raise ValueError(f"Mismatched checkpoint: {name} is {architecture}")
        networks[name]=network
        report=Path(path).parent/"training.json"
        training_times[name]=(float(json.loads(report.read_text())["train_seconds"])
                              if report.is_file() else None)
    methods=("oc","ridge",*networks)
    rows=[]
    rng=np.random.default_rng(seed)
    for i in indices["test"]:
        c=case_from_record(data["cases"][i])
        reference=float(data["reference_compliance"][i])
        for method in rng.permutation(methods):
            start=perf_counter()
            model=Model3D(c)
            if method=="oc":
                final=optimize(model,max_iter=max_iter,min_iter=15,tolerance=tolerance)
                attempt=final
                direct_ratio=None
                solves=final["fe_solves"]
                fallback=False
            else:
                early=optimize(model,max_iter=5,min_iter=15,tolerance=tolerance)
                ceiling=model.analyze(early["x"])["compliance"]
                features=features3d(model,early["x"])
                if method=="ridge":
                    guess=model.project_volume(ridge.predict(features.reshape(1,-1))[0])
                else:
                    with torch.no_grad():
                        logits=networks[method](torch.from_numpy(features[None]))
                        prediction=logits[0,0].sigmoid().numpy().ravel()
                    guess=model.project_volume(prediction)
                direct=model.analyze(guess)
                direct_ratio=direct["compliance"]/reference
                attempt=optimize(model,initial=guess,max_iter=max_iter,
                                 min_iter=15,tolerance=tolerance)
                solves=early["fe_solves"]+2+attempt["fe_solves"]
                accepted=(attempt["converged"] and
                          np.isfinite(attempt["compliance"]) and
                          attempt["compliance"]<=ceiling*(1+1e-8) and
                          abs(attempt["volume"]-c.volume)<1e-6 and
                          attempt["equilibrium_error"]<1e-7)
                fallback=not accepted
                final=attempt
                if fallback:
                    final=optimize(model,max_iter=max_iter,min_iter=15,tolerance=tolerance)
                    solves+=final["fe_solves"]
            quality=bool(final["converged"] and
                         final["compliance"]<=1.05*reference and
                         abs(final["volume"]-c.volume)<1e-6 and
                         final["equilibrium_error"]<1e-7)
            rows.append({"case_id":int(i),"method":str(method),"fe_solves":solves,
                         "seconds":perf_counter()-start,"fallback":fallback,
                         "quality_gate":quality,"final_compliance_ratio":final["compliance"]/reference,
                         "direct_compliance_ratio":direct_ratio})
            if method=="oc" and len([r for r in rows if r["method"]=="oc"])<=2:
                write_vtk(Path(output).parent/f"comparison_case_{int(i)}.vtk",c,final["rho"])
    if not rows:
        raise ValueError("No test cases")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline={row["case_id"]:row for row in rows if row["method"]=="oc"}
    manifest=Path(dataset).with_suffix(".json")
    offline=json.loads(manifest.read_text()).get("generation_seconds") if manifest.is_file() else None
    summary={"dimension":3,"test_cases":len(indices["test"]),"ridge_alpha":chosen_alpha,
             "model_summary":{}}
    for method in methods:
        selected=[r for r in rows if r["method"]==method]
        paired=np.mean([baseline[r["case_id"]]["seconds"]-r["seconds"] for r in selected])
        training=training_times.get(method,0.)
        total=(offline+training if training is not None and offline is not None else None)
        summary["model_summary"][method]={
            "mean_fe_solves":float(np.mean([r["fe_solves"] for r in selected])),
            "mean_seconds":float(np.mean([r["seconds"] for r in selected])),
            "mean_paired_seconds_saved":float(paired),
            "fallbacks":sum(r["fallback"] for r in selected),
            "quality_passes":sum(r["quality_gate"] for r in selected),
            "training_seconds":training,
            "offline_dataset_seconds":offline if method!="oc" else 0,
            "break_even_cases":total/paired if method!="oc" and total is not None and paired>0 else None}
    summary["caveat"]=("Unseen aspect ratios and load positions only, fixed-left cantilever; "
                       "3D models are trained from scratch, and CPU timing is platform dependent")
    output.with_suffix(".json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("train")
    a.add_argument("--dataset",default="artifacts/mlnext/dataset3d.npz")
    a.add_argument("--model",choices=ARCHITECTURES,default="unet3d")
    a.add_argument("--output",default="artifacts/mlnext/unet3d")
    a.add_argument("--epochs",type=int,default=400)
    a.add_argument("--patience",type=int,default=90)
    a.add_argument("--batch-size",type=int,default=4)
    a.add_argument("--lr",type=float,default=.001)
    a.add_argument("--seed",type=int,default=17)
    a.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    b=sub.add_parser("evaluate")
    b.add_argument("--dataset",default="artifacts/mlnext/dataset3d.npz")
    b.add_argument("--output",default="artifacts/mlnext/comparison3d.csv")
    b.add_argument("--checkpoint",action="append",default=[],
                   help="name=path/to/best.pt")
    args=p.parse_args()
    if args.mode=="train":
        print(json.dumps(train(args.dataset,args.model,args.output,epochs=args.epochs,
                               patience=args.patience,batch_size=args.batch_size,
                               lr=args.lr,seed=args.seed,device=args.device),indent=2))
    else:
        print(json.dumps(evaluate(args.dataset,
                                  dict(item.split("=",1) for item in args.checkpoint),
                                  args.output),indent=2))
