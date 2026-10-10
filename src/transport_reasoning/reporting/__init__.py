"""Summaries are generated exclusively from recorded run artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
from transport_reasoning.config import ROOT,write_json


def threshold(curve,value):
    for i in range(1,len(curve)):
        if curve[i-1]["solved_rate"]>=value and curve[i]["solved_rate"]>=value:
            previous=curve[i-2] if i>=2 else {"step":0,"training_seconds":0,"examples":0}
            return {"status":"reached","first_passing_step":curve[i-1]["step"],
                    "confirmed_step":curve[i]["step"],"previous_failing_step":previous["step"],
                    "training_seconds_lower":previous["training_seconds"],
                    "training_seconds_upper":curve[i-1]["training_seconds"]}
    return {"status":"right_censored","last_step":curve[-1]["step"] if curve else 0}


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(""); return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def summarize(runs_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(runs_dir) if runs_dir else ROOT/"runs"
    rows=[]; curves=[]; final={}; timing=[]
    for path in sorted(root.rglob("result.json")):
        result=json.loads(path.read_text()); c=result["config"]
        mp=path.parent/"metrics.jsonl"
        curve=[json.loads(line) for line in mp.read_text().splitlines()] if mp.exists() else []
        curves.append((c,curve))
        ev=path.parent/"evaluation_test.json"
        evaluated=json.loads(ev.read_text()) if ev.exists() else result.get("last_validation") or {}
        filt=result.get("filters",[{}])[0]
        row={"task":c["task"],"stage":c["stage"],"split":"test" if ev.exists() else "validation",
             "support":c["support"],"method":c["method"],"seed":c["seed"],"config_hash":result["config_hash"],
             "commit":result["commit"],"source_sha256":result["source_sha256"],"data_hash":result["data_hash"],
             "L":c["layers"],"H":c["hops"] if c["method"] not in ("onehop","recurrent") else 1,
             "T":c["recurrent_steps"] if c["method"]=="recurrent" else 1,"width":c["width"],"heads":c["heads"],
             "parameter_count":result["parameter_count"],"mu":filt.get("mu"),"nu":filt.get("nu"),
             "LR":c["lr"],"schedule":"linear_warmup_cosine","updates":result["step"],"examples":result["examples"],
             "gpu_seconds":result["gpu_seconds"],"training_seconds":result["training_seconds"],"elapsed_seconds":result["elapsed_seconds"],
             "train_memory":result["train_peak_memory_bytes"],"inference_memory":evaluated.get("inference_peak_memory_bytes"),
             "inference_seconds_per_example":evaluated.get("inference_seconds_per_example"),"solved_rate":evaluated.get("solved_rate"),
             "best_checkpoint_rule":result["best_checkpoint_rule"],"run_status":result["status"],"failure_reason":result["failure_reason"],
             "threshold_90":json.dumps(threshold(curve,.9)),"threshold_50":json.dumps(threshold(curve,.5)) if c["task"]=="sudoku" else None,
             "mlp_type":"GELU","inner_mlp_width":(c["inner_width"] or c["width"]) if c["inner_beta"] else 0,
             "inner_beta":c["inner_beta"],"gate_refresh_period":1,"tied":c["method"]=="recurrent","LN_mode":"pre_LN",
             "total_FFN_evaluations":c["recurrent_steps"] if c["method"]=="recurrent" else c["layers"]*(1+(c["hops"] if c["inner_beta"] else 0)),
             "cache_cost":"included_in_forward","gate_drift":"not_measured"}
        rows.append(row)
        timing.append({k:row[k] for k in ("task","stage","method","seed","inner_beta","training_seconds","elapsed_seconds","gpu_seconds","inference_seconds_per_example")})
        if c["stage"]=="final" and ev.exists() and result["status"]=="completed":
            key=(c["task"],c["support"],c["inner_beta"],c["method"])
            final.setdefault(key,{})[c["seed"]]=evaluated["solved_rate"]
    summary=ROOT/"summaries"; figures=ROOT/"figures"; figures.mkdir(parents=True,exist_ok=True)
    table(summary/"results.csv",rows); table(summary/"timing.csv",timing)
    pairs=[]; rng=np.random.default_rng(7531)
    for key,bins in final.items():
        if key[-1]!="binomial": continue
        for other in ("exponential_truncated","finite_geometric"):
            controls=final.get((*key[:-1],other),{}); seeds=sorted(set(bins)&set(controls))
            if not seeds: continue
            values=np.array([bins[s]-controls[s] for s in seeds])
            interval=np.quantile(rng.choice(values,(10000,len(values)),replace=True).mean(1),[.025,.975]) if len(values)>1 else [None,None]
            pairs.append({"task":key[0],"support":key[1],"inner_beta":key[2],"comparison":"binomial-"+other,
                          "seeds":json.dumps(seeds),"paired_values":json.dumps(values.tolist()),"mean_difference":float(values.mean()),
                          "sample_sd":float(values.std(ddof=1)) if len(values)>1 else None,"bootstrap_seed_CI_low":interval[0],
                          "bootstrap_seed_CI_high":interval[1],"full_five_seed_coverage":len(seeds)==5})
    table(summary/"paired_differences.csv",pairs)
    aggregates=[]
    for (task,support,beta,method),seed_scores in final.items():
        values=list(seed_scores.values())
        aggregates.append({"task":task,"support":support,"inner_beta":beta,"method":method,
                           "seeds":json.dumps(sorted(seed_scores)),"per_seed":json.dumps(seed_scores),
                           "mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                           "full_five_seed_coverage":len(values)==5})
    table(summary/"final_aggregates.csv",aggregates)
    from transport_reasoning.filters import coefficients,moments
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for method in ("binomial","exponential_truncated","finite_geometric"):
        m=moments(coefficients(method,8)); s=np.arange(1,9)
        label=f'{method}: mean={m["mu"]:.3f}, nu={m["nu"]:.3f}'
        axes[0].plot(s,m["coefficients"],"o-",label=label)
        axes[1].plot(s,m["tail"],"o-")
    axes[0].set(xlabel="Hops",ylabel="Weight",title="H=8 matched mean")
    axes[1].set(xlabel="Distance",ylabel="Positive-hop tail")
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(figures/"filter_weights_H8.png",dpi=160); plt.close(fig)
    measured=[r for r in rows if r["split"]=="test" and r["solved_rate"] is not None]
    if measured:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for r in measured:
            label=f'{r["task"]} {r["method"]} s{r["seed"]}'
            axes[0].scatter(r["inference_seconds_per_example"],r["solved_rate"],label=label,s=12)
            if r["inference_memory"] is not None: axes[1].scatter(r["inference_memory"]/1e6,r["solved_rate"],s=12)
        axes[0].set(xlabel="Inference seconds / example",ylabel="Solved fraction")
        axes[1].set(xlabel="Inference peak memory MB",ylabel="Solved fraction")
        fig.tight_layout(); fig.savefig(figures/"accuracy_compute.png",dpi=160); plt.close(fig)
    for task in ("maze","sudoku"):
        for stage in sorted({c["stage"] for c,_ in curves if c["task"]==task}):
            selected=[(c,curve) for c,curve in curves if c["task"]==task and c["stage"]==stage and curve]
            if not selected: continue
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for c,curve in selected:
                label=f'{c["method"]} s{c["seed"]} beta={c["inner_beta"]}'
                for ax,x in zip(axes,["examples","training_seconds"]):
                    ax.plot([p[x] for p in curve],[p["solved_rate"] for p in curve],label=label,alpha=.7)
                    ax.set_xlabel(x); ax.set_ylabel("Validation solved fraction"); ax.set_ylim(-.02,1.02)
            axes[1].legend(fontsize=5,loc="upper left",bbox_to_anchor=(1,1))
            fig.tight_layout(); fig.savefig(figures/f"{task}_{stage}_learning.png",dpi=150); plt.close(fig)
    complete=sum(r["run_status"]=="completed" for r in rows)
    report=["# Отчёт transport-reasoning","",f"Зарегистрировано {len(rows)} запусков; завершено {complete}.",
            "Метрики ниже рассчитаны по фактическим журналам. Smoke и pilot не являются финальным доказательством.",
            "Время обучения включает загрузку батчей, forward/backward и optimizer; evaluation и checkpointing учтены отдельно.",
            "GPU-бюджет считается по полному времени выделения, включая накладные расходы, в budget.json.","",
            "| Задача | Этап | Метод | Seed | inner beta | Split | Solved | Статус |",
            "|---|---|---|---:|---:|---|---:|---|"]
    for r in rows:
        score=f'{r["solved_rate"]:.4f}' if r["solved_rate"] is not None else "—"
        report.append(f'| {r["task"]} | {r["stage"]} | {r["method"]} | {r["seed"]} | {r["inner_beta"]} | {r["split"]} | {score} | {r["run_status"]} |')
    report += ["","Сравнения формы используют общий H, matched mean, gain=1 и общий evaluator. Learned polynomial может менять mean.",
               "Recurrent baseline обновляет attention, LayerNorm и FFN на каждом шаге; число параметров и цена отличаются.",
               "Неполное покрытие, низкие solved rates и широкие интервалы не подтверждают преимущество или эквивалентность методов.",
               "Заявления об ускорении требуют сравнения при сопоставимом solved rate. CPU-проверки алгебры не подтверждают скорость GPU.",
               "Размеры, объёмы данных и число updates определяются сохранённым protocol; OOD и дополнительные абляции могут быть отложены.",
               "Файл remaining_queue.json содержит отложенные этапы. Точные команды воспроизведения находятся в commands.txt и README.md."]
    (ROOT/"REPORT_RU.md").write_text("\n".join(report)+"\n")
    return {"runs":len(rows),"completed":complete,"final_paired_comparisons":len(pairs)}
