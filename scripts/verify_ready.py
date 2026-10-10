"""Final CPU gate before profiling: rerun tests, inspect source and data provenance."""
import json
import subprocess
import sys
from pathlib import Path
from transport_reasoning.config import ROOT,load,write_json
from transport_reasoning.data import TaskData,sudoku_solutions,file_hash
from transport_reasoning.evaluation import sudoku_valid
from transport_reasoning.__main__ import check_artifacts

def run(args):
    result=subprocess.run(args)
    if result.returncode: raise SystemExit(result.returncode)

run([sys.executable,"-m","pytest","--junitxml=experiments/transport_reasoning/tests_final_code.xml"])
run([sys.executable,"scripts/pointer_bridge.py"])
run([sys.executable,"scripts/source_inventory.py"])
for command in ("doctor","prepare-data","train","evaluate","run-queue","summarize","check-artifacts"):
    run([sys.executable,"-m","transport_reasoning",command,"--help"])
run([sys.executable,"-m","transport_reasoning","run-queue","--config","configs/queue_smoke.json","--dry-run"])
c=load("configs/pilot_sudoku.json")
data=TaskData(c,"validation")
sample=[]
for i in range(min(3,len(data))):
    q=data.data["puzzle"][i]; answer=data.data["y"][i]
    sample.append({"id":data.ids[i],"target_valid":sudoku_valid(answer,q),"solution_count_capped_at_two":len(sudoku_solutions(q))})
if not all(r["target_valid"] and r["solution_count_capped_at_two"]==1 for r in sample):
    raise RuntimeError("Sampled official Sudoku validation failed")
write_json(ROOT/"rrn_sampled_solver_audit.json",{"sample":sample,"all_rows_certified":False,"selection":"first three official validation rows"})
for task in ("maze","sudoku"):
    c=load(f"configs/pilot_{task}.json")
    for split in ("train","validation","test"):
        data=TaskData(c,split)
        if file_hash(c.data_dir()/f"{split}.npz")!=data.manifest["files"][f"{split}.npz"]:
            raise RuntimeError("Dataset integrity check failed")
write_json(ROOT/"ready_for_gpu.json",{"status":"passed","artifact_check":check_artifacts(),"pointer_bridge":True})
print("READY_FOR_GPU",flush=True)
