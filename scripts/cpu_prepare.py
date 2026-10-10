"""CPU allocation: install, audit inputs, verify the pipeline, prepare pilot data."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

root=Path("experiments/transport_reasoning"); root.mkdir(parents=True,exist_ok=True)
commands=json.loads((root/"cpu_commands.json").read_text()) if (root/"cpu_commands.json").exists() else []
def run(args):
    started=time.time(); p=subprocess.run(args)
    commands.append({"command":args,"exit_code":p.returncode,"elapsed_seconds":time.time()-started})
    (root/"cpu_commands.json").write_text(json.dumps(commands,indent=2))
    if p.returncode: raise SystemExit(p.returncode)

if "--installed" not in sys.argv:
    run([sys.executable,"-m","pip","install","-e","."])
    os.execv(sys.executable,[sys.executable,__file__,"--installed"])
# Exact installed package versions, no local editable paths or inherited project imports.
from importlib.metadata import distributions
locked=sorted({f'{d.metadata["Name"]}=={d.version}' for d in distributions() if d.metadata["Name"] and d.metadata["Name"].lower()!="transport-reasoning"})
Path("requirements.lock").write_text("\n".join(locked)+"\n")
run([sys.executable,"-m","transport_reasoning","doctor"])
run([sys.executable,"-m","pytest","--junitxml=experiments/transport_reasoning/tests.xml"])
run([sys.executable,"-m","transport_reasoning","audit-mlp"])
from transport_reasoning.config import load,write_json
from transport_reasoning.data import file_hash
from transport_reasoning.training import train,evaluate_checkpoint
from transport_reasoning.reporting import summarize
for task in ("maze","sudoku"):
    c=load(f"configs/cpu_smoke_{task}.json")
    if not (c.run_dir()/"result.json").exists():
        train(c,stop_after=3)
    train(c,resume=c.run_dir()/"checkpoint_last.pt")
    evaluate_checkpoint(c,c.run_dir()/"checkpoint_best.pt")
summarize()
from pypdf import PdfReader
from openpyxl import load_workbook
inputs=Path("/inputs")
book=load_workbook(inputs/"binomial_attention_data.xlsx",read_only=True,data_only=True)
inventory={"pdf_pages":len(PdfReader(inputs/"binomial_attention_iclr2027_draft.pdf").pages),
           "input_files":{p.name:file_hash(p) for p in inputs.iterdir() if p.is_file()},
           "workbook_sheets":{s.title:{"rows":s.max_row,"columns":s.max_column} for s in book},
           "old_code_found":False,"pointer_status":"reimplementation; original source/checkpoints not supplied",
           "git_commit":None,"git_author_configured":False}
write_json(root/"input_inventory.json",inventory)
for name in ("Final evaluations","Learning runs","Threshold aggregates","Filter weights","Protocol values"):
    if name not in book.sheetnames: continue
    ws=book[name]
    # Keep original rows external; the repository only receives numeric comparisons below.
    p=inputs/(name.lower().replace(" ","_")+".csv")
    with p.open("w",newline="") as f:
        csv.writer(f).writerows(ws.values)
book.close()
archive=inputs/"sudoku-hard.zip"
if not archive.exists():
    url="https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"
    urllib.request.urlretrieve(url,str(archive)+".tmp")
    Path(str(archive)+".tmp").replace(archive)
destination=inputs/"rrn"; destination.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if Path(name).name in ("train.csv","valid.csv","test.csv"):
            (destination/Path(name).name).write_bytes(z.read(name))
write_json(root/"rrn_download.json",{"sha256":file_hash(archive),"bytes":archive.stat().st_size,"source":"official RRN loader URL"})
for task in ("maze","sudoku"):
    run([sys.executable,"-m","transport_reasoning","prepare-data","--config",f"configs/pilot_{task}.json"])
run([sys.executable,"-m","transport_reasoning","check-artifacts"])
write_json(root/"cpu_ready.json",{"status":"passed","commands":commands})
(root/"STATUS.md").write_text("# Status\n\nCPU correctness and train/save/resume/evaluate checks passed. GPU profiling and task pilots are pending.\n")
