"""Record small reproducibility metadata without machine paths or raw text."""
import hashlib
import json
import os
from pathlib import Path
from transport_reasoning.config import ROOT,write_json
from transport_reasoning.data import file_hash

files={}
for directory in ("src","tests","configs","scripts"):
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts: files[str(p)]=file_hash(p)
for name in ("README.md","AGENTS.md","pyproject.toml","requirements.lock","Makefile",".gitignore"):
    p=Path(name)
    if p.exists(): files[name]=file_hash(p)
write_json(ROOT/"source_manifest.json",{"files":files,"commit":os.environ.get("TRANSPORT_COMMIT") or None})
inventory=json.loads((ROOT/"input_inventory.json").read_text())
lines=["# Inventory","","Standalone private repository created from scratch.",
       "External PDF and workbook are stored outside this repository and excluded from publication.",
       f"Paper: {inventory['pdf_pages']} pages. Workbook: {len(inventory['workbook_sheets'])} sheets.",
       "Input checksums are in input_inventory.json. Exact package versions are in requirements.lock.",
       "No original pointer source or checkpoint was provided. The bridge is labeled reimplementation.",
       "No Git author configuration was found; no author identity or commit is fabricated.",
       "CPU and GPU work uses scheduler allocations. Existing unrelated projects remain untouched.",
       "The private machine environment selects the existing container, account and partitions.",
       "No remote was configured or published. Task artifacts are generated from logged measurements."]
(ROOT/"INVENTORY.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"source_files":len(files),"input_checksums":inventory["input_files"]}),flush=True)
