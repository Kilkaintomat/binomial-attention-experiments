#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C LANG=C
transport_root="$(cd "${TRANSPORT_ROOT:?}" && pwd -P)"
transport_inputs="$(cd "${TRANSPORT_INPUTS:?}" && pwd -P)"
apptainer exec --cleanenv --bind "$transport_root:/workspace" --bind "$transport_inputs:/inputs:ro" --pwd /workspace "${TRANSPORT_IMAGE:?}" bash -c '
set -euo pipefail
export XDG_CACHE_HOME=/workspace/.cache PIP_DISABLE_PIP_VERSION_CHECK=1
python -m venv --system-site-packages /workspace/.venv
/workspace/.venv/bin/python -m pip install openpyxl pypdf pytest setuptools wheel
/workspace/.venv/bin/python - <<"PY"
import torch,sys,importlib.metadata as m,json
from pathlib import Path
names=["torch","numpy","matplotlib","openpyxl","pypdf","pytest","setuptools","wheel"]
d={"python":sys.version.split()[0],"packages":{n:m.version(n) for n in names},"cuda_build":torch.version.cuda,"container_inherited_packages":True}
Path("experiments/transport_reasoning/bootstrap_environment.json").write_text(json.dumps(d,indent=2))
print(json.dumps(d))
PY
'
