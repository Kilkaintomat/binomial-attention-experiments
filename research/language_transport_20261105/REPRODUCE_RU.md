# Воспроизведение языковой кампании

Точный протокол — [PROTOCOL_RU.md](PROTOCOL_RU.md). Исходники в `code/`; модель, данные и private scheduler account в публикацию не включаются. Версии: Python 3.11.11, PyTorch 2.6.0+cu124, Transformers 4.51.3, PEFT 0.15.2, NumPy 2.2.2. Plotting-зависимости при необходимости устанавливаются в отдельную `.deps`, общий ML environment не меняется.

## В существующем окружении проекта

Каталог кампании должен находиться по относительному пути `research/language_transport_20261105`. Корень содержит `.venv`, `models/Qwen3-1.7B`, `artifacts/models.json` и локальный `slurm.env`. Последний задаёт `QBM_ROOT`, `QBM_IMAGE`, `QBM_ACCOUNT`, `QBM_GPU_PARTITION`, `QBM_CPU_PARTITION`; значения зависят от кластера и не публикуются. Контейнер должен предоставлять CUDA и совместимый Python.

Все проверки и подготовка выполняются в CPU allocation:

```bash
source slurm.env
prepare_id=$(sbatch --parsable --account="$QBM_ACCOUNT" \
  --partition="$QBM_CPU_PARTITION" --cpus-per-task=4 --mem=8G --time=30 \
  --output=research/language_transport_20261105/logs/prepare-%j.log \
  research/language_transport_20261105/code/runtime.sh \
  research/language_transport_20261105/code/prepare.py)
sbatch --account="$QBM_ACCOUNT" --partition="$QBM_CPU_PARTITION" \
  --dependency="afterok:$prepare_id" --cpus-per-task=1 --mem=512M --time=5 \
  --output=research/language_transport_20261105/logs/watch-%j.log \
  research/language_transport_20261105/code/watch.sbatch
```

Это команды **первой активации**. Уже активированную кампанию не нужно запускать повторно: её CPU dispatcher продолжает себя сам. `artifacts/status.json` и `watcher.json` показывают этап и актуальный job ID. До завершения старой MLP-серии GPU новой кампании не запрашиваются. После неё сначала идут GPU preflight, шесть pilot, 48 scratch и затем 18 LoRA + две frozen оценки. Проверки качества короткого baseline могут остановить продвижение только новой кампании с сохранением причины.

Фиксируются `manifest.json`, его SHA-256, source hash, protocol hash и контрольные суммы данных. Нельзя редактировать код активной кампании: исправления требуют отдельной документированной версии и сохранения старых результатов.

## Отдельное воспроизведение без старой очереди

На другом кластере можно использовать библиотечные функции напрямую в выделенном Slurm job, не запуская наш dispatcher с зависимостью от предыдущего MLP. Сначала создать совместимое окружение и скачать через CPU allocation Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, в `models/Qwen3-1.7B`. Модель нужна только для LM-ветки; scratch и его генераторы не используют pretrained weights.

В этих командах рабочая папка — корень репозитория; `PYTHONPATH` указывает на код кампании. Они выполняются **в scheduler allocation**, а не на login node. Данные генерируются заново из зафиксированных seed; опубликованный data manifest предназначен для проверки hashes, его не следует копировать вместо самих данных.

```bash
export PYTHONPATH="$PWD/research/language_transport_20261105/code"
python -m unittest -v test_correctness test_scheduler
python -c "from data import make_data; from common import SIDE; make_data(SIDE/'data')"
```

Пример точного отдельного scratch-прогона после выбора LR/steps по pilot (здесь показана ветка LR=0.001, 8000 updates):

```python
import time
from plan import spec
from engine import train, evaluate

run = spec('scratch', 'vt', 'binomial', 11, lr=0.001, steps=8000)
deadline = time.monotonic() + 6900  # внутри двухчасовой GPU allocation
if train(run, deadline):
    evaluate(run, deadline)
```

При исчерпании slice повторить те же вызовы с тем же spec и новым deadline: состояние optimizer/RNG/weights загружается из checkpoint, уже записанные предсказания пропускаются. LR/steps для воспроизведения всей кампании надо брать из её `selected_protocol.json`, когда pilot завершится, а не подбирать по test. `plan.py` содержит полную матрицу и правило выбора.

Для Qwen вместо `scratch` задаётся `qwen`, task `vt` или `phop_conditional`, method `baseline` либо `binomial`, layers 2 либо 4, LR=0.0001 и steps=2000. Для frozen-оценки поставить `frozen=True`, steps=0, method=baseline; environment metadata должно содержать зафиксированный model revision. Полная рабочая матрица создаётся `plan.pretrained()`.

## Выходные данные

Для каждого запуска: неизменяемый spec, `training_log.jsonl`, `validation.jsonl`, `checkpoint_last.pt`, `training.json`, `predictions.jsonl`, `result.json`. При жёстком прерывании сырые журналы могут содержать повторно выполненные обновления после последнего checkpoint; подтверждённый прогресс определяется checkpoint и training.json, сырые записи сохраняются.

После завершения либо для промежуточного среза выполнить в CPU allocation:

```bash
python research/language_transport_20261105/code/report.py
```

Отчёт берёт итоговую accuracy только из запусков с полностью завершённой оценкой. `metrics_by_seed.csv` и `attention_columns.csv` хранят результаты/дельты к ordinary baseline. Исходная LM (`phase=frozen`) и LoRA (`phase=main`), канонический p-hop и адаптации сохраняются раздельно. `n=0` не превращается в нулевую accuracy. Стандартное отклонение — между независимыми seed. Аналитический oracle-график всегда подписан отдельно от измерений обученных моделей.
