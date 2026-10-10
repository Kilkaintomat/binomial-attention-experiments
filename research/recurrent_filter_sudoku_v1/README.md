# Recurrent filter Sudoku v1

Новая серия отделяет тип фильтра от внешней рекуррентности. Исходный `src/`, архивные JSON и checkpoint не изменяются. Нужны Python ≥3.10, PyTorch, NumPy, pytest и matplotlib. Ни данные, ни веса, ни сведения о вычислительной площадке не добавляются в git.

## Установка

В новом checkout, до выделения вычислительного задания:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
export PYTHONPATH="$PWD/src:$PWD"
```

`requirements.lock` в исходном архиве — снимок большого окружения, без hashes пакетов и образа; это не строгий переносимый lockfile. В исполняемой серии используется отдельное дерево кода и существующее проверенное окружение. CPU-проверка сохраняет фактические версии всех установленных пакетов в `checks/installed_versions.txt`, основные версии и CUDA — в `checks/environment.json`; каждое обучение пишет собственное `environment.json`. Для точного повторения используйте эти версии и тот же контейнер; перенос на другое оборудование требует нового замера времени.

## Данные и приватная конфигурация

Получите официальный RRN Sudoku через существующий загрузчик, передав один из архивных final Sudoku config командой `python -m transport_reasoning prepare-data --config <legacy_config.json>`. Подготовка выполняется в CPU-выделении. Не подменяйте данные синтетическим Sudoku для основной серии.

Задайте переменные окружения:

```bash
export RECURRENT_DATA_DIR=<directory_with_train_validation_test_npz_and_manifest>
export RECURRENT_LEGACY_ROOT=<read_only_legacy_artifact_root>
export RECURRENT_OUTPUT=experiments/recurrent_filter_sudoku_v1
export RECURRENT_CODE_COMMIT=$(git rev-parse HEAD)
```

Существующая площадка использует read-only bind старого проекта в `/legacy` и новый рабочий каталог в `/workspace`. Служебные значения задаются только в игнорируемом `.cache/site.sh`: `RUN_PROJECT`, `RUN_LEGACY_PROJECT`, `RUN_IMAGE`, `RUN_PYTHON_IN_CONTAINER`, `RUN_DATA_IN_CONTAINER`, `RUN_ACCOUNT`, `RUN_GPU_PARTITION`, `RUN_CPU_PARTITION`, `RUN_GPU_GRES`, `RUN_HOST_PYTHON`, `RECURRENT_CODE_COMMIT`. Все значения должны быть экспортированы. `RUN_HOST_PYTHON` должен указывать на Python ≥3.10 с доступным scheduler CLI. Не коммитьте этот файл.

## Подготовка, проверки и очередь

```bash
python -m research.recurrent_filter_sudoku_v1 prepare
```

Эта команда фиксирует code hash, протокол и 32 строки конфигураций: 12 обязательных пилотов, 15 обязательных финальных результатов и 5 необязательных B32. Manifest сохраняет явный статус каждой строки.

Следующие команды выполняются внутри настоящего CPU-выделения планировщика, с доступными переменными данных:

```bash
python -m research.recurrent_filter_sudoku_v1 checks
python -m research.recurrent_filter_sudoku_v1 overfit --label R16
python -m research.recurrent_filter_sudoku_v1 overfit --label B16
python -m research.recurrent_filter_sudoku_v1 overfit --label R32
python -m research.recurrent_filter_sudoku_v1 overfit --label B32
python -m research.recurrent_filter_sudoku_v1 report
```

Четыре overfit-команды можно распределить по независимым CPU-выделениям. Каждая использует первые 64 train Sudoku, отдельный seed 211, максимум 1000 обновлений и ограничение 3300 секунд. Критерий допуска — хотя бы 95% решённых задач этого тренировочного набора у всех четырёх вариантов. Это не оценка обобщения. Ошибка или незавершённый overfit блокирует массовое обучение.

После загрузки `.cache/site.sh` на узле с scheduler CLI:

```bash
source .cache/site.sh
export PYTHONPATH="$RUN_PROJECT/src:$RUN_PROJECT"
python -m research.recurrent_filter_sudoku_v1 submit
```

Создаётся собственное held GPU-задание профиля, отдельное accounting-задание и CPU gate, зависящий от accounting существующей кампании. Gate читает её состояние и переносит только свою зависимость на следующий её chunk. Он не меняет и не отменяет прежние задания. До завершения всех 20 прежних MLP-прогонов новая серия не получает GPU.

Дополнительный бюджет по умолчанию **не разрешён**. После явного поручения пользователя с абсолютными лимитами:

```bash
python -m research.recurrent_filter_sudoku_v1 allow --gpu-hours <approved_cap> --wall-hours <approved_wall_cap>
```

Wall-clock отсчитывается от первого GPU-выделения новой серии. Profiling занимает отдельное выделение до 20 минут. Accounting списывает весь фактически выделенный интервал, включая overhead. Следующие задания идут последовательно, максимум по 2 часа; доступный остаток резервируется до запуска. Ошибка, отсутствие accounting, предел времени или нехватка бюджета останавливают цепочку. Profile не даёт автоматического разрешения расширять лимит. B32 final включается только по численному бюджету, до новых test-метрик; другие расширения выключены.

Оценка обязательной стоимости: измеренные step time × updates, полная validation × частота, сохранения, test и reserve на диагностику, затем +20%. Решение и стоимость хранятся в `admission.json` и `profile/budget_estimate.json`. Если не хватает, `needs_budget.json` содержит точный недостающий остаток. Исходный 24 GPU-hour бюджет прежней кампании не перераспределяется.

## Ручной resume и оценка

В отдельном разрешённом GPU-выделении, при соблюдении того же лимита:

```bash
python -m research.recurrent_filter_sudoku_v1 train --config experiments/recurrent_filter_sudoku_v1/configs/pilot/B16_seed101.json --resume
python -m research.recurrent_filter_sudoku_v1 evaluate --config experiments/recurrent_filter_sudoku_v1/configs/final/B16_seed11.json
python -m research.recurrent_filter_sudoku_v1 report
```

Без `--resume` существующий checkpoint не перезаписывается. Resume требует совпадения config, кода и данных и восстанавливает optimizer, scheduler и все RNG. На границе задания обучение сохраняет текущий шаг без искусственной validation вне сетки. Test разрешён только полному final на 30000 обновлений; диспетчер выполняет его после окончания основной тренировки и заморозки решения B32.

`queue/state.json` показывает состояние цепочки и GPU-часы. Если очередь остановилась, сначала прочитайте причину. После исправления ошибки реализации нужна новая версия серии; нельзя заменить код под старым hash и продолжить молча. Расширение лимита требует отдельного поручения; ручной запуск train не заменяет accounting.

## Метрики и отчёт

Отчёт `REPORT_RECURRENT_SUDOKU_RU.md` и `report/` формируются из фактических артефактов. Пилоты и финалы разделены. Исторические R32 могут участвовать только в качестве; inference измеряется вновь, старое training time не служит доказательством ускорения. Таблицы содержат отсутствующие/незавершённые строки, sample SD и парный percentile bootstrap по seeds (20000 повторов, seed 90421). 18000 test-задач не заменяют пять обучающих seeds.

Порог 50%/90% подтверждается двумя последовательными validation; сохраняются последний failing и первый passing checkpoint. Pending и not_reached остаются цензурированными. Диагностика выполняется на первых 16 validation-задачах; падение дисперсии не доказывает причинность, а промежуточный readout обучался только на финальном выходе.

Графики показывают validation против updates и elapsed, а также test quality против измеренного inference cost. Временные кривые не представляются как отдельно обученные equal-wall-time модели.

Расширения семейства фильтров, gate/fixed mix и deep supervision подготовлены как отключённые спецификации с числом запусков и формулами бюджета. Они требуют отдельного решения на validation и отдельной версии реализации. Точные нормированные exp/geometric коэффициенты r=4 и ошибка совпадения среднего 32/15 сохраняются CPU-проверкой.

При проверке развёртывания обнаружен устаревший системный Python у служебного dispatcher. Он заменён явно заданным современным host interpreter; обучение использует прежний проверенный контейнер. Четыре начальные CPU-overfit проверки были остановлены до их завершения и помещены в приватный preflight-архив, затем запущены с окончательным source hash. Они не считаются пройденными проверками или обучениями основной сетки. GPU при этой корректировке не использовался.

## Отличия от исходной реализации

- Независимые `architecture`, `attention_filter`, `recurrent_steps`, `hops`, `weight_sharing`, `supervision`.
- Новый wrapper использует один исходный блок T раз; A пересчитывается на каждом внешнем шаге и фиксируется внутри r переносов.
- `inner_beta=0`, полный BPTT, без дополнительного solver/repair; legacy weights и исходные hashes сохранены.
- Пилот имеет stop 10000 при scheduler horizon 30000; финал стартует с новых весов и других seeds.
- Фактические размеры данных отражены явно, в отличие от неиспользуемых 100000/5000/5000 в старых config.
- Отдельные code/config/data hashes, resume проверки, стоимость и переходы бюджета; подготовленные результаты не выдаются за завершённые.
