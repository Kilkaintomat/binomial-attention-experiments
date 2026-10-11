# Sudoku local mask v1

Отдельная серия из публичного снимка `9ea79b57259e9a6619098279ec4274b8efb4dd13`. Она не меняет исходный `src/`, старые Config hashes, данные или две существующие кампании. Новая версия Config/TaskData принимает `support=graph|grid4`; прежний `graph` сохраняет связи строки/столбца/квадрата. В таблицах он называется `relational`.

## Установка и окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export PYTHONPATH="$PWD/src:$PWD"
```

Host dispatcher требует Python ≥3.10; модель и проверки исполняются в разрешённом контейнере/окружении. Фактические версии сохраняются в `checks/environment.json` и `checks/installed_versions.txt`. Старый `requirements.lock` — архивный список пакетов, а не проверенный переносимый lockfile с hashes. Совпадение версий и оборудования для сравнения времени проверяется отдельно.

Обучение, smoke, overfit, модельные тесты и отчёт выполняются только в реальном scheduler allocation. Подготовка JSON и отправка `sbatch` могут выполняться без GPU. Существующее окружение подключается read-only; запись идёт только в новый каталог.

## Данные

Используется существующий общий проверенный кэш официального Sudoku-Hard. При его отсутствии подготовьте исходные RRN CSV предусмотренным загрузчиком в CPU-выделении:

```bash
export TRANSPORT_RRN_DIR=<official_csv_directory>
python -m transport_reasoning prepare-data --config <archived_rrn_sudoku_config.json>
export LOCAL_MASK_DATA_DIR=<verified_npz_cache>
export LOCAL_MASK_LEGACY_ROOT=<read_only_legacy_artifact_root>
export LOCAL_MASK_OUTPUT=experiments/sudoku_local_mask_v1
export LOCAL_MASK_CODE_COMMIT=$(git rev-parse HEAD)
```

`support` не входит в адрес кэша: обе маски читают одни puzzle/target массивы. Data hash, hashes массивов и отдельные mask hashes сохраняются в аудите. 38 исходных признаков не меняются. Валидатор всегда проверяет полные правила Sudoku.

## Проверки

В CPU allocation:

```bash
python -m research.sudoku_local_mask_v1 checks
```

Проверяются геометрия grid4 и relational, нулевые запрещённые attention, нормировка, конечные gradients, фактические 4/16/32 переноса, 32 обновления R32, нулевая дальняя производная F4-onehop, r1-эквивалентность, явный полином float64, неизменный legacy forward, отсутствие target в inputs/mask и точный interrupted/resume. Аудит официальных данных включает все splits; 15 исторических финальных контролей допускаются только после проверки инициализации, всех 30000 batch RNG, scheduler/optimizer, best checkpoint и сохранённых предсказаний. При несовместимости хотя бы одного контроля консервативно планируются все 48 новых обучений.

Команда checks также выполняет smoke: 6 условий, seed227, по10 обновлений на64 train Sudoku; проверяет finite loss/gradients и изменение весов. Затем отдельно выполните все шесть сочетаний:

```bash
python -m research.sudoku_local_mask_v1 overfit --label F4-onehop --support graph
python -m research.sudoku_local_mask_v1 overfit --label F4-onehop --support grid4
python -m research.sudoku_local_mask_v1 overfit --label F4-binomial --support graph
python -m research.sudoku_local_mask_v1 overfit --label F4-binomial --support grid4
python -m research.sudoku_local_mask_v1 overfit --label R32-onehop --support graph
python -m research.sudoku_local_mask_v1 overfit --label R32-onehop --support grid4
```

Каждый overfit ограничен1000 обновлениями/3300 секундами и64 train-задачами, seed229. Повторная команда возобновляет собственный checkpoint. Слабое запоминание у grid4 не блокирует остальные модели само по себе; NaN, ошибки маски/локальности или незавершённое исполнение проверок блокируют массовую очередь. Overfit не является оценкой обобщения.

## Протокол и очередь

После фиксации кода и успешного аудита:

```bash
python -m research.sudoku_local_mask_v1 prepare
```

Manifest включает ровно48 строк:18 пилотов и30 финальных результатов. При успешном переиспользовании15 relational финалов новых обучений33. Во всех группировках присутствуют stage, support, model и seed. Слабые модели из сетки не удаляются. Пилоты останавливаются на10000 при cosine horizon30000; финалы стартуют с новых весов на seeds11,23,37,53,71 и идут30000 шагов.

В приватном `.cache/site.sh` задайте и экспортируйте `RUN_PROJECT`, `RUN_LEGACY_PROJECT`, `RUN_PRIOR_SERIES_PROJECT`, `RUN_IMAGE`, `RUN_PYTHON_IN_CONTAINER`, `RUN_DATA_IN_CONTAINER`, `RUN_HOST_PYTHON`, `RUN_ACCOUNT`, `RUN_GPU_PARTITION`, `RUN_CPU_PARTITION`, `RUN_GPU_GRES`, `LOCAL_MASK_CODE_COMMIT`. Файл исключён из git. Значения описывают только конкретную площадку и не публикуются.

```bash
source .cache/site.sh
export PYTHONPATH="$RUN_PROJECT/src:$RUN_PROJECT"
"$RUN_HOST_PYTHON" -m research.sudoku_local_mask_v1 submit
```

Создаётся собственное held GPU-задание до20минут для профиля, accounting и CPU gate. Gate читает предыдущие очереди и ждёт их завершения, не изменяя/отменяя их задания. Затем проверяет научные тесты, smoke/overfit и отдельный бюджет. По умолчанию дополнительный GPU-бюджет отсутствует.

Только после явного подтверждения пользователем абсолютных лимитов:

```bash
"$RUN_HOST_PYTHON" -m research.sudoku_local_mask_v1 allow --gpu-hours <approved_cap> --wall-hours <approved_wall_cap>
```

Считается весь реально выделенный GPU-интервал по `sacct`, включая профиль, validation, checkpoint и test. Wall-clock начинается с первого GPU-выделения этой серии. Каждое следующее задание максимум2часа, одновременно один GPU; остаток резервируется до отправки. Профиль измеряет все6 условий. Оценка включает18 пилотов,15/30 новых финалов, full validation/test, checkpoint,900с резерва для измерений/диагностики и ещё20%. Если лимита недостаточно, до массового обучения запись `needs_budget.json` содержит точный недостающий остаток. Бюджет, seeds и длина обучения автоматически не расширяются и не уменьшаются.

Неожиданная ошибка останавливает собственную цепочку; failed seed сохраняется. `budget_stopped` допускает продолжение только с подтверждённым дополнительным лимитом; повторная команда allow возобновляет собственную очередь после проверок. Никаких сообщений другим людям или автоматической публикации нет.

## Resume и test

В разрешённом GPU allocation, соблюдая тот же учёт ресурсов:

```bash
python -m research.sudoku_local_mask_v1 train --config experiments/sudoku_local_mask_v1/configs/pilot/grid4_F4-binomial_seed101.json --resume
python -m research.sudoku_local_mask_v1 evaluate --config experiments/sudoku_local_mask_v1/configs/final/grid4_F4-binomial_seed11.json
```

Без явного `--resume` checkpoint не перезаписывается. Проверяются config, научный code hash, data hash и mask hash; восстанавливаются optimizer, scheduler и все RNG. Пилотам/незавершённым финалам test запрещён. Диспетчер оценивает test после обучения всей сетки и фиксированного протокола. Clues копируются, пустые клетки — argmax, внешнего решателя нет.

## Hashes и версия

`source_sha256` включает исходный `src/` и научные файлы новой серии: config/data/model/training/measure. Полный `snapshot_sha256` дополнительно фиксирует audit, tests runner, analysis/reporting, dispatcher и shell wrapper. Протокол проверяет оба hash перед массовым запуском; commit сохраняется отдельно. Поэтому презентационная правка не приписывает старому checkpoint другой научный source hash, а смена модели/данных/обучения всегда обнаруживается. Исторические результаты сохраняют свой исходный source hash.

Существующие файлы не заменяются. При изменении вычислений после начала основной сетки нужна явно отмеченная версия и раздельные результаты. Smoke/overfit начаты на отдельном зафиксированном commit научной реализации и имеют тот же научный hash, что будущие пилоты/финалы.

## Анализ и отчёт

```bash
python -m research.sudoku_local_mask_v1 report
```

Команда выполняется в allocation и строит `REPORT_SUDOKU_LOCAL_MASK_RU.md`, CSV по каждому seed, агрегаты, пороги, position accuracy, кривые solved/CE/blank accuracy против updates/training seconds и quality-cost scatter. Отсутствующие измерения остаются отсутствующими.

Основная величина — парная разность разностей `I=G_grid−G_rel` в процентных пунктах. Для G, I, mask changes и R32 comparisons используются sample SD и95% percentile bootstrap по seeds,20000 повторов, фиксированный seed90421. Неполные пары отмечаются; пять seeds дают ограниченную точность. Положительное I не означает высокого абсолютного качества и может отражать деградацию onehop. Смена графа одновременно меняет степень, диаметр и соответствие ограничениям — это не чистая причинная абляция диаметра.

Числа training/checkpoint/validation/elapsed разделены. Полные выделенные GPU-часы учитываются на уровне цепочки; запуск может пересекать chunks. Inference заново измеряется и для исторических весов: batch1/64, warmup10,30 повторов, synchronization, forward-only/полный pipeline отдельно. Плотные tensors не превращаются в sparse из-за маски.

Первые64 validation-задачи используются для micro blank accuracy по4углам/28краям/49внутренним клеткам. Число пустых клеток и числитель сохраняются для каждой группы. Широкая диагностика и дополнительные топологические контроли не включаются.

Официальный test известен из прежней работы. Серия честно обозначается как последующее исследование на известном benchmark.
