# Binomial attention: эксперименты и результаты

Публичный исследовательский снимок сравнения binomial, exponential, geometric и learned polynomial attention с one-hop и recurrent моделями на Maze и Sudoku. Основная серия завершена: **60 обучений по 30 000 обновлений, пять seeds на каждую пару задача × метод**. Срез: 10 октября 2026, 20:44:53 UTC.

**[Большой подробный отчёт на русском](REPORT_RU.md)** · **[Все таблицы запусков](docs/EXPERIMENT_TABLES_RU.md)** · **[CSV результатов](results/tables/results.csv)**

В отчёте описаны данные, формулы, архитектура, инициализация, loss, выбор гиперпараметров, обучение, checkpoint/resume, измерение времени, каждый этап экспериментов, найденные ошибки и ограничения выводов.

| Метод | Maze test solved, % | Sudoku test solved, % |
| --- | --- | --- |
| One hop | 0,62 ± 0,04 | 5,49 ± 0,33 |
| Binomial | 98,76 ± 0,18 | 5,23 ± 1,08 |
| Truncated exponential | 99,58 ± 0,22 | 5,76 ± 0,94 |
| Finite geometric | 99,88 ± 0,04 | 5,34 ± 0,86 |
| Learned polynomial | 99,88 ± 0,13 | 6,05 ± 0,44 |
| Recurrent | 96,40 ± 6,94 | 94,77 ± 0,37 |

Здесь mean ± sample SD по пяти seeds. В этой постановке особое преимущество binomial не подтверждено. Рекуррентная модель отличается также числом параметров, глубиной вычисления и временем; сравнение не выделяет причинный эффект одного механизма.

![Финальные результаты](results/figures/final_solved.png)

Новая финальная серия с внутренним MLP **не завершена: 0/20** на момент среза. Её 24 коротких пилота приведены отдельно. Частичные логи первого финального MLP-запуска не считаются финальным результатом. Датированный [статус выполнения](results/metadata/execution_snapshot.json) не является живым монитором.

## Содержимое

- [REPORT_RU.md](REPORT_RU.md): подробный отчёт, 24 раздела.
- [src](src): исходники моделей, генераторов, обучения, проверки решений и метрик.
- [tests](tests): проверки математических операций, валидаторов и воспроизводимого продолжения.
- [results/records](results/records): конфигурации, result/evaluation JSON и validation-кривые запусков.
- [results/tables](results/tables): агрегаты, парные различия, timing и pointer diagnostics.
- [results/protocols](results/protocols): замороженные протоколы и очереди.
- [results/audit](results/audit): результаты независимой ревизии и её ограничения.
- [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json): происхождение снимка и изменения переносимых служебных скриптов.

Исходные PDF/XLSX, сторонние исходные данные, model checkpoints и массивы индивидуальных предсказаний не включены. Maze воспроизводится генератором. Sudoku использует официальный sudoku-hard из [кода Recurrent Relational Networks](https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/master/tasks/sudoku/data.py). Наша recurrent-модель не повторяет архитектуру этой статьи.

## Окружение и быстрая проверка

Рабочая среда исходных запусков: Python 3.11.11, PyTorch 2.6.0+cu124, NumPy 2.2.2, NVIDIA A100-SXM4-80GB. Точные сведения сохранены в [gpu_environment.json](results/metadata/gpu_environment.json).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
python scripts/build_public_assets.py
python scripts/check_public_snapshot.py
```

На вычислительном кластере выполняйте tests, подготовку данных и построение рисунков внутри CPU allocation, а обучение — внутри GPU allocation. `requirements.lock` — архивный перечень пакетов объединённого контейнера и venv; в нём есть повторяющиеся версии pip/setuptools. Он сохранён как свидетельство окружения и не является готовым входом для `pip install -r`. Установка через pyproject обеспечивает запуск, но сама по себе не гарантирует побитное совпадение с историческим окружением.

Проверить полный путь на маленьком синтетическом Maze можно так:

```bash
python -m transport_reasoning prepare-data --config configs/cpu_smoke_maze.json
python -m transport_reasoning train --config configs/cpu_smoke_maze.json
```

Новые артефакты пишутся в `experiments/transport_reasoning`; опубликованные `results` остаются архивом. Запускайте команды из корня репозитория: вычисление source hash использует относительные пути `src`.

## Повторное обучение финальной конфигурации

Все фактически использованные конфигурации находятся рядом с метриками в `results/records`. Например, Maze/binomial/seed 71:

```bash
python -m transport_reasoning prepare-data --config results/records/maze/binomial/0cf228f2a68c50e7/seed_71/config.json
python -m transport_reasoning train --config results/records/maze/binomial/0cf228f2a68c50e7/seed_71/config.json
```

Вторая команда требует CUDA и действующий Slurm allocation; автоматического заказа GPU она не делает. Пример с ресурсами, заданными пользователем:

```bash
srun --account="$TRANSPORT_ACCOUNT" --partition="$TRANSPORT_GPU_PARTITION" \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=02:00:00 \
  .venv/bin/python -m transport_reasoning train \
  --config results/records/maze/binomial/0cf228f2a68c50e7/seed_71/config.json
```

Для Sudoku скачайте sudoku-hard по ссылке из официального upstream-кода, распакуйте `train.csv`, `valid.csv`, `test.csv` в одну директорию и задайте `TRANSPORT_RRN_DIR` её путём. Файлы должны содержать 180 000 / 18 000 / 18 000 строк. SHA-256 исходных CSV и manifest приведены в [протоколе](results/protocols/mlp_final_protocol.json). Поля train_count/val_count/test_count в старых RRN-конфигах не ограничивают эти CSV: фактические размеры исправлены в сводных таблицах и подробно разобраны в отчёте.

Выберите Sudoku config из `results/records/sudoku`, затем выполните те же `prepare-data` и `train`. Во время train выполняются validation и выбор лучшего checkpoint. После завершения всех 30 000 обновлений выполните test отдельно; для приведённого выше Maze-примера команда такая:

```bash
python -m transport_reasoning evaluate \
  --config results/records/maze/binomial/0cf228f2a68c50e7/seed_71/config.json \
  --checkpoint experiments/transport_reasoning/runs/maze/binomial/0cf228f2a68c50e7/seed_71/checkpoint_best.pt \
  --split test
```

Команда evaluate также исполняется внутри GPU allocation для CUDA-конфигурации. Для resume используется `train --config ... --resume <checkpoint_last.pt>`; для старых запусков необходимы не включённые сюда веса. Из CSV восстановить состояние optimizer невозможно.

## Служебные скрипты и границы воспроизводимости

`scripts/cpu_prepare.py` и `scripts/pointer_bridge.py` сохранены как исторические оркестраторы и ожидают исходные документы в `/inputs`. Первый также устанавливает зависимости и готовит всю кампанию. Они не являются входной командой для публичной копии без этих документов. Ядро pointer-задачи доступно в [data/pointer.py](src/transport_reasoning/data/pointer.py), а весь обучающий цикл — в `pointer_bridge.py`; исторический workbook audit выполняется после него и требует исходный XLSX.

Для продолжения уже начатой приватной кампании использовались `mlp_final_campaign.py`, `audited_mlp_runtime.py`, `chunked_mlp_runtime.py` и `mlp_chunk_scheduler.py`. Они ожидают полный рабочий каталог с бюджетом, протоколами, данными и checkpoint. Публичный снимок сам по себе таким каталогом не является. Перед переносом задайте `TRANSPORT_ROOT`, `TRANSPORT_INPUTS`, `TRANSPORT_IMAGE`, `TRANSPORT_ACCOUNT`, `TRANSPORT_GPU_PARTITION`, `TRANSPORT_CPU_PARTITION` и проверьте политику своего scheduler. Исторический dispatcher не следует запускать вслепую как команду новой кампании.

Архивный аудит заново проверил 570 000 сохранённых test-предсказаний и повторил inference на 960 примерах из checkpoints. Эти исходные массивы и веса не опубликованы; публичная проверка сверяет скаляры evaluation JSON, конфигурации, hashes, validation-кривые, выбор best checkpoint и агрегаты. [Результат проверки публичного снимка](results/metadata/publication_validation.json) описывает этот более узкий объём проверки.

Нерешённые пункты — MLP final, OOD Maze, dense controls, parameter-matched и equal-wall-time controls, полная Sudoku symmetry/uniqueness certification — перечислены в отчёте. Успешная техническая проверка не означает завершения всего исходного плана.
