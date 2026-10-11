# Directed graph reachability: воспроизведение новой серии

Серия `directed_reachability_v1` подготовлена в отдельной ветке от
`e9d676f3bd02a99c05913243c4590255e5a48394`. Старые исходники моделей,
фильтров, данных, training и evaluation не изменены. Новая задача использует
отдельный пакет `transport_reasoning.reachability` и собственные конфигурации.
Публикация и push этой серии не выполняются.

## Данные и первоисточники

Первоисточник: [Transformers Meet Directed Graphs](https://proceedings.mlr.press/v202/geisler23a.html)
и [исходный код](https://github.com/google-deepmind/digraph_transformer),
commit `22b880e8b00486e3d2ab916150d0138f538d5c50`. Исходный код и статья
скачиваются в отдельный игнорируемый каталог inputs; лицензия upstream
сохраняется без изменений. Его зависимости JAX не устанавливаются.

Это не готовый скачиваемый датасет: протокол задаёт новую генерацию
`er_exact_n_v1` и `dag_exact_n_v1`. Сохраняются все n вершин; largest weak
component не выделяется. Размеры и параметры генераторов закреплены в
PROTOCOL_REACHABILITY.json. Всего 272000 случайных графов, 2176000 запросов,
а также 6000 stress-шаблонов с 12000 парными запросами.

Метки считает directed BFS, независимо проверяемый транзитивным замыканием.
Маска равна I∨Bᵀ. Upstream CONNECTED_CA использует внутреннее имя dist_adj,
которое попадает в adjacency-ветку loader. Комбинация flatten и default
meshgrid восстанавливает транспонированную матрицу; её направление проверено
на асимметричной цепочке. Эти наблюдения о коде не являются утверждениями
об ошибочности опубликованных scores. Наши scores не считаются их точной
репликацией.

Случайные потоки независимы для семейства, split, n, graph index, попытки,
query и перестановки. Перестановки фиксируются при генерации, онлайн
аугментация не используется. DAG rank и ID не являются признаками.
Точные дубли adjacency запрещены между всеми split одного family/n.
Directed WL служит только группировкой для ограниченного точного
изоморфного аудита. Количество проверенных/непроверенных пар и найденные
изоморфизмы сохраняются; отсутствие изоморфных дублей без полной проверки
не заявляется.

## Окружение и CLI

Установить Python-пакет можно командой `pip install -e '.[reachability]'`.
`requirements-reachability.lock` фиксирует фактически проверенные версии
окружения. На сервере используется существующий контейнер и Python runtime
с зависимостями; исходники новой задачи задаются через отдельный PYTHONPATH.
Сведения о приватных путях и аккаунтах находятся только в игнорируемом
`.cache/site.sh`. Никакого sudo или установки системных пакетов нет.

Все compute-команды выполняются в scheduler allocation. Рабочий интерфейс:

```bash
python -m transport_reasoning.reachability --help
python -m transport_reasoning.reachability prepare-data --family er --n 16 --dry-run
python -m transport_reasoning.reachability prepare-data --family er --n 16
python -m transport_reasoning.reachability prepare-data --stress
python -m transport_reasoning.reachability audit-data
python -m transport_reasoning.reachability build-grid
python -m transport_reasoning.reachability preflight --seconds 1500
python -m transport_reasoning.reachability profile
python -m transport_reasoning.reachability run-queue --seconds 1500 --remaining-seconds 86400
python -m transport_reasoning.reachability evaluate --config <effective_config.json>
python -m transport_reasoning.reachability baselines
python -m transport_reasoning.reachability report
```

Для малого набора задаётся отдельный REACH_DATA и флаг --small. Команды
`scripts/reachability/data_job.py small`, `generate-worker`, `audit`
организуют малые проверки, четыре независимых CPU-работника и финальный
аудит полного manifest. Полные данные не подменяются малым набором.

## Очередь и бюджет

Потолок новой кампании — 24 GPU-часа по разделу 2 запроса. Учитываются
фактические выделения, включая preflight, profile, ошибки, evaluation и
накладные расходы. Старые бюджеты не используются. Покупок и увеличения
квоты нет. До запуска пилотов профиль определяет стоимость сетки с резервом
20%. Если 144 обучения не помещаются, выбирается симметричная ER/DAG
подматрица по заранее заданному приоритету, минимум one-hop и binomial.
Сначала выбирается самый длинный доступный префикс моделей на пяти seeds;
только если даже первые два метода не помещаются, сокращается число
полных раундов seeds. Состав и недостающий бюджет записываются до test.

GPU-задания запрашивают одну GPU, 4 CPU, 8 GiB RAM и до 30 минут.
Не фиксируется конкретный GPU-узел; тип оборудования сохраняется в
измерениях. Короткий слот повышает возможность backfill, но не гарантирует
момент выдачи ресурсов. Внутри кампании одновременно работает не более
одной GPU; существующие серии не отменяются и не изменяются.

CPU-генерация разбита на четыре задания по 1 CPU/2 GiB/20 минут.
Первоначальная попытка массива из 31 задания была отклонена лимитом числа
заявок; генератор теперь распределяет все те же части между четырьмя
работниками, не обходя квоту и не меняя данные.

После предварительных проверок запуск scheduler:

```bash
bash scripts/reachability/host.sh submit <successful_checks_job_id>
bash scripts/reachability/host.sh status
```

Следующий слот появляется только после учёта фактического времени
предыдущего. Atomic checkpoints сохраняют optimizer, scheduler, RNG,
global step и hashes. Любая несовместимость config/source/data останавливает
возобновление. Ошибка allocation останавливает автоматическую цепочку;
неудачные артефакты сохраняются для ревизии. Лимит не повышается автоматически.

## Выбор моделей и результаты

54 пилота и 90 финалов — полная плановая сетка. LR выбирается отдельно для
каждой model/family по full validation в конце 3000 шагов seed101;
seed103 только подтверждает выбор. Финалы обучаются с нуля, раундами seeds.
Configs и best checkpoints замораживаются до доступа к новым test scores.
Отчёт обозначает неполное покрытие и не выдаёт пилоты за финальные результаты.

Fixed compute использует полные тесты. Extra compute использует первые
500 графов каждого far n и первые 50 шаблонов каждой stress-клетки согласно
фиксированному manifest. Для exp/geo lambda/rho сохраняются от train r=4;
learned r>4 отмечается unsupported. Сохраняются logits/labels/graph IDs,
finite checks, нормы состояний и отдельные измерения времени.

Все запросы графа зависимы; stress positive/negative оцениваются вместе.
Основные интервалы — exploratory paired-seed bootstrap. При одном seed
SD и CI не выдумываются. Отдельная функция graph-bootstrap группирует
запросы по graph ID, но не смешивает эти интервалы с seed SD.

`REPORT_REACHABILITY_RU.md` и CSV/графики в отдельном report namespace
воспроизводятся из фактически сохранённых результатов. Пустая таблица
означает отсутствие оценок, а не нулевое качество. Исходные данные,
checkpoints и сырые предсказания в git не входят.

## Фактическая подготовка и принятый состав

Полная генерация и аудит завершены. Выполнены 36 CPU-тестов, установка
пакета, точное восстановление optimizer/scheduler/RNG и 18 GPU-smoke
проверок; все 9 архитектур прошли overfit-проверку.

Измеренная оценка полной сетки с резервом 20% — 25,34 GPU-часа плюс уже
выполненные проверки. В доступный остаток 24 GPU-часов допущены первые
8 моделей по приоритету, включая one-hop с inner MLP, обе семьи и все
5 финальных seeds: 48 пилотов и 80 финалов. Learned polynomial остаётся
в подготовленном плане продолжения. Новые test-результаты при выборе
состава не использовались.

Дополнительный точный изоморфный аудит завершил все WL-кандидаты: у ER
топология 3 из 5000 ID-test графов встречается в train, у DAG — 610 из 5000
(12,2%). Это число затронутых тестовых графов, а не число пар совпадений.
Данные не менялись. ID-test не является топологически непересекающимся
holdout; near/far отличаются по n. Подробности:
`results/reachability/DATA_ISOMORPHISM_COMPLETE.json`.
