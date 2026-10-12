# Directed reachability — фактический статус

Это адаптация ER/DAG к query-conditioned бинарной достижимости с сохранением всех вершин. Она не воспроизводит опубликованные scores исходной статьи.

Состояние очереди: completed. Завершено новых финальных обучений: 80. Полная исходная сетка — 90 финалов и 54 пилота.
Учтённое GPU-время завершённых выделений: 20.3564 ч из 24; активное выделение добавляется при accounting.

## Дизайн

Два независимых семейства er_exact_n_v1 и dag_exact_n_v1. Train n=16–17, validation n=18–19, ID n=16–17, near n=20–27, far n=32/64/128. По восемь равномерных разных запросов на граф.
Пять признаков вершины; маска I∨Bᵀ переносит маркер источника по направлению рёбер; readout только в target. Label, расстояния и component metadata не передаются в forward.
AdamW, batch64, 30000 обновлений, одинаковая последовательность batches и общие начальные тензоры при одинаковом seed. Pilots3000 сохраняют scheduler horizon30000.

## Покрытие и результаты

Выбранная по измеренной стоимости подматрица: {"status": "admitted", "models": ["F4-one", "F4-bin", "F4-exp", "F4-one-inner", "F4-bin-inner", "F4-exp-inner", "R32-one", "F4-geo"], "seeds": [11, 23, 37, 53, 71], "families": ["er", "dag"], "required_with_reserve_seconds": 84360.57378200535, "remaining_seconds_at_admission": 86160.72237135842, "selection_rule": "longest priority prefix with five seeds; minimum two models, reduced seed rounds only if needed", "source_hash": "a98bc726b261c66a552a50b3fcf5e7dc165f7ba3fe12a52a722fe377ba9a8ed8", "full_grid_admitted": false, "omitted_models": ["F4-learned"], "omitted_seeds": [], "additional_seconds_for_full_grid": 5065.020892680623, "before_any_new_test": true}

| family | model | split | n | seeds | BA mean | SD |
|---|---|---|---|---:|---:|---:|
| dag | F4-bin | stress | macro | 5 | 0.7736333333333334 | 0.02712942232255516 |
| dag | F4-bin | test_ID | macro | 5 | 1.0 | 0.0 |
| dag | F4-bin | test_far | macro | 15 | 0.9980261144952932 | 0.0022706784111851184 |
| dag | F4-bin | test_near | macro | 5 | 0.9999853559619792 | 3.274506447966271e-05 |
| dag | F4-bin-inner | stress | macro | 5 | 0.7797833333333333 | 0.029474895155783744 |
| dag | F4-bin-inner | test_ID | macro | 5 | 1.0 | 0.0 |
| dag | F4-bin-inner | test_far | macro | 15 | 0.9981445277575786 | 0.002351412737675994 |
| dag | F4-bin-inner | test_near | macro | 5 | 0.9999692921236173 | 6.866489903644359e-05 |
| dag | F4-exp | stress | macro | 5 | 0.7841666666666667 | 0.03774917217635373 |
| dag | F4-exp | test_ID | macro | 5 | 1.0 | 0.0 |
| dag | F4-exp | test_far | macro | 15 | 0.9983155912119875 | 0.002261896913322353 |
| dag | F4-exp | test_near | macro | 5 | 1.0 | 0.0 |
| dag | F4-exp-inner | stress | macro | 5 | 0.8215166666666667 | 0.03975294363663878 |
| dag | F4-exp-inner | test_ID | macro | 5 | 1.0 | 0.0 |
| dag | F4-exp-inner | test_far | macro | 15 | 0.998368241404111 | 0.0021809007796185545 |
| dag | F4-exp-inner | test_near | macro | 5 | 0.9999929932735426 | 1.5667516658475986e-05 |
| dag | F4-geo | stress | macro | 5 | 0.7792666666666667 | 0.028174456516497328 |
| dag | F4-geo | test_ID | macro | 5 | 0.99997642621405 | 5.27125878712633e-05 |
| dag | F4-geo | test_far | macro | 15 | 0.9980261144952932 | 0.0022706784111851184 |
| dag | F4-geo | test_near | macro | 5 | 0.999954340687126 | 5.0141898076046025e-05 |
| dag | F4-one | stress | macro | 5 | 0.7666666666666667 | 0.0 |
| dag | F4-one | test_ID | macro | 5 | 0.9987247490493717 | 0.0 |
| dag | F4-one | test_far | macro | 15 | 0.9896650555954527 | 0.0069323715811159155 |
| dag | F4-one | test_near | macro | 5 | 0.9976722264559001 | 3.2591255545523083e-05 |
| dag | F4-one-inner | stress | macro | 5 | 0.7666666666666667 | 0.0 |
| dag | F4-one-inner | test_ID | macro | 5 | 0.9987011752634217 | 5.2712587871312955e-05 |
| dag | F4-one-inner | test_far | macro | 15 | 0.989776912418719 | 0.006774705407321507 |
| dag | F4-one-inner | test_near | macro | 5 | 0.9976945160489507 | 1.2412670766236366e-16 |
| dag | R32-one | stress | macro | 5 | 0.7828999999999999 | 0.032142121464244125 |
| dag | R32-one | test_ID | macro | 5 | 1.0 | 0.0 |
| dag | R32-one | test_far | macro | 15 | 0.9984274480352539 | 0.002121576970346148 |
| dag | R32-one | test_near | macro | 5 | 1.0 | 0.0 |
| er | F4-bin | stress | macro | 5 | 0.8666666666666668 | 1.2412670766236366e-16 |
| er | F4-bin | test_ID | macro | 5 | 1.0 | 0.0 |
| er | F4-bin | test_far | macro | 15 | 0.9858246208661369 | 0.014882262222925435 |
| er | F4-bin | test_near | macro | 5 | 0.9997465554634004 | 7.331675838830821e-05 |
| er | F4-bin-inner | stress | macro | 5 | 0.8666666666666668 | 1.2412670766236366e-16 |
| er | F4-bin-inner | test_ID | macro | 5 | 1.0 | 0.0 |
| er | F4-bin-inner | test_far | macro | 15 | 0.9902373974112925 | 0.011373556457812857 |
| er | F4-bin-inner | test_near | macro | 5 | 0.9998389360067822 | 0.00010061574778252097 |
| er | F4-exp | stress | macro | 5 | 0.8666666666666668 | 1.2412670766236366e-16 |
| er | F4-exp | test_ID | macro | 5 | 1.0 | 0.0 |
| er | F4-exp | test_far | macro | 15 | 0.9866897908099811 | 0.01415694581657339 |
| er | F4-exp | test_near | macro | 5 | 0.9997766169628501 | 4.146707481860163e-05 |
| er | F4-exp-inner | stress | macro | 5 | 0.8666666666666668 | 1.2412670766236366e-16 |
| er | F4-exp-inner | test_ID | macro | 5 | 1.0 | 0.0 |
| er | F4-exp-inner | test_far | macro | 15 | 0.9894537343408426 | 0.012034934190558063 |
| er | F4-exp-inner | test_near | macro | 5 | 0.9998099626520649 | 0.00015350878264922433 |
| er | F4-geo | stress | macro | 5 | 0.8666666666666668 | 1.2412670766236366e-16 |
| er | F4-geo | test_ID | macro | 5 | 1.0 | 0.0 |
| er | F4-geo | test_far | macro | 15 | 0.9861616076224429 | 0.014618659858696925 |
| er | F4-geo | test_near | macro | 5 | 0.9997594797641991 | 9.192233939144121e-05 |
| er | F4-one | stress | macro | 5 | 0.7637833333333334 | 0.006447329335124428 |
| er | F4-one | test_ID | macro | 5 | 0.9504320983002612 | 0.0006941908186755048 |
| er | F4-one | test_far | macro | 15 | 0.7921552139947863 | 0.07141952501770919 |
| er | F4-one | test_near | macro | 5 | 0.9149008351749887 | 0.0012407359461848527 |
| er | F4-one-inner | stress | macro | 5 | 0.7651333333333333 | 0.0034286375654996974 |
| er | F4-one-inner | test_ID | macro | 5 | 0.9505949441757391 | 0.00029571768750263655 |
| er | F4-one-inner | test_far | macro | 15 | 0.7900975358787387 | 0.07187276502028704 |
| er | F4-one-inner | test_near | macro | 5 | 0.9144173741643122 | 0.0003210578691311 |
| er | R32-one | stress | macro | 5 | 0.8933333333333333 | 0.036514837167011066 |
| er | R32-one | test_ID | macro | 5 | 1.0 | 0.0 |
| er | R32-one | test_far | macro | 15 | 0.9957216257450909 | 0.007073448184204471 |
| er | R32-one | test_near | macro | 5 | 0.9999583835954813 | 6.040865437334675e-05 |

## Проверки и ограничения

Фактические проверки: DATA_AUDIT_REACHABILITY.json, PREFLIGHT_REACHABILITY.json и JUnit. Изоморфизм проверяется отдельно от точного равенства adjacency; неполный аудит не доказывает отсутствие изоморфных дублей.
Основные CSV содержат все фактически записанные seeds. Отсутствующие и частичные запуски не заменяются ожидаемыми результатами.
Extra compute использует фиксированную подвыборку; lambda/rho сохраняют значения обучения. Learned r>4 — unsupported. Таблицы fixed/extra разделены.
Mean±SD относится к seeds. Парные интервалы exploratory. Все запросы одного графа являются зависимыми; восемь пар не считаются восемью независимыми графами. Graph-bootstrap интервалы в основных таблицах пока не рассчитываются.
Внутренний MLP меняет параметры и объём вычислений. Recurrent имеет другую архитектуру. Границы локальности и degree shortcuts ограничивают причинные выводы.
До завершения оценок нет ответа о переносе на большие n, длинные пути и эффекте inner MLP; результаты Maze/Sudoku не подставляются.

## Воспроизведение

См. docs/REACHABILITY_RU.md; большие графы, checkpoints, logits, зависимости upstream и приватные настройки исключены из git. Внешняя публикация этой серии не выполнялась.
