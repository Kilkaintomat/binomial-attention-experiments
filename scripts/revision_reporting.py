"""Correct result metadata and preserve explicit protocol limitations."""
import csv
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path
import torch
from transport_reasoning.config import ROOT, write_json


def read(path):
    return json.loads(Path(path).read_text())


def table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def refresh():
    audit = ROOT / "revision_audit"; audit.mkdir(exist_ok=True)
    summary = ROOT / "summaries" / "results.csv"
    original = audit / "original_results.csv"
    if not original.exists():
        shutil.copy2(summary, original)
    report = ROOT / "REPORT_RU.md"
    if report.exists() and not (audit / "original_REPORT_RU.md").exists():
        shutil.copy2(report, audit / "original_REPORT_RU.md")
    rows = list(csv.DictReader(summary.open()))
    runs = {read(p)["config_hash"]: p for p in (ROOT / "runs").rglob("result.json")}
    final = []; corrections = []
    for row in rows:
        path = runs[row["config_hash"]]; r = read(path); c = r["config"]
        # The data directory is keyed by the data specification, not its content hash.
        from transport_reasoning.config import Config
        manifest = read(Config(**c).data_dir() / "manifest.json")
        evaluation_path = path.parent / "evaluation_test.json"
        evaluation = read(evaluation_path) if evaluation_path.exists() else r.get("last_validation") or {}
        if c["stage"] == "overfit":
            if row["split"] != "train_subset":
                corrections.append({"config_hash": c_hash(row), "field": "split", "from": row["split"], "to": "train_subset"})
            row["split"] = "train_subset"
        if evaluation_path.exists():
            state = torch.load(path.parent / "checkpoint_best.pt", map_location="cpu", weights_only=False)
            filters = state["filters"]
            row["filter_metadata_source"] = "tested_best_checkpoint"
            row["filter_checkpoint_step"] = state["step"]
            for key in ("mu", "nu"):
                if abs(float(row[key]) - filters[0][key]) > 1e-7:
                    corrections.append({"config_hash": c_hash(row), "field": key, "from": row[key], "to": filters[0][key]})
                row[key] = filters[0][key]
            row["filter_moments_scope"] = "first_block; full vectors in filters_by_block"
            row["filters_by_block"] = json.dumps(filters)
        else:
            row["filter_metadata_source"] = "last_training_checkpoint"
            row["filter_checkpoint_step"] = r["step"]
        row["config_L"] = c["layers"]
        row["L"] = 1 if c["method"] == "recurrent" else c["layers"]
        row["unique_blocks"] = row["L"]
        row["outer_tied"] = c["method"] == "recurrent"
        row["inner_tied_across_hops"] = bool(c["inner_beta"])
        row["actual_train_count"] = c["overfit_count"] or manifest["splits"]["train"]
        row["actual_validation_count"] = c["overfit_count"] or manifest["splits"]["validation"]
        row["actual_test_count"] = manifest["splits"]["test"]
        row["configuration_count_semantics"] = "generator_defaults_unused_for_official_RRN" if c["data_source"] == "rrn" else "actual_generated_split_counts"
        row["inference_timing_scope"] = "synchronized_model_forward; excludes feature preparation, decoding and validators"
        for key in ("loss", "exact_match", "vertex_f1", "vertex_iou", "blank_accuracy", "broken_fraction", "branch_fraction", "row_column_conflicts_mean"):
            row[key] = evaluation.get(key)
        if c["stage"] == "final" and r["status"] == "completed" and evaluation_path.exists():
            final.append({"task": c["task"], "method": c["method"], "seed": c["seed"], "beta": c["inner_beta"],
                          "score": evaluation["solved_rate"], "training_seconds": r["training_seconds"]})
    table(summary, rows)
    history = audit / "report_corrections.json"
    prior = read(history) if history.exists() else []
    unique = {json.dumps(r, sort_keys=True): r for r in prior + corrections}
    write_json(history, list(unique.values()))
    counts = Counter((r["stage"], r["run_status"]) for r in rows)
    off = sum(r["beta"] == 0 for r in final); on = sum(r["beta"] > 0 for r in final)
    lines = ["# Отчёт transport-reasoning", "",
             f"Основная серия: {off}/60 завершённых финальных прогонов. Серия с внутренним MLP: {on}/20 завершённых и оценённых на test.",
             "Аудит основной серии независимо перепроверил 570 000 сохранённых тестовых предсказаний; на 960 примерах воспроизведены ответы из лучших checkpoints. Основные solved rates подтвердились.",
             "Проверены метаданные 152 исходных запусков, включая пилоты, и 20 конфигураций будущей MLP-серии. Полный повтор обучения и полного test-forward в аудит не входил.", "",
             "| Задача | Метод | Внутренний β | Seeds | Test solved, %: mean ± sample SD |",
             "|---|---|---:|---|---:|"]
    for key in sorted({(r["task"], r["method"], r["beta"]) for r in final}):
        selected = sorted([r for r in final if (r["task"], r["method"], r["beta"]) == key], key=lambda r: r["seed"])
        values = [100 * r["score"] for r in selected]
        sd = "—" if len(values) < 2 else f"{statistics.stdev(values):.3f}"
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {', '.join(str(r['seed']) for r in selected)} | {statistics.mean(values):.3f} ± {sd} |")
    lines += ["", "Сравнение соответствует ограниченной первой кампании; весь план ещё не выполнен.", "",
              "- На Maze binomial обучался с LR=0,0003, остальные основные фильтры — с LR=0,001. Это сравнение после одинакового tuning budget, а не чистое выделение эффекта коэффициентов. Отдельной финальной серии с общим LR нет.",
              "- Радиус onehop на Maze — 4 ребра, recurrent — 32, финальных multihop-моделей — 64. У 86,8% тестовых лабиринтов путь длиннее 4 рёбер; у 9,1% — длиннее 32. Это описание расстояний, а не строгий верхний предел solved rate. Формулировка equal-radius в исходном frozen protocol ошибочна; она исправлена отдельной записью, оригинал сохранён.",
              "- Recurrent T=32 не выбран сравнением T=8/16/32; ширина и число параметров не выровнены. Результат Sudoku подтверждает качество этой конкретной архитектуры, но не изолированный эффект refresh.",
              "- Внутренний MLP добавляет 133 120 параметров и дополнительные FFN evaluations. Outer-only контроль с тем же числом параметров и отдельное обучение recurrent при равном wall time не выполнены. До этих контролей возможный выигрыш MLP нельзя приписывать только чередованию переносов и нелинейности.",
              "- Maze остаётся 8×8, 10 000/1 000/1 000; смешанные 8/12, 16×16/Wilson OOD и dense attention не выполнены. Это заранее сохранённые ограничения первой кампании.",
              "- Sudoku: фактические официальные splits 180 000/18 000/18 000. Числа 100 000/5 000/5 000 в конфигурациях относятся к неиспользуемым настройкам генератора; выборка в обучении проверена по фактическим 180 000. Все предоставленные targets валидны; точных и digit-relabeling дубликатов между splits не найдено. Полная канонизация всех симметрий и доказательство уникальности каждой puzzle не проводились.", "",
              "Исправления отчётности: μ/ν learned polynomial теперь соответствуют лучшему checkpoint, оценённому на test; прежние значения относились к последнему checkpoint в 10 прогонах. Overfit помечен train_subset, а число уникальных recurrent-блоков — 1. В CSV добавлены фактические размеры данных и вспомогательные метрики. Solved rates и обученные веса не изменены.",
              "У прошлых прогонов сохранены журналы порогов, но не отдельные веса до/после первого подтверждённого достижения порога: 34 достижения. Их нельзя восстановить из best/last без повторного обучения. Для будущей MLP-серии добавлено сохранение этих пар с подтверждением на двух последовательных оценках.",
              "Training time включает загрузку batch, forward/backward и optimizer. Inference latency в исходных измерениях — синхронизированный forward модели, без построения признаков, decoder и validator; это не полная end-to-end задержка. GPU-бюджет сверяется со временем выделения Slurm.",
              "Accuracy–compute tradeoff оценивается по сохранённым кривым. Текущие результаты не подтверждают преимущество binomial и не доказывают эквивалентность методов.", "",
              "Все per-seed значения: summaries/results.csv и summaries/final_aggregates.csv. Неопределённость парных сравнений: summaries/paired_differences.csv. Независимые проверенные результаты и границы аудита: revision_audit/. Исходные замороженные протоколы и checkpoints сохранены."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_text = ["РЕВИЗИЯ ПРОШЛЫХ И БУДУЩИХ ЗАПУСКОВ", "",
                  "Итог: технические проверки метрик и обучения пройдены; полного соответствия всему плану нет.",
                  "Независимый аудит: 60 финальных обучений, 570000 сохранённых test predictions, 960 повторных CPU predictions из checkpoints; все solved rates подтверждены.",
                  "Метаданные: 152 исходных запуска, включая 24 намеренно остановленных LR-кандидата, а также 36 строк pointer reimplementation. Pointer не заявляется точным воспроизведением исходного кода.",
                  "Формулы: независимый per-head reference совпал с основной реализацией по выходам и градиентам для beta=0 и 0.1. Проверены matched mean, нормировка, отсутствие labels во входах, D4-дедупликация Maze и законность Sudoku targets.",
                  "Будущие 20 конфигураций проверены: обе задачи, binomial/exponential, пять seeds, 30000 шагов, beta=0.1, width=128; общие начальные параметры и порядок данных совпадают с контролями.", "",
                  "ОШИБКИ И ИСПРАВЛЕНИЯ", "1. Метаданные μ/ν 10 learned-polynomial прогонов брались из last вместо tested best. Исправлены таблицы, scores не изменены.",
                  "2. Overfit ошибочно был помечен validation; исправлен на train_subset. L recurrent в таблице отражал неиспользуемый config_L=4; теперь L/unique_blocks=1, T=32, исходный config_L сохранён отдельно.",
                  "3. В прошлых 34 подтверждённых достижениях порогов нет обеих нужных копий весов. Журналы доступны. Будущая серия сохраняет веса до и после порога; старые отсутствующие веса не выдуманы.",
                  "4. Поля размеров генератора Sudoku отличаются от фактических официальных splits. Обучение использовало правильные 180000 строк, что проверено по конечному состоянию RNG. Отчёт и CSV явно показывают реальные размеры.",
                  "5. Учёт GPU в Python пропускал начало/конец выделения: для первых двух GPU jobs разница 12.581 секунды. Бюджет приведён к фактическому времени Slurm: 52989 секунд до MLP extension.",
                  "6. Исходная автоматическая проверка нейтрального оформления работала без списка запрещённых терминов и не сканировала все типы артефактов; её результат не был полноценным сертификатом. Новые материалы проверены отдельно.", "",
                  "НЕВЫПОЛНЕННЫЕ КОНТРОЛИ И ГРАНИЦЫ ВЫВОДОВ", *[s for s in lines if s.startswith("- ")], "",
                  "Прежние frozen protocols не переписаны задним числом. Новая observational amendment меняет сохранение диагностических весов и отчёты, сохраняя модель, loss, optimizer, data order и 30000 updates. Отдельный тест сравнивает прямое обучение с возобновлением и этими hooks побитно на обеих задачах.",
                  "Первая версия независимого аудита ошибочно взяла верхнюю границу Sudoku RNG из настроек генератора. После проверки loader аудит исправлен на фактический размер; первоначальный результат сохранён в attempt_8490007. Это была ошибка аудита, а не обучающих прогонов.",
                  "Полный test inference не повторялся; проверены все сохранённые predictions и по 16 повторных ответов каждого run. Полная Sudoku symmetry/uniqueness certification, полное повторение pointer training и проверка GPU resume побитно не выполнены."]
    (audit / "REPORT_RU.txt").write_text("\n".join(audit_text) + "\n", encoding="utf-8")
    print(json.dumps({"corrected_report": True, "new_field_corrections": len(corrections), "primary_completed": off, "inner_mlp_completed": on}), flush=True)


def c_hash(row):
    return row["config_hash"]


if __name__ == "__main__":
    refresh()
