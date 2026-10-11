"""Explicit user amendment: five new finals in the original queue order.

The frozen training implementation, configs, protocols and checkpoints remain
unchanged. This separately hashed dispatcher selects which configs to execute.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

MODULE = 'research.recurrent_filter_sudoku_v1'
FROZEN = 'c3b99115ee9b0a49b0a238609eec282ee722bf7bd09645d2d6013c20ea011ca7'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection():
    from research.recurrent_filter_sudoku_v1.config import code_hash, grid, root
    assert code_hash() == FROZEN, 'Frozen implementation changed'
    amendment = json.loads((root() / 'amendments/final_cap_5.json').read_text())
    assert amendment['runtime_sha256'] == digest(__file__), 'Dispatcher changed'
    assert amendment['original_admission_sha256'] == digest(root() / 'admission.json')
    assert amendment['original_protocol_sha256'] == digest(root() / 'protocol.json')
    original = grid(include_b32=True)
    fresh = [c for c in original if c.stage == 'final' and c.label() != 'R32']
    kept = fresh[:5]
    assert [(c.label(), c.seed) for c in kept] == [('R16', 11), ('B16', 11), ('B32', 11), ('R16', 23), ('B16', 23)]
    assert [c.hash() for c in kept] == amendment['selected_config_hashes']
    assert [c.hash() for c in fresh[5:]] == amendment['omitted_config_hashes']
    for c in fresh[5:]:
        assert not c.run_dir().exists(), 'Omitted run already has artifacts; preserve and review'
    configs = [c for c in original if c.stage != 'final' or c.label() == 'R32' or c.hash() in amendment['selected_config_hashes']]
    assert len(configs) == 22  # 12 pilots + five new finals + five historical controls.
    return amendment, configs


def rewrite_container_command(command):
    command = list(command)
    marker = ['-m', MODULE, 'run']
    if command and command[0] == 'apptainer':
        for i in range(len(command) - 2):
            if command[i:i+3] == marker:
                return command[:i] + ['/workspace/scripts/recurrent_final_cap_runtime.py', 'run'] + command[i+3:]
    return command


def checks():
    from research.recurrent_filter_sudoku_v1.campaign import preflight
    preflight()
    amendment, configs = selection()
    original = ['apptainer', 'exec', 'image', 'python', '-m', MODULE, 'run', '--seconds', '7140']
    assert rewrite_container_command(original) == ['apptainer', 'exec', 'image', 'python', '/workspace/scripts/recurrent_final_cap_runtime.py', 'run', '--seconds', '7140']
    assert rewrite_container_command(['sacct', '-X']) == ['sacct', '-X']
    assert len([c for c in configs if c.stage == 'pilot']) == 12
    assert len([c for c in configs if c.stage == 'final' and c.label() == 'R32']) == 5
    return {'status': 'passed', 'new_finals': 5, 'omitted_finals': 10, 'pilots': 12,
            'historical_controls': 5, 'source_sha256': FROZEN, 'runtime_sha256': digest(__file__)}


def run(args):
    from research.recurrent_filter_sudoku_v1 import campaign, reporting
    from research.recurrent_filter_sudoku_v1.config import root
    from transport_reasoning.config import write_json
    amendment, configs = selection()
    manifest = campaign.read(root() / 'manifest.json')
    for row in manifest['rows']:
        if row['config_hash'] in amendment['omitted_config_hashes']:
            row.update(status='omitted_by_user_amendment', amendment='final_cap_5')
    manifest['active_amendment'] = 'amendments/final_cap_5.json'
    write_json(root() / 'manifest.json', manifest)
    campaign.grid = lambda include_b32=False: configs
    original_report = reporting.report

    def amended_report():
        result = original_report()  # Retain every planned, omitted and historical row.
        report = Path('REPORT_RECURRENT_SUDOKU_RU.md')
        content = report.read_text()
        old = 'Завершено новых обязательных обучений: 16/22.'
        content = content.replace(old, 'Завершены 12 пилотов и 5 новых финальных обучений по сокращённому плану.')
        content = content.replace('Финалы R16/B16/R32: 5 seeds, 30000; новые финальные веса не продолжают пилоты.',
                                  'Исходный план финалов сокращён пользователем: R16/B16 — seeds 11 и 23; B32 — seed 11. Каждый новый финал — 30000 шагов с новой инициализацией; пять R32 сохранены как исторические контроли.')
        notice = ('> Изменение плана final_cap_5: по запросу пользователя выполнены только первые пять новых финалов в исходном порядке; десять оставшихся не запускались. '
                  'Решение принято после просмотра части validation, до новых test-оценок, ради более раннего запуска локальных масок. '
                  'Выбор не основан на качестве. Сравнения предварительные: две пары R16/B16 и один B32; исходная цель сравнения на пяти seeds не выполнена.\n\n')
        report.write_text(notice + content)
        path = root() / 'report/report_manifest.json'
        summary = campaign.read(path)
        summary.update(active_amendment=amendment, amended_new_final_count=5,
                       original_five_seed_design_completed=False, inference='exploratory_reduced_seed_count')
        write_json(path, summary)
        return result

    reporting.report = amended_report
    status = campaign.run_slot(args.seconds, args.remaining_seconds)
    if status == 'completed':
        for c in configs:
            if c.stage != 'final' or c.label() == 'R32':
                continue
            result = campaign.read(c.run_dir() / 'result.json')
            assert result['status'] == 'completed' and result['completed_steps'] == 30000
            assert (c.run_dir() / 'evaluation_test.json').exists()
            assert (c.run_dir() / 'measurement.json').exists()
        completion = campaign.read(root() / 'completion.json')
        completion.update(active_amendment='amendments/final_cap_5.json', new_final_trainings=5,
                          omitted_final_trainings=10, original_design_complete=False,
                          runtime_sha256=digest(__file__))
        write_json(root() / 'completion.json', completion)
    write_json(root() / ('queue/slot_result_' + os.environ['SLURM_JOB_ID'] + '.json'),
               {'status': status, 'amendment': 'final_cap_5', 'runtime_sha256': digest(__file__)})
    print(json.dumps({'status': status, 'new_final_limit': 5}), flush=True)


def main():
    mode = sys.argv[1]
    if mode == 'host':
        assert sys.argv[2:] == ['-m', MODULE, 'slot'], 'Only the GPU slot dispatcher may use this wrapper'
        from research.recurrent_filter_sudoku_v1 import scheduler
        selection()
        original_run = scheduler.subprocess.run
        scheduler.subprocess.run = lambda command, *args, **kwargs: original_run(rewrite_container_command(command), *args, **kwargs)
        print(json.dumps(scheduler.slot(argparse.Namespace())), flush=True)
    elif mode == 'check':
        assert os.environ.get('SLURM_JOB_ID'), 'Scheduler allocation required for checks'
        from transport_reasoning.config import write_json
        from research.recurrent_filter_sudoku_v1.config import root
        result = checks()
        write_json(root() / 'amendments/final_cap_5_checks.json', result)
        print(json.dumps(result), flush=True)
    elif mode == 'run':
        parser = argparse.ArgumentParser()
        parser.add_argument('--seconds', type=int, required=True)
        parser.add_argument('--remaining-seconds', type=float, required=True)
        run(parser.parse_args(sys.argv[2:]))
    else:
        raise ValueError('Unknown runtime mode')


if __name__ == '__main__':
    main()
