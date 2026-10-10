"""Frozen protocol, resumable training order, and separate test/measurement phase."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from transport_reasoning.config import write_json
from .config import BASE_COMMIT, Config, code_hash, grid, load, root


def read(path): return json.loads(Path(path).read_text())


def prepare():
    target = root(); target.mkdir(parents=True, exist_ok=True)
    protocol = dict(series='recurrent_filter_sudoku_v1', base_commit=BASE_COMMIT,
                    source_sha256=code_hash(), code_commit=os.environ.get('RECURRENT_CODE_COMMIT'), schema_version=1,
                    precision='float32; TF32 disabled', batch_sampling='uniform replacement, numpy default_rng(seed+987123)',
                    checkpoint_selection='maximum validation solved, minimum loss, earliest step',
                    evaluation='step 1 and every 250, complete 18000 validation',
                    pilot='12 fresh trainings; 10000 updates, scheduler horizon 30000; no test',
                    final='15 results; 30000 updates; five historical R32 only after compatibility audit',
                    optional_B32='first priority if measured budget fits; decision frozen before any new test',
                    practical_targets={'B16_minus_R16_pp': '>0', 'B16_minus_R32_pp': '>=-1', 'speed_reduction': '>=10% measured separately'},
                    noninferiority='lower 95% paired seed bootstrap bound > -1 percentage point',
                    bootstrap={'seed': 90421, 'samples': 20000, 'unit': 'paired training seed', 'interval': 'percentile, 2.5/97.5'},
                    thresholds='50%,90%; two consecutive validations; retain previous failing and first passing checkpoints; censor failures',
                    mechanism='first16 validation; observational; intermediate readout trained only at final iteration',
                    test_caveat='follow-up on a previously evaluated benchmark, not untouched confirmatory data',
                    failures='retain all seeds; stop dispatcher on unexpected failure',
                    resource_policy='one GPU; old campaign untouched and complete before release; grant required; actual allocated seconds charged',
                    extra_stages='disabled; no automatic family/gate/supervision experiments')
    path = target / 'protocol.json'
    if path.exists() and read(path) != protocol: raise RuntimeError('Frozen protocol differs; use a new series version')
    write_json(path, protocol)
    rows = []
    compatibility_path = target / 'checks/legacy_compatibility.json'
    reusable = compatibility_path.exists() and read(compatibility_path)['status'] == 'passed' and read(compatibility_path)['source_sha256'] == code_hash()
    for c in grid(include_b32=True):
        filename = f'configs/{c.stage}/{c.label()}_seed{c.seed}.json'
        write_json(target / filename, c.dict())
        historical = reusable and c.stage == 'final' and c.label() == 'R32'
        rows.append({'config_file': filename, 'config_hash': c.hash(), 'label': c.label(), 'seed': c.seed,
                     'stage': c.stage, 'status': 'completed' if historical else 'planned',
                     'origin': 'historical_quality_only' if historical else 'new_training',
                     'optional': c.stage == 'final' and c.label() == 'B32'})
    if not (target / 'manifest.json').exists(): write_json(target / 'manifest.json', {'source_sha256': code_hash(), 'rows': rows})
    write_json(target / 'optional_stages.json', {
        'status': 'disabled_pending_separate_validation_decision_and_budget',
        'families': {'T': 'select on validation before test', 'hops': 4, 'mean_hops': 32 / 15,
                     'families': ['exponential', 'geometric'], 'pilot_seeds': [101,103,107], 'final_seeds': [11,23,37,53,71],
                     'new_pilots': 6, 'new_finals': 10, 'gate': 'B16 better than R16 in >=2/3 seeds and mean',
                     'budget_seconds_formula': '1.2*(3*sum(two measured pilot costs)+5*sum(two measured final costs))'},
        'mixture': {'T': 'freeze before test', 'variants': ['local_scalar_shared_gate_w0_b0', 'fixed_g0.5'],
                    'controls': ['onehop', 'binomial'], 'new_pilots': 6, 'new_finals': 10,
                    'budget_seconds_formula': '1.2*(3*sum(two measured pilot costs)+5*sum(two measured final costs))'},
        'supervision': {'T': 'freeze before test', 'variants': ['onehop', 'binomial'], 'steps': ['T/4','T/2','3T/4','T'],
                        'loss_weights': [.25]*4, 'controls': 'matching final_only', 'new_pilots': 6, 'new_finals': 10,
                        'budget_seconds_formula': '1.2*(3*sum(two measured pilot costs)+5*sum(two measured final costs))'},
        'combination_policy': 'no Cartesian product; each extension needs a new implementation version and checks'})
    return protocol


def preflight():
    frozen = read(root() / 'protocol.json')
    if frozen['source_sha256'] != code_hash(): raise RuntimeError('Frozen code hash changed')
    for name in ['pytest', 'data', 'legacy_compatibility', 'overfit']:
        check = read(root() / f'checks/{name}.json')
        if check['status'] != 'passed' or check['source_sha256'] != code_hash():
            raise RuntimeError('Preflight failed: ' + name)


def check_all():
    """Execute checks inside a real CPU allocation and record actual pass/fail."""
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU allocation required')
    import sys
    import xml.etree.ElementTree as ET
    target = root() / 'checks'; target.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, '-m', 'pytest', '--junitxml=' + str(target / 'pytest.xml')])
    xml = ET.parse(target / 'pytest.xml').getroot()
    suites = list(xml.iter('testsuite'))
    write_json(target / 'pytest.json', {'status': 'passed' if result.returncode == 0 else 'failed', 'source_sha256': code_hash(),
                                      'tests': sum(int(s.get('tests', 0)) for s in suites), 'returncode': result.returncode})
    if result.returncode: raise RuntimeError('Tests failed')
    from .audit import audit
    audit()
    from importlib.metadata import distributions
    from transport_reasoning.training import environment
    write_json(target / 'environment.json', environment())
    packages = sorted(set((d.metadata['Name'], d.version) for d in distributions() if d.metadata['Name']))
    (target / 'installed_versions.txt').write_text(''.join(name + '==' + version + '\n' for name, version in packages))
    from transport_reasoning.filters import coefficients, moments
    families = {}
    for name, method in [('exponential','exponential_truncated'),('geometric','finite_geometric')]:
        weights = coefficients(method,4); row = moments(weights)
        row['parameter'] = float(2*weights[1]/weights[0] if name == 'exponential' else weights[1]/weights[0])
        row['mean_matching_error'] = abs(row['mu'] - 32/15); families[name] = row
    write_json(root() / 'optional_filter_coefficients.json', {'status':'prepared_not_queued','hops':4,'target_mean':32/15,'families':families})


def admission(remaining_seconds):
    path = root() / 'admission.json'
    if path.exists(): return read(path)
    estimate = read(root() / 'profile/budget_estimate.json')
    required = estimate['mandatory_seconds_with_20_percent_margin']
    if remaining_seconds < required:
        write_json(root() / 'needs_budget.json', {'status': 'budget_stopped', 'required_remaining_seconds': required,
                   'available_seconds': remaining_seconds, 'additional_seconds_needed': required - remaining_seconds})
        return None
    optional = required + estimate['optional_B32_additional_seconds_with_20_percent_margin'] <= remaining_seconds
    value = {'source_sha256': code_hash(), 'include_final_B32': optional, 'reuse_historical_R32': estimate['reuse_historical_R32_quality'],
             'required_with_margin_seconds': required, 'remaining_at_decision_seconds': remaining_seconds,
             'decision_before_new_test': True, 'new_training_runs': estimate['new_training_runs'] + (5 if optional else 0)}
    if list((root() / 'runs/final').glob('**/evaluation_test.json')): raise RuntimeError('Admission decision must precede test')
    write_json(path, value); return value


def update_row(c, status, **details):
    path = root() / 'manifest.json'; manifest = read(path)
    for row in manifest['rows']:
        if row['config_hash'] == c.hash(): row.update(status=status, **details)
    write_json(path, manifest)


def legacy_folder(seed):
    base = Path(os.environ['RECURRENT_LEGACY_ROOT'])
    candidates = []
    for path in (base / 'runs/sudoku/recurrent').glob(f'*/seed_{seed}/result.json'):
        r = read(path)
        if r['config']['stage'] == 'final' and r['config']['inner_beta'] == 0: candidates.append(path.parent)
    if len(candidates) != 1: raise RuntimeError('Ambiguous historical checkpoint')
    return candidates[0]


def run_slot(seconds, remaining_seconds):
    from .measure import profile, benchmark, diagnose
    from .training import train, evaluate, setup
    from .model import load_checkpoint_model
    from transport_reasoning.data import TaskData
    preflight(); started = time.perf_counter(); deadline = started + seconds
    if not (root() / 'profile/budget_estimate.json').exists():
        profile()
        # Accounting must charge the profiling allocation before admission.
        return 'profile_completed'
    decision = admission(remaining_seconds)
    if decision is None: return 'budget_stopped'
    configs = grid(include_b32=decision['include_final_B32'])
    for c in configs:
        if c.stage == 'final' and c.label() == 'R32' and decision['reuse_historical_R32']:
            update_row(c, 'completed', origin='historical_quality_only'); continue
        if time.perf_counter() >= deadline - 180: return 'paused'
        update_row(c, 'running', origin='new_training')
        checkpoint = c.run_dir() / 'checkpoint_last.pt'
        try:
            result = train(c, resume=checkpoint if checkpoint.exists() else None, deadline=deadline)
        except Exception as error:
            update_row(c, 'failed', error_type=type(error).__name__); raise
        update_row(c, result['status'], completed_steps=result['completed_steps'])
        if result['status'] != 'completed': return 'paused'
    # Frozen decision and all training complete before access to any new test metrics.
    for c in [x for x in configs if x.stage == 'final']:
        if time.perf_counter() >= deadline - 300: return 'paused'
        folder = c.run_dir(); folder.mkdir(parents=True, exist_ok=True)
        historical = c.label() == 'R32' and decision['reuse_historical_R32']
        if not historical and not (folder / 'evaluation_test.json').exists(): evaluate(c)
        if not (folder / 'measurement.json').exists():
            setup(c)
            source = legacy_folder(c.seed) if historical else folder
            model, state = load_checkpoint_model(c, source / 'checkpoint_best.pt'); model.to(c.device)
            data = TaskData(c, 'validation')
            write_json(folder / 'measurement.json', {'source_sha256': code_hash(), 'label': c.label(), 'seed': c.seed,
                       'checkpoint_step': state['step'], 'historical_weights': historical,
                       'inference': benchmark(model, data, c), 'mechanism': diagnose(model, data, c)})
    from .reporting import report
    report()
    write_json(root() / 'completion.json', {'status': 'completed', 'source_sha256': code_hash(), 'decision': decision})
    return 'completed'
