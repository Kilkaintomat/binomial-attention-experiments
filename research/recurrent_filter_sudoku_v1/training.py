import copy
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.config import write_json
from transport_reasoning.data import TaskData
from transport_reasoning.evaluation import evaluate_model, loss_for, sync
from transport_reasoning.training import atomic_checkpoint, restore_rng, environment
from .config import code_hash
from .model import Encoder


def setup(c):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training and compute checks require a scheduler allocation')
    torch.set_num_threads(c.threads)
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    if c.device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one allocated GPU is required')
        if not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'):
            raise RuntimeError('No scheduler GPU assignment')
        torch.cuda.manual_seed_all(c.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device(c.device)


def optimization(model, c):
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    warmup = max(1, round(c.planned_steps * c.warmup_fraction))
    def schedule(step):
        return (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, c.planned_steps - warmup))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def atomic_copy(source, destination):
    temp = destination.with_suffix('.copy.tmp')
    shutil.copyfile(source, temp)
    temp.replace(destination)


def train(c, resume=None, stop_after=None, deadline=None):
    c.validate(); device = setup(c)
    train_data = TaskData(c, 'train'); validation = TaskData(c, 'validation')
    if c.overfit_count:
        train_data.subset(c.overfit_count)
        validation = TaskData(c, 'train').subset(c.overfit_count)
    directory = c.run_dir(); directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    source = code_hash()
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['status'] == 'completed':
            if old['config'] != c.dict() or old['source_sha256'] != source:
                raise ValueError('Completed run has different configuration or source')
            return old
    if (directory / 'checkpoint_last.pt').exists() and resume is None:
        raise ValueError('Existing run requires explicit resume')
    env = environment(); env.update(study_source_sha256=source, schema_version=1)
    write_json(directory / 'environment.json', env); write_json(directory / 'config.json', c.dict())
    model = Encoder(c).to(device); opt, scheduler = optimization(model, c)
    generator = np.random.default_rng(c.seed + 987123)
    step = 0; best = (-1., -float('inf'))
    timings = dict(training=0., validation=0., checkpoints=0., elapsed=0.)
    thresholds = {str(p): {'status': 'not_reached', 'last_failing_step': None} for p in [.5, .9]}
    last_metric = None
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['config'] != c.dict() or state['source_sha256'] != source or state['data_hash'] != train_data.manifest['data_hash']:
            raise ValueError('Resume configuration, source or data mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        for values in opt.state.values():
            for key, value in values.items():
                if torch.is_tensor(value): values[key] = value.to(device)
        restore_rng(state, generator)
        step = state['step']; best = tuple(state['best']); timings = state['timings']
        thresholds = state['thresholds']; last_metric = state['last_metric']
        path = directory / 'metrics.jsonl'
        if path.exists():
            kept = [s for s in path.read_text().splitlines() if json.loads(s)['step'] <= step]
            path.write_text('\n'.join(kept) + ('\n' if kept else ''))
    started = time.perf_counter(); previous_elapsed = timings['elapsed']
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

    def save(improved=False):
        tick = time.perf_counter()
        timings['elapsed'] = previous_elapsed + time.perf_counter() - started
        state = dict(config=c.dict(), source_sha256=source, data_hash=train_data.manifest['data_hash'],
                     model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                     step=step, best=list(best), timings=copy.deepcopy(timings), thresholds=copy.deepcopy(thresholds), last_metric=last_metric,
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                     data_rng=generator.bit_generator.state, filters=model.filter_summary())
        atomic_checkpoint(directory / 'checkpoint_last.pt', state)
        if improved: atomic_copy(directory / 'checkpoint_last.pt', directory / 'checkpoint_best.pt')
        timings['checkpoints'] += time.perf_counter() - tick

    target = min(c.completion_steps, stop_after) if stop_after is not None else c.completion_steps
    status = 'running'
    while step < target:
        if deadline is not None and time.perf_counter() >= deadline - 120:
            status = 'budget_stopped'; save(); break
        sync(device); tick = time.perf_counter()
        indices = generator.integers(0, len(train_data), size=c.batch_size)
        batch = train_data.batch(indices, device)
        model.train(); opt.zero_grad(set_to_none=True)
        loss = loss_for(model(batch), batch, c)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip, error_if_nonfinite=True)
        opt.step(); scheduler.step(); step += 1
        sync(device); timings['training'] += time.perf_counter() - tick
        if step == 1 or step % c.eval_every == 0 or step == c.completion_steps:
            tick = time.perf_counter()
            metric = evaluate_model(model, validation, c, device)
            timings['validation'] += time.perf_counter() - tick
            metric.update(step=step, planned_steps=c.planned_steps, completion_steps=c.completion_steps,
                          train_loss=float(loss), gradient_norm=float(norm), lr=opt.param_groups[0]['lr'],
                          training_seconds=timings['training'], elapsed_seconds=previous_elapsed + time.perf_counter() - started,
                          train_peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                          filters=model.filter_summary())
            rank = (metric['solved_rate'], -metric['loss']); improved = rank > best
            if improved: best = rank
            archives = []
            for key, entry in thresholds.items():
                if entry['status'] == 'reached': continue
                if metric['solved_rate'] >= float(key):
                    if entry['status'] == 'pending':
                        entry.update(status='reached', confirmed_step=step)
                    else:
                        entry.update(status='pending', first_passing_step=step,
                                     training_seconds_upper=timings['training'])
                        archives.append(key)
                else:
                    entry.clear(); entry.update(status='not_reached', last_failing_step=step, training_seconds_lower=timings['training'])
            last_metric = metric
            with (directory / 'metrics.jsonl').open('a') as f: f.write(json.dumps(metric, allow_nan=False) + '\n')
            save(improved)
            for key in archives:
                prefix = directory / ('threshold_' + str(round(float(key) * 100)))
                atomic_copy(directory / 'checkpoint_last.pt', prefix.with_name(prefix.name + '_first_passing.pt'))
                previous = directory / 'checkpoint_last_evaluated.pt'
                if previous.exists(): atomic_copy(previous, prefix.with_name(prefix.name + '_previous_failing.pt'))
            atomic_copy(directory / 'checkpoint_last.pt', directory / 'checkpoint_last_evaluated.pt')
    if step == c.completion_steps: status = 'completed'
    elif status == 'running': status = 'paused'; save()
    timings['elapsed'] = previous_elapsed + time.perf_counter() - started
    result = dict(status=status, config=c.dict(), config_hash=c.hash(), source_sha256=source,
                  data_hash=train_data.manifest['data_hash'], planned_steps=c.planned_steps, completed_steps=step,
                  examples=step * c.batch_size, parameter_count=model.parameter_count(), operations=c.operations(),
                  timings=timings, last_validation=last_metric, thresholds=thresholds,
                  split='train_subset' if c.overfit_count else 'validation', best_validation_solved_rate=best[0])
    write_json(result_path, result)
    return result


def evaluate(c, split='test'):
    if split == 'test' and (c.stage != 'final' or c.completion_steps != 30000):
        raise ValueError('Pilot and partial training must not access test')
    r = json.loads((c.run_dir() / 'result.json').read_text())
    if r['status'] != 'completed': raise ValueError('Training is unfinished')
    device = setup(c)
    state = torch.load(c.run_dir() / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
    if state['config'] != c.dict() or state['source_sha256'] != code_hash(): raise ValueError('Checkpoint mismatch')
    model = Encoder(c).to(device); model.load_state_dict(state['model']); data = TaskData(c, split)
    if state['data_hash'] != data.manifest['data_hash']: raise ValueError('Data mismatch')
    metric = evaluate_model(model, data, c, device, c.run_dir() / f'predictions_{split}.npz')
    metric.update(split=split, checkpoint_step=state['step'], source_sha256=code_hash(), data_hash=state['data_hash'])
    write_json(c.run_dir() / f'evaluation_{split}.json', metric)
    return metric
