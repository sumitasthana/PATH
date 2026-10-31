"""Fixed paired study queue, verified R2 progress, and descriptive summaries."""
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from . import checkpoint
from .config import Config
from .hypernet import HyperNetwork, build_target
from .learning_diagnostic import _validate_source, _write_json, file_hash
from .paired_diagnostic import _frozen_hash, _hash_named, _restore_training_rng, tensor_hash
from .storage import _key
from .study_training import capture_rng, cpu_copy, restore_rng, run_request, score
from .telemetry import environment, progress_bar
from .trainer import UnCLe

FORMAT = 1
ORDERS = {
    'order_01': ['3', '0', '9', '5', '17', '1', '7', '14'],
    'order_02': ['14', '0', '9', '3', '17', '1', '7', '5'],
    'order_03': ['5', '0', '9', '14', '17', '1', '7', '3'],
}


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def identifier(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True).encode()).hexdigest()


def make_plan(code_version, partition_path, study_id='paired_generalization_v1'):
    """27 comparisons reuse nine controls: 9 source jobs and 36 branch jobs."""
    _key(study_id)
    if '/' in study_id or not code_version:
        raise ValueError('Use one study name and a recorded code revision.')
    config = Config(dataset='tiny_imagenet', tasks=tuple(str(i) for i in range(20)),
        requests=(), backbone='resnet50', initialization='hyperfan_in', device='cuda',
        epochs=5, batch_size=64, eval_batch_size=256, learning_rate=0.0001,
        forgetting_learning_rate=0.0001, beta=0.1, gamma=0.01,
        noise_samples=10, burn_in=100)
    plan = {'version': FORMAT, 'study_id': study_id, 'code_version': code_version,
        'partition_sha256': file_hash(partition_path), 'partition_seed': 42,
        'config': vars(config), 'orders': ORDERS, 'seeds': [0, 1, 2],
        'forget_tasks': ['3', '5', '14'], 'new_tasks': ['15'],
        'forget_checkpoint_interval': 20, 'forget_evaluation_interval': 20,
        'keep_post_unlearning_checkpoint': False,
        'keep_branch_checkpoints': False,
        'forgetting_tolerance_pp': 2.0,
        'data_policy': '500 training images per class; validation evaluation; no recovery probe',
        'scope': 'Diagnostic generalization of experiment 06, not paper sequence reproduction',
        'resume_policy': 'Epoch boundaries for learning, periodic steps for unlearning; same code and runtime signature',
        'writer_policy': 'One active writer per study; sequential jobs',
        'aggregation': 'Paired differences; seed means per setting; equal setting weight; no independent-task inference'}
    plan = normalized(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan['version'] != FORMAT:
        raise ValueError('Unknown study format.')
    _key(plan['study_id'])
    if '/' in plan['study_id'] or not plan['code_version']:
        raise ValueError('Invalid study identity.')
    if not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']):
        raise ValueError('Seeds must be nonempty and distinct.')
    if any(type(seed) is not int or seed < 0 for seed in plan['seeds']):
        raise ValueError('Seeds must be nonnegative integers.')
    for field in ('forget_tasks', 'new_tasks'):
        if not plan[field] or len(set(plan[field])) != len(plan[field]):
            raise ValueError('Task selections must be nonempty and distinct.')
    if not plan['orders']:
        raise ValueError('At least one order is required.')
    cohorts = []
    for name, order in plan['orders'].items():
        _key(name)
        if '/' in name or len(set(order)) != len(order):
            raise ValueError('Orders must have unique tasks and simple names.')
        if not set(plan['forget_tasks']).issubset(order) or set(plan['new_tasks']).intersection(order):
            raise ValueError('Forget targets must be learned; incoming tasks must be unseen.')
        cohorts.append(set(order))
    if any(cohort != cohorts[0] for cohort in cohorts):
        raise ValueError('Every order must use the same task cohort.')
    config = Config(**plan['config'])
    if not set.union(*cohorts, set(plan['new_tasks'])).issubset(config.tasks):
        raise ValueError('Study uses unknown tasks.')
    for name in ('forget_checkpoint_interval', 'forget_evaluation_interval'):
        if type(plan[name]) is not int or plan[name] < 1:
            raise ValueError('Checkpoint and evaluation intervals must be positive integers.')
    if not math.isfinite(plan['forgetting_tolerance_pp']) or plan['forgetting_tolerance_pp'] < 0:
        raise ValueError('Forgetting tolerance must be finite and nonnegative.')
    for name in ('keep_post_unlearning_checkpoint', 'keep_branch_checkpoints'):
        if name in plan and type(plan[name]) is not bool:
            raise ValueError('Checkpoint retention settings must be booleans.')


def jobs(plan):
    validate_plan(plan)
    sources, branches = [], []
    for order_name, order in plan['orders'].items():
        for seed in plan['seeds']:
            base = f'{order_name}/seed_{seed}'
            source_id = 'sources/' + base
            sources.append({'id': source_id, 'kind': 'source', 'order': order_name,
                'seed': seed, 'source': None, 'operations': [['learn', task] for task in order]})
            for new_task in plan['new_tasks']:
                common = {'order': order_name, 'seed': seed, 'source': source_id, 'new_task': new_task}
                branches.append({**common, 'id': f'controls/{base}/learn_{new_task}',
                    'kind': 'control', 'operations': [['learn', new_task]]})
                for target in plan['forget_tasks']:
                    branches.append({**common, 'id': f'branches/{base}/forget_{target}/learn_{new_task}',
                        'kind': 'unlearned', 'forget_task': target,
                        'operations': [['forget', target], ['learn', new_task]]})
    # Finish a source and its branches before moving to another source.
    return [job for source in sources for job in [source, *[b for b in branches if b['source'] == source['id']]]]


class StudyStore:
    """Two temporary slots per unfinished job; publish the progress pointer last."""
    def __init__(self, store, plan, root, *, cleanup_completed=True):
        validate_plan(plan)
        self.store, self.plan = store, normalized(plan)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = 'uncle/paired_generalization/' + plan['study_id']
        self.cleanup_completed = cleanup_completed

    def read(self, key):
        if self.store.info(key) is None:
            return None
        return json.loads(self.store.download(key, self.root / 'cache').read_text(encoding='utf-8'))

    def prepare(self):
        key = self.prefix + '/study.json'
        existing = self.read(key)
        if existing is not None and existing != self.plan:
            raise ValueError('Study settings or code differ. Restore them or choose a new study ID.')
        path = self.root / 'study.json'
        _write_json(path, self.plan)
        self.store.upload(path, key)

    def progress(self, job):
        if job not in jobs(self.plan):
            raise ValueError('Job is outside this study queue.')
        value = self.read(self.prefix + '/' + job['id'] + '/progress.json')
        if value is not None and value['contract_sha256'] != identifier({'plan': self.plan, 'job': job}):
            raise ValueError('Saved progress belongs to a different job or study.')
        return value

    def validate_artifact(self, artifact):
        info = self.store.info(artifact['key'])
        if (info is None or info['ContentLength'] != artifact['bytes']
                or info.get('Metadata', {}).get('sha256') != artifact['sha256']):
            raise ValueError('Missing or changed R2 artifact: ' + artifact['key'])

    def validate_progress_artifacts(self, job, progress):
        self.validate_artifact(progress['report'])
        if progress.get('checkpoint_retained', True):
            self.validate_artifact(progress['checkpoint'])
        elif (progress['status']!='complete' or job['kind']=='source'
              or self.plan.get('keep_branch_checkpoints', True)):
            raise ValueError('Required checkpoint cannot be discarded.')
        elif not progress.get('checkpoint_verified_before_discard'):
            raise ValueError('Discarded checkpoint lacks its verification receipt.')

    def load(self, job):
        progress = self.progress(job)
        if progress is None:
            return None
        self.validate_progress_artifacts(job, progress)
        if not progress.get('checkpoint_retained', True):
            raise ValueError('This completed branch model was discarded after verification. Its report remains available.')
        path = self.store.download(progress['checkpoint']['key'], self.root / 'cache')
        return torch.load(path, map_location='cpu', weights_only=False)

    def completed_report(self, job):
        progress = self.progress(job)
        if progress is None or progress['status'] != 'complete':
            return None
        self.validate_progress_artifacts(job, progress)
        report = self.read(progress['report']['key'])
        if report['status'] != 'complete' or report['job'] != job:
            raise ValueError('Invalid completed job report.')
        if report.get('post_unlearning_checkpoint'):
            self.validate_artifact(report['post_unlearning_checkpoint'])
        return report

    def publish(self, job, payload, report, *, complete=False):
        current = self.progress(job)
        generation = 0 if current is None else current['generation'] + 1
        if current is not None and current['status'] == 'complete':
            raise ValueError('A completed job cannot be overwritten.')
        slot = generation % 2
        base = self.prefix + '/' + job['id']
        directory = self.root / job['id']
        directory.mkdir(parents=True, exist_ok=True)
        post_unlearning = (not complete and self.plan['keep_post_unlearning_checkpoint']
            and job['kind']=='unlearned' and payload['job_requests_done']==1
            and payload['active_request'] is None)
        model_name = ('checkpoint.pt' if complete else 'after_unlearning.pt'
                      if post_unlearning else f'resume_{slot}.pt')
        report_name = ('report.json' if complete else 'after_unlearning.json'
                       if post_unlearning else f'resume_{slot}.json')
        path = directory / model_name
        temporary = path.with_suffix('.pt.writing')
        torch.save(payload, temporary)
        temporary.replace(path)
        report_path = directory / report_name
        _write_json(report_path, report)
        started = time.monotonic()
        model = self.store.upload(path, base + '/' + model_name, overwrite=True)
        measured = self.store.upload(report_path, base + '/' + report_name, overwrite=True)
        if self.progress(job) != current:
            raise ValueError('Another writer changed this job. Use one session per study.')
        pointer = {'version': FORMAT, 'contract_sha256': identifier({'plan': self.plan, 'job': job}),
            'generation': generation, 'status': 'complete' if complete else 'resumable',
            'checkpoint': model, 'report': measured, 'completed_requests': len(payload['history']),
            'active': ({key:payload['active_request'][key] for key in ('action','task','epoch','step')}
                       if payload['active_request'] else None),
            'upload_verify_seconds': time.monotonic()-started,
            'updated_at': datetime.now(timezone.utc).isoformat()}
        pointer['checkpoint_retained'] = not (complete and job['kind']!='source'
            and not self.plan.get('keep_branch_checkpoints', True))
        if not pointer['checkpoint_retained']:
            pointer['checkpoint_verified_before_discard'] = True
        pointer_path = directory / 'progress.json'
        _write_json(pointer_path, pointer)
        self.store.upload(pointer_path, base + '/progress.json', overwrite=True)
        # Reading the remote pointer checks that publication itself is durable.
        if self.progress(job) != pointer:
            raise ValueError('R2 progress publication was not verified.')
        if complete and self.cleanup_completed:
            self.cleanup(job)
        return pointer

    def cleanup(self, job):
        """Remove completed-job model files only after a durable verification receipt."""
        progress = self.progress(job)
        if progress is None or progress['status'] != 'complete':
            raise ValueError('Temporary slots can be cleaned only after completion.')
        self.validate_progress_artifacts(job, progress)
        base = self.prefix + '/' + job['id']
        removed = []
        keys = [base+f'/resume_{slot}.{suffix}' for slot in (0,1) for suffix in ('pt','json')]
        if not progress.get('checkpoint_retained', True):
            if progress['checkpoint']['key']!=base+'/checkpoint.pt':
                raise ValueError('Unexpected final checkpoint location.')
            keys.append(base+'/checkpoint.pt')
        for key in keys:
            info = self.store.info(key)
            if info is not None:
                if key==base+'/checkpoint.pt':
                    self.validate_artifact(progress['checkpoint'])
                condition = {'IfMatch':info['ETag']} if info.get('ETag') else {}
                self.store.client.delete_object(Bucket=self.store.bucket, Key=key, **condition)
                if self.store.info(key) is not None:
                    raise ValueError('Completed-job model cleanup failed.')
                removed.append(key)
        directory = self.root / job['id']
        for path in directory.glob('resume_*.*'):
            path.unlink()
        if not progress.get('checkpoint_retained', True):
            (directory/'checkpoint.pt').unlink(missing_ok=True)
        if removed:
            path = directory / 'resume_cleanup.json'
            _write_json(path, {'job':job['id'], 'removed':removed,
                'completed_checkpoint':progress['checkpoint'],
                'checkpoint_retained':progress.get('checkpoint_retained',True),
                'historical_files_touched':False})
            self.store.upload(path, base + '/resume_cleanup.json', overwrite=True)


def runtime_signature(config):
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'device_type': config.torch_device.type,
        'gpu': torch.cuda.get_device_name(0) if config.torch_device.type == 'cuda' else None,
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_precision': torch.get_float32_matmul_precision()}


def run_job(plan, job, tasks, storage, *, device='cuda', progress=True,
            stop_requested=lambda: False, on_boundary=None, observer=None):
    """Run one job. A stop returns only after its resume state is verified in R2."""
    complete = storage.completed_report(job)
    if complete is not None:
        return complete
    requests = [('learn', task) for task in plan['orders'][job['order']]]
    if job['kind'] != 'source':
        requests += [tuple(op) for op in job['operations']]
    config = replace(Config(**plan['config']), seed=job['seed'], device=device,
                     requests=tuple(requests))
    source = storage.load(next(j for j in jobs(plan) if j['id'] == job['source'])) if job['source'] else None
    if source is not None:
        if source['report']['status'] != 'complete':
            raise ValueError('Source model is not complete.')
        _validate_source(source)
    resumed = storage.load(job)
    state = resumed if resumed is not None else source
    if resumed is not None and resumed['runtime_signature'] != runtime_signature(config):
        raise ValueError('Resume runtime differs. Restore its GPU/software settings or use a new study.')
    needed = set(plan['orders'][job['order']]) | ({job['new_task']} if job['source'] else set())
    if not needed.issubset(tasks):
        raise ValueError('Task datasets are missing.')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.torch_device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    target = build_target(config)
    h = HyperNetwork(target, config)
    trainer = UnCLe(h, config, target, tasks, progress=progress)
    if state is not None:
        checkpoint.restore(state, hypernet=h, uncle=trainer)
        restore_rng(state['all_rng'])
    history = cpu_copy(state['history']) if state else []
    seen = list(state['seen']) if state else []
    forgotten = list(state['forgotten']) if state else []
    previous = dict(state['previous']) if state else {}
    report = cpu_copy(resumed['report']) if resumed else {
        'status': 'running', 'job': job, 'plan_sha256': identifier(plan),
        'code_version': plan['code_version'], 'environment': environment(config, h, tasks),
        'starting_accuracies': dict(previous), 'requests': [], 'history': history,
        'compute_seconds': 0.0, 'boundary_count': 0,
        'source_sha256': (storage.progress(next(j for j in jobs(plan) if j['id']==job['source']))['checkpoint']['sha256']
                          if job['source'] else None)}
    active = resumed['active_request'] if resumed else None
    done = resumed['job_requests_done'] if resumed else 0
    if (resumed and job['kind']=='unlearned' and done>=1
            and plan['keep_post_unlearning_checkpoint'] and not report.get('post_unlearning_checkpoint')):
        key = storage.prefix + '/' + job['id'] + '/after_unlearning.pt'
        info = storage.store.info(key)
        if info is None:
            raise ValueError('Post-unlearning checkpoint is missing.')
        report['post_unlearning_checkpoint'] = {'key':key, 'bytes':info['ContentLength'],
                                                'sha256':info['Metadata']['sha256']}
    base_seen = list(source['seen']) if source else []
    if job['source'] and resumed is None:
        actual = score(trainer, base_seen)
        if actual != previous:
            raise ValueError('Restored source scores differ. Stop before training.')
        report['initial_model_sha256'] = _hash_named(list(h.state_dict().items()))
        report['frozen_before_sha256'] = _frozen_hash(h, trainer, base_seen)

    segment_started = time.monotonic()
    stopped = False

    def save(active_state, *, complete=False):
        nonlocal segment_started, stopped
        report['compute_seconds'] += time.monotonic() - segment_started
        report['boundary_count'] += 1
        report['status'] = 'complete' if complete else 'resumable'
        rng = capture_rng()
        payload = {'format': checkpoint.FORMAT, 'config': vars(config),
            'tasks_with_codes': list(h.task_codes), 'hypernet': cpu_copy(h.state_dict()),
            'task_buffers': cpu_copy(trainer.task_buffers), 'history': cpu_copy(history),
            'seen': list(seen), 'forgotten': list(forgotten), 'previous': dict(previous),
            'costs': [], 'setup_seconds': 0.0, 'rng': rng['torch'], 'cuda_rng': rng['cuda'],
            'all_rng': rng, 'runtime_signature': runtime_signature(config),
            'active_request': active_state, 'job_requests_done': done, 'report': cpu_copy(report)}
        pointer = storage.publish(job, payload, report, complete=complete)
        if pointer['checkpoint']['key'].endswith('/after_unlearning.pt'):
            report['post_unlearning_checkpoint'] = pointer['checkpoint']
        restore_rng(rng)
        if on_boundary:
            on_boundary(job, pointer)
            restore_rng(rng)
        if observer is not None:
            # Numbered by tasks learned, so epoch saves land on a number that
            # is already written and cost nothing.
            observer.boundary(trainer, len(seen), seen[-1] if seen else None)
            restore_rng(rng)
        segment_started = time.monotonic()
        stopped = not complete and stop_requested()
        if stopped:
            raise _SafeStop()

    try:
        for index in range(done, len(job['operations'])):
            action, task = job['operations'][index]
            if active is None:
                if job['source'] and action == 'learn':
                    _restore_training_rng(source, config.torch_device)
                    report['learning_rng_sha256'] = tensor_hash(torch.get_rng_state())
                    report['learning_cuda_rng_sha256'] = ([tensor_hash(v) for v in torch.cuda.get_rng_state_all()]
                        if config.torch_device.type == 'cuda' else [])
                report['requests'].append({'action': action, 'task': task,
                    'before': dict(previous), 'trace': [], 'epochs': []})
            request = report['requests'][-1]
            protected = [t for t in seen if action == 'learn' or t != task]

            def record(row):
                if row.get('boundary'):
                    request['epochs'].append(row)
                else:
                    request['trace'].append(row)
                    if row['step'] == 0 and action == 'learn':
                        report['new_task_code_sha256'] = row['code_sha256']

            def boundary(active_state):
                save(active_state)

            active = run_request(trainer, action, task, protected, active, record, boundary,
                forget_interval=plan['forget_checkpoint_interval'],
                evaluation_interval=plan['forget_evaluation_interval'],
                on_step=None if observer is None else observer.step)
            after = score(trainer, list(h.task_codes))
            if action == 'learn':
                seen.append(task)
            else:
                forgotten.append(task)
                report['after_unlearning_accuracies'] = after
            request['after'] = after
            history.append({'index':len(history), 'action':action, 'task':task,
                'before':dict(previous), 'after':after, 'seen':list(seen),
                'forgotten':list(forgotten), 'burn_in':config.burn_in if action=='forget' else None,
                'final_loss':(active['epoch_losses'][-1] if action=='learn' else request['trace'][-1]['loss'])})
            previous = after
            done = index + 1
            active = None
            report['history'] = history
            # Preserve a lightweight checkpoint immediately after each request.
            save(None)
        report['final_accuracies'] = previous
        if job['source']:
            report['frozen_after_sha256'] = _frozen_hash(h, trainer, base_seen)
            if report['frozen_after_sha256'] != report['frozen_before_sha256']:
                raise ValueError('Old codes, chunk codes, or running statistics changed.')
        save(None, complete=True)
        return report
    except _SafeStop:
        return {'status':'resumable', 'job':job, 'completed_requests':done}
    finally:
        del trainer, h, target, state, resumed, source, active
        gc.collect()
        if config.torch_device.type == 'cuda':
            torch.cuda.empty_cache()


class _SafeStop(Exception):
    pass


def inspect_queue(plan, storage, *, progress=True):
    rows = []
    queue = jobs(plan)
    bar = progress_bar(len(queue), 'Inspect study queue', progress, leave=False)
    try:
        for job in queue:
            pointer = storage.progress(job)
            status = 'pending' if pointer is None else pointer['status']
            if pointer:
                storage.validate_progress_artifacts(job, pointer)
            rows.append({**job, 'status':status, 'active':pointer['active'] if pointer else None})
            bar.update(1)
    finally:
        bar.close()
    return rows


def run_queue(plan, storage, tasks_factory, *, max_jobs=1, max_minutes=120,
              reserve_minutes=15, device='cuda', progress=True):
    """Soft session budget; stop at a boundary and never report an unsaved stop."""
    if type(max_jobs) is not int or max_jobs < 1 or not 0 < reserve_minutes < max_minutes:
        raise ValueError('Use a positive job limit and a reserve smaller than the session budget.')
    began = time.monotonic()
    deadline = began + (max_minutes-reserve_minutes)*60
    storage.prepare()
    rows = inspect_queue(plan, storage, progress=progress)
    completed = {row['id'] for row in rows if row['status']=='complete'}
    attempted = []
    current_job = None
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'session_id':session_id, 'status':'running', 'attempted':attempted,
        'max_jobs':max_jobs, 'max_minutes':max_minutes, 'reserve_minutes':reserve_minutes,
        'plan_sha256':identifier(plan)}
    bar = progress_bar(len(rows), 'Study work units', progress)
    bar.update(len(completed))
    try:
        for row in rows:
            if row['status']=='complete':
                if storage.cleanup_completed:
                    storage.cleanup(next(j for j in jobs(plan) if j['id']==row['id']))
                continue
            if len(attempted) >= max_jobs or time.monotonic() >= deadline:
                break
            if row['source'] and row['source'] not in completed:
                continue
            job = next(j for j in jobs(plan) if j['id']==row['id'])
            current_job = job['id']
            tasks = tasks_factory(job)
            if time.monotonic() >= deadline:
                del tasks
                break
            result = run_job(plan, job, tasks, storage, device=device, progress=progress,
                             stop_requested=lambda:time.monotonic() >= deadline)
            attempted.append({'job':job['id'], 'status':result['status']})
            del tasks
            gc.collect()
            if result['status']=='complete':
                completed.add(job['id'])
                bar.update(1)
            else:
                break
        session.update(status='saved', completed_jobs=len(completed),
            total_jobs=len(rows), session_seconds=time.monotonic()-began,
            budget_is_soft=True, note='Boundary duration and network retries can exceed the configured budget.')
        return session
    except BaseException as error:
        session.update(status='interrupted', error_type=type(error).__name__,
                       current_job=current_job, session_seconds=time.monotonic()-began)
        raise
    finally:
        bar.close()
        path = storage.root / 'sessions' / (session_id + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, session)
        try:
            storage.store.upload(path, storage.prefix + '/sessions/' + path.name)
        except Exception:
            print('Session log saved locally; R2 session-log upload failed. Resume from the verified job pointer.')


def compare_reports(plan, control, unlearned):
    """Compare a shared control and one deletion branch; reject broken pairing."""
    if control['status'] != 'complete' or unlearned['status'] != 'complete':
        raise ValueError('Only complete branch reports can be paired.')
    a, b = control, unlearned
    job = b['job']
    if a['job']['kind'] != 'control' or job['kind'] != 'unlearned':
        raise ValueError('Incorrect comparison branches.')
    target, new = job['forget_task'], job['new_task']
    fields = ('source_sha256', 'initial_model_sha256', 'starting_accuracies',
              'learning_rng_sha256', 'learning_cuda_rng_sha256', 'new_task_code_sha256')
    checks = {field:a[field]==b[field] for field in fields}
    checks['source_identity'] = all(a['job'][field]==job[field] for field in ('source','order','seed','new_task'))
    checks['plan'] = a['plan_sha256']==b['plan_sha256']==identifier(plan)
    checks['frozen_state'] = all(r['frozen_before_sha256']==r['frozen_after_sha256'] for r in (a,b))
    ta, tb = a['requests'][-1]['trace'][1:], b['requests'][-1]['trace'][1:]
    checks['batch_order'] = [r['indices'] for r in ta]==[r['indices'] for r in tb]
    checks['input_values'] = [r['input_sha256'] for r in ta]==[r['input_sha256'] for r in tb]
    expected_updates = (math.ceil(a['environment']['task_sizes'][new]['train']/plan['config']['batch_size'])
                        * plan['config']['epochs'])
    checks['update_count'] = (len(ta)==len(tb)==expected_updates
        and len(a['requests'][-1]['epochs'])==len(b['requests'][-1]['epochs'])==plan['config']['epochs'])
    retained = [t for t in a['starting_accuracies'] if t != target]
    differences = {t:b['final_accuracies'][t]-a['final_accuracies'][t] for t in retained}
    chance = 100/plan['config']['classes_per_task']
    tolerance = plan['forgetting_tolerance_pp']
    post = b['after_unlearning_accuracies'][target]
    later = [r['accuracies'][target] for r in b['requests'][-1]['epochs']]
    measured = [r['accuracies'] for r in b['requests'][0]['trace'] if 'accuracies' in r]
    start = a['starting_accuracies']
    return {'status':'complete' if all(checks.values()) else 'invalid_pair',
        'order':job['order'], 'seed':job['seed'], 'forget_task':target, 'new_task':new,
        'control_id':a['job']['id'], 'branch_id':job['id'], 'pair_checks':checks,
        'starting_target_accuracy':start[target], 'after_unlearning_accuracy':post,
        'final_target_accuracy':b['final_accuracies'][target], 'chance_accuracy':chance,
        'target_start_above_chance':start[target]>chance,
        'forgetting_within_tolerance':post<=chance+tolerance,
        'relapse_above_tolerance':post<=chance+tolerance and any(value>chance+tolerance for value in later),
        'new_task_a':a['final_accuracies'][new], 'new_task_b':b['final_accuracies'][new],
        'new_task_difference':b['final_accuracies'][new]-a['final_accuracies'][new],
        'retained_differences':differences,
        'retained_mean_difference':statistics.mean(differences.values()),
        'largest_paired_retained_drop':max([0.0,*[-v for v in differences.values()]]),
        'largest_temporary_retained_drop':max([0.0,*[start[t]-r[t] for r in measured for t in retained]]),
        'control_compute_seconds':a['compute_seconds'], 'branch_compute_seconds':b['compute_seconds']}


def summarize(plan, comparisons):
    """Seed variation and equal-setting averages; no task-level pseudo replication."""
    valid = [r for r in comparisons if r['status']=='complete']
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row['order'], row['forget_task'], row['new_task'])].append(row)
    metrics = ('new_task_difference','retained_mean_difference','after_unlearning_accuracy',
               'final_target_accuracy','largest_paired_retained_drop','largest_temporary_retained_drop')
    settings = []
    for (order,target,new), rows in sorted(grouped.items()):
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('Duplicate seed in a setting.')
        record = {'order':order, 'forget_task':target, 'new_task':new,
            'seeds':[r['seed'] for r in rows], 'n_seeds':len(rows),
            'complete_setting':set(r['seed'] for r in rows)==set(plan['seeds'])}
        for metric in metrics:
            values = [r[metric] for r in rows]
            record[metric] = {'mean':statistics.mean(values),
                'sd':statistics.stdev(values) if len(values)>1 else None, 'values':values}
        settings.append(record)
    full = [r for r in settings if r['complete_setting']]
    expected = len(plan['orders'])*len(plan['seeds'])*len(plan['forget_tasks'])*len(plan['new_tasks'])
    return {'status':'complete' if len(valid)==expected else 'partial',
        'expected_pairs':expected, 'valid_pairs':len(valid),
        'invalid_pairs':[r for r in comparisons if r['status']!='complete'],
        'settings':settings, 'complete_settings':len(full),
        'equal_setting_means':{metric:statistics.mean(r[metric]['mean'] for r in full)
                              for metric in metrics} if full else {},
        'new_task_negative_pairs':sum(r['new_task_difference']<0 for r in valid),
        'forgetting_within_tolerance_pairs':sum(r['forgetting_within_tolerance'] for r in valid),
        'relapse_pairs':sum(r['relapse_above_tolerance'] for r in valid),
        'limitation':'Shared controls and tasks are correlated; three seeds offer limited precision. No erasure or recovery claim.'}


def review(plan, storage, *, progress=True):
    """Download reports only. Model objects are checked by metadata, not downloaded."""
    queue = jobs(plan)
    reports, missing = {}, []
    bar = progress_bar(len(queue), 'Read study reports', progress, leave=False)
    try:
        for job in queue:
            report = storage.completed_report(job)
            if report:
                reports[job['id']] = report
            else:
                pointer = storage.progress(job)
                missing.append({'job':job['id'], 'status':pointer['status'] if pointer else 'pending'})
            bar.update(1)
    finally:
        bar.close()
    comparisons = []
    for job in queue:
        if job['kind'] != 'unlearned' or job['id'] not in reports:
            continue
        control = f"controls/{job['order']}/seed_{job['seed']}/learn_{job['new_task']}"
        if control in reports:
            comparisons.append(compare_reports(plan,reports[control],reports[job['id']]))
    result = summarize(plan, comparisons)
    result.update(comparisons=comparisons, unfinished_jobs=missing, plan_sha256=identifier(plan))
    session_keys = list(storage.store.keys(storage.prefix + '/sessions/'))
    result['session_records'] = []
    bar = progress_bar(len(session_keys), 'Read session records', progress, leave=False)
    try:
        for key in session_keys:
            if key.endswith('.json'):
                result['session_records'].append(storage.read(key))
            bar.update(1)
    finally:
        bar.close()
    return result
