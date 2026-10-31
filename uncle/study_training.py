"""Request training that can resume at epoch and unlearning boundaries."""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from .config import random_stream
from .paired_diagnostic import generated_noise_loss, _hash_named
from .telemetry import progress_bar
from .trainer import _IndexedDataset


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def capture_rng():
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs.')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def score(trainer, tasks):
    """Measure validation with task and batch bars, without consuming train RNG."""
    state = capture_rng()
    modes = [(module, module.training) for module in
             [*trainer.hypernet.modules(), *trainer.target.modules()]]
    bar = progress_bar(len(tasks), 'Evaluate tasks', trainer.progress, leave=False)
    result = {}
    try:
        for task in tasks:
            trainer.hypernet.eval()
            trainer.target.eval()
            with torch.no_grad():
                weights = trainer.hypernet.weights_for(task)
                buffers = {name:value.clone() for name,value in trainer.task_buffers[task].items()}
                loader = DataLoader(trainer.tasks[task]['test'], batch_size=trainer.config.eval_batch_size)
                batches = progress_bar(len(loader), f'Validate {task}', trainer.progress, leave=False)
                correct = total = 0
                try:
                    for images, labels in loader:
                        outputs = functional_call(trainer.target, (weights,buffers),
                                                  (images.to(trainer.device),), strict=True)
                        correct += (outputs.argmax(1)==labels.to(trainer.device)).sum().item()
                        total += labels.numel()
                        batches.update(1)
                finally:
                    batches.close()
                result[task] = 100.0 * correct / total
            bar.update(1)
            bar.set_postfix_str(f'task {task}: {result[task]:.1f}%')
        return result
    finally:
        restore_rng(state)
        for module, mode in modes:
            module.training = mode
        bar.close()


def run_request(trainer, action, task, protected, active, record, boundary,
                *, forget_interval=20, evaluation_interval=20, on_step=None):
    """boundary(active) must persist state before returning or requesting a stop.

    Model and task buffers are saved by the caller. active contains the original
    protection snapshot, Adam state, loader/noise generator state, and counters.
    Learning saves at epoch boundaries; an interrupted epoch is replayed.

    on_step(trainer, row) sees every learning update, with the batch tensors
    that record rows leave out. Unlike boundary, it is not wrapped in anything:
    a callback that reads the model must guard its own randomness.
    """
    h, config = trainer.hypernet, trainer.config
    resumed = active is not None
    if not resumed:
        snapshot = h.snapshot()
        active = {'action': action, 'task': task, 'protected': list(protected),
                  'epoch': 0, 'step': 0, 'epoch_losses': [],
                  'snapshot': cpu_copy(snapshot.state_dict()),
                  'snapshot_tasks': list(snapshot.task_codes),
                  'optimizer': None, 'generator': None}
        if action == 'learn':
            h.add_task(task)
            trainer.task_buffers[task] = {name: value.clone()
                for name, value in trainer.buffer_template.items()}
    else:
        if (active['action'], active['task'], active['protected']) != (action, task, list(protected)):
            raise ValueError('Active request does not match the requested operation.')
        snapshot = h.snapshot()
        # A learning checkpoint also has the new code; the reference predates it.
        for name in list(snapshot.task_codes):
            if name not in active['snapshot_tasks']:
                del snapshot.task_codes[name]
        snapshot.load_state_dict(active['snapshot'])
    trainer._reference = {}
    trainable = list(h.generator_parameters())
    if action == 'learn':
        trainable.append(h.task_codes[task])
        if len(h.task_codes) == 1:
            trainable.append(h.chunk_codes)
    elif action != 'forget':
        raise ValueError('Unknown study action.')
    h.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    rate = config.learning_rate if action == 'learn' else (
        config.forgetting_learning_rate or config.learning_rate)
    optimizer = torch.optim.Adam(trainable, lr=rate)
    if active['optimizer'] is not None:
        optimizer.load_state_dict(active['optimizer'])
    generator = (torch.Generator().manual_seed(config.seed) if action == 'learn'
                 else random_stream(config.seed, 'forget', task, trainer.device))
    if active['generator'] is not None:
        generator.set_state(active['generator'].cpu())

    def save_boundary():
        active['optimizer'] = cpu_copy(optimizer.state_dict())
        active['generator'] = generator.get_state().cpu()
        # Evaluation and saving callbacks must not change training randomness.
        rng = capture_rng()
        try:
            boundary(active)
        finally:
            restore_rng(rng)

    h.train()
    trainer.target.train()
    if not resumed:
        record({'action': action, 'task': task, 'step': 0,
                'code_sha256': _hash_named([(task, h.task_codes[task])])})
        save_boundary()
    if action == 'learn':
        loader = DataLoader(_IndexedDataset(trainer.tasks[task]['train']),
            batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
        bar = progress_bar(config.epochs * len(loader), f'Learn {task}', trainer.progress, leave=False)
        bar.update(active['step'])
        try:
            for epoch in range(active['epoch'], config.epochs):
                running = task_running = protection_running = 0.0
                for images, labels, indices in loader:
                    weights = h.weights_for(task)
                    scores = functional_call(trainer.target, (weights, trainer.task_buffers[task]),
                                             (images.to(trainer.device),), strict=True)
                    task_loss = F.cross_entropy(scores, labels.to(trainer.device))
                    preservation = trainer.preserve(protected, snapshot)
                    loss = task_loss + config.beta * preservation
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite learning loss.')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    active['step'] += 1
                    running += loss.item()
                    task_running += task_loss.item()
                    protection_running += preservation.item()
                    record({'action': action, 'task': task, 'step': active['step'],
                        'epoch': epoch + 1, 'loss': loss.item(), 'task_loss': task_loss.item(),
                        'protection_loss': preservation.item(), 'indices': indices.tolist(),
                        'input_sha256': _hash_named([('images', images), ('labels', labels)])})
                    if on_step is not None:
                        on_step(trainer, {'task': task, 'step': active['step'],
                                          'epoch': epoch + 1,
                                          'task_loss': task_loss.item(),
                                          'images': images, 'labels': labels})
                    bar.update(1)
                    bar.set_postfix_str(f'epoch {epoch+1}/{config.epochs}, loss {loss.item():.3f}')
                active['epoch'] = epoch + 1
                active['epoch_losses'].append(running / len(loader))
                record({'action': action, 'task': task, 'epoch': epoch+1,
                    'step': active['step'], 'boundary': True,
                    'loss': running / len(loader), 'task_loss': task_running / len(loader),
                    'protection_loss': protection_running / len(loader),
                    'accuracies': score(trainer, list(h.task_codes))})
                save_boundary()
        finally:
            bar.close()
    else:
        steps = config.burn_in
        bar = progress_bar(steps, f'Unlearn {task}', trainer.progress, leave=False)
        bar.update(active['step'])
        try:
            for step in range(active['step'] + 1, steps + 1):
                noise_loss = generated_noise_loss(h, task, generator, config.noise_samples)
                preservation = trainer.preserve(protected, snapshot)
                loss = config.gamma * noise_loss + preservation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite unlearning loss.')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                active['step'] = step
                row = {'action': action, 'task': task, 'step': step,
                       'loss': loss.item(), 'noise_loss': noise_loss.item(),
                       'protection_loss': preservation.item()}
                if step == 1 or step % evaluation_interval == 0 or step == steps:
                    row['accuracies'] = score(trainer, list(h.task_codes))
                record(row)
                bar.update(1)
                if step % forget_interval == 0 or step == steps:
                    save_boundary()
        finally:
            bar.close()
    trainer._reference = {}
    return active
