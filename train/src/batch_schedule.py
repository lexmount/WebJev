"""Assign whole training examples to ranks without changing original-batch weights."""
from pathlib import Path
import shutil


def chunks(indices, items, limit):
    current = []
    longest = 0
    for index in indices:
        length = ((len(items[index]['ids']) + 63) // 64) * 64
        maximum = max(longest, length)
        if current and maximum * (len(current) + 1) > limit:
            yield current
            current = []
            longest = 0
        current.append(index)
        longest = max(longest, length)
    if current:
        yield current


def assign_batches(group, items, world, micro_tokens):
    """LPT on padded tokens; each task retains its source batch's denominator."""
    if not group or world < 1 or micro_tokens < 1:
        raise ValueError('A nonempty group, positive world size and token budget are required')
    tasks = []
    for batch_index, original in enumerate(group):
        denominator = sum(sum(gold >= 0 for gold in items[index]['golds']) for index in original)
        if denominator <= 0:
            raise ValueError('No supervised slots in an original batch')
        for portion in chunks(original, items, micro_tokens):
            nvalid = sum(sum(gold >= 0 for gold in items[index]['golds']) for index in portion)
            if nvalid <= 0:
                raise ValueError('No supervised slots in a micro-batch')
            length = ((max(len(items[index]['ids']) for index in portion) + 63) // 64) * 64
            tasks.append(dict(indices=portion, original_batch=batch_index, denominator=denominator,
                              nvalid=nvalid, padded_tokens=len(portion) * length,
                              attention_shape=len(portion) * length * length,
                              loss_weight=(world / len(group)) * (nvalid / denominator)))
    ranks = [[] for _ in range(world)]
    loads = [0] * world
    for task in sorted(tasks, key=lambda item: (item['padded_tokens'], item['attention_shape']), reverse=True):
        owner = min(range(world), key=lambda rank: (loads[rank], len(ranks[rank]), rank))
        ranks[owner].append(task)
        loads[owner] += task['padded_tokens']
    return ranks, loads


def checkpoint_due(step, total_steps, every):
    if every < 1:
        raise ValueError('checkpoint interval must be positive')
    return step % every == 0 or step == total_steps


def prune_complete_checkpoints(root, keep):
    """Delete older complete checkpoints only after the new one is durable."""
    if keep < 1:
        raise ValueError('keep_checkpoints must be positive')
    completed = sorted(Path(root).glob('step-*/ready.json'))
    removed = []
    for ready in completed[:-keep]:
        shutil.rmtree(ready.parent)
        removed.append(str(ready.parent))
    return removed
