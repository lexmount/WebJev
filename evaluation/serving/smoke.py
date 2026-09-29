#!/usr/bin/env python3
"""Check a WebJev model directory offline: load it in vLLM (no server) and answer a few typed decisions.

Same readout as adapter.py: the prompt ends at the answer slot, the option-label tokens are the only allowed ids, and
the processed logits vLLM returns for them are softmaxed at the model's temperature over the options.

    python smoke.py <model dir> [examples.json] [out.json]

examples.json: [[context, [{"question": ..., "options": [...]}, ...]], ...]  (built-in examples when omitted)
out.json:      {"model", "temperature", "answers": [[{"choice", "probs": {option: p}}, ...], ...]}
"""
import json
import math
import os
import sys
from dataclasses import dataclass

from vllm import LLM, SamplingParams

EXAMPLES = [
    ["The mug I received arrived smashed into pieces.",
     [{"question": "Which intent does the user's message express?",
       "options": ["track_order", "cancel_order", "change_address", "report_damage", "billing_question"]}]],
    ["URL: https://www.example.com/cart\nTitle: Cart\nText: Your cart has 1 item. Subtotal $24.99. [1] Proceed to checkout [2] Continue shopping",
     [{"question": "Task: buy the item in the cart. Which element should be clicked next?",
       "options": ["[1] Proceed to checkout", "[2] Continue shopping"]}]],
]

model_dir = sys.argv[1]
examples = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else EXAMPLES
out_path = sys.argv[3] if len(sys.argv) > 3 else None
if os.path.isdir(os.path.join(model_dir, "decider")):
    sys.path.insert(0, model_dir)                           # prompt package shipped with the model
from decider.prompt import MAX_OPTIONS, build, label_table  # noqa: E402  (upstream Mapika/decider)


@dataclass
class Q:                                                    # the fields prompt.build reads
    text: str; options: list; gold: int = 0


@dataclass
class Example:
    context: str; qs: list; task: str = "infer"; image: bytes = None


class _KeepOrder:                                           # keep option order as given
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


cfg = json.load(open(os.path.join(model_dir, "decider_config.json")))  # upstream model-directory format
temperature = float(cfg.get("temperature", 1.0))
llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=34816, gpu_memory_utilization=0.9, max_num_seqs=64,
          logprobs_mode="processed_logits", max_logprobs=256, seed=0)
tok = llm.get_tokenizer()
_, label_ids, _ = label_table(tok)
prompts, params, index = [], [], []
for e, (context, questions) in enumerate(examples):
    for k, q in enumerate(questions):
        item = build(Example(context, [Q(q["question"], list(q["options"]), 0)]), tok, _KeepOrder(), max_options=MAX_OPTIONS,
                     max_ctx_tokens=int(cfg.get("max_state_tokens", 32768)))
        n = item["nopts"][0]
        prompts.append({"prompt_token_ids": item["ids"]})
        params.append(SamplingParams(max_tokens=1, temperature=1.0, logprobs=n, allowed_token_ids=label_ids[:n]))
        index.append((e, k, n))
outs = llm.generate(prompts, params, use_tqdm=False)
answers = [[None] * len(qs) for _, qs in examples]
for (e, k, n), o in zip(index, outs):
    top = o.outputs[0].logprobs[0]
    logits = [top[label_ids[j]].logprob for j in range(n)]  # processed logits, despite the field name
    m = max(logits)
    z = [math.exp((x - m) / temperature) for x in logits]
    s = math.fsum(z)
    p = [x / s for x in z]
    options = examples[e][1][k]["options"]
    j = max(range(n), key=p.__getitem__)
    answers[e][k] = {"choice": options[j], "probs": dict(zip(options, p))}
result = {"model": model_dir, "engine": "vllm", "temperature": temperature, "answers": answers}
if out_path:
    json.dump(result, open(out_path, "w"), indent=1)
print(json.dumps(answers, indent=1))
