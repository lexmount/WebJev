# Serving WebJev-35B-A3B (Jev-compatible API)

This directory serves WebJev-35B-A3B on one GPU with [vLLM](https://github.com/vllm-project/vllm) and puts a small
adapter in front of it. The adapter speaks Jev's API: the same routes, request bodies and response bodies. Any client
that already calls Jev (a browser agent, the benchmark runner in [`../benchmarks/`](../benchmarks/), the
[live-web harness](../liveweb/), the [demo app](../../apps/browser-agent/)) switches to WebJev by changing only the base
URL and the key.

| File | What it does |
| --- | --- |
| [`serve.sh`](serve.sh) | Starts `vllm serve` on 127.0.0.1 and the adapter in front of it, waits until both are healthy |
| [`adapter.py`](adapter.py) | FastAPI adapter: Jev-shaped request → prompts → vLLM → Jev-shaped answers |
| [`smoke.py`](smoke.py) | Offline check of a model directory: loads it in vLLM (no server) and answers a few decisions |
| [`requirements.txt`](requirements.txt) | Serving environment |

## Requirements

- Linux with one NVIDIA GPU with 80 GB of memory (tested on one A100 80GB). The model is a mixture of experts with
  35B total and 3B active parameters, served in bf16.
- Python 3.12, vLLM 0.30.0 (tested with torch 2.13.0+cu130 and transformers 5.17.0).
- The model directory (`MODEL_DIR`): the [WebJev-35B-A3B release](https://huggingface.co/Lexmount/WebJev-35B-A3B) with weights, tokenizer, `decider_config.json` (prompt settings) and the upstream `decider/` prompt code.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# Prompt code of the upstream training framework (Mapika/decider, Apache-2.0), installed without its dependencies.
# Skip this when the model directory contains its own decider/ folder.
uv pip install --python .venv/bin/python --no-deps \
  "decider @ git+https://github.com/Mapika/decider@c4daaac28af9fea95d627015cffa2dd5a5926ee6"
```

## Serve

```bash
huggingface-cli download Lexmount/WebJev-35B-A3B --local-dir ./WebJev-35B-A3B   # weights, tokenizer, prompt code
MODEL_DIR=./WebJev-35B-A3B VENV=.venv ./serve.sh
```

The API is then at `http://127.0.0.1:8200`. `GPU`, `PORT`, `HOST`, `VLLM_PORT`, `RUN_DIR` and `SERVED_MODEL_NAME` (the
model id returned in every response, default `webjev-35b-a3b`) override the defaults (see the header of
[`serve.sh`](serve.sh)). Logs, pid files and the exact vLLM command go to `RUN_DIR`. To stop the
server, kill the two pids in `RUN_DIR/adapter.pid` and `RUN_DIR/vllm.pid`.

The very first start compiles the model (`torch.compile`, cached on disk) and takes several minutes. Restart the
server once after that first start. On an A100 80GB, a start with a cold compile cache left 1.9 GiB for the KV cache;
a start with a warm cache left 4.2 GiB.

The adapter listens on 127.0.0.1 unless you set `HOST`. To listen on another address you must set a key. Every route
except `/health` then requires `Authorization: Bearer <key>`:

```bash
WEBJEV_API_KEY=<choose a key> HOST=0.0.0.0 MODEL_DIR=/path/to/WebJev-35B-A3B VENV=.venv ./serve.sh
```

The connection is plain HTTP. The key and the page content your agent sends travel unencrypted, so put a TLS proxy
in front of the adapter before you expose it beyond a trusted network.

## Call it

A request holds a `state` (a string or any JSON value) and named `questions`. There are three question types:
`choice` (2–255 options), `noul` (yes/no) and `score` (2–10 ordered levels). Both routes accept the same body.

```bash
curl -s http://127.0.0.1:8200/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "The mug I received arrived smashed into pieces.",
  "questions": {
    "intent":   {"type": "choice", "instructions": "Which intent does the user'\''s message express?",
                 "criteria": ["track_order", "cancel_order", "change_address", "report_damage", "billing_question"]},
    "angry":    {"type": "noul",   "instructions": "Is the customer upset?"},
    "priority": {"type": "score",  "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]}
  }
}'
```

The response below was returned by WebJev-35B-A3B for this request. Probabilities are rounded to 4 digits.

```json
{
  "model": "webjev-35b-a3b",
  "answers": {
    "intent": {"type": "choice", "choice": "report_damage", "confidence": 0.9999, "certainty": 0.9992,
               "probabilities": {"track_order": 0.0001, "cancel_order": 0.0, "change_address": 0.0,
                                 "report_damage": 0.9999, "billing_question": 0.0}},
    "angry": {"type": "noul", "noul": 0.9855},
    "priority": {"type": "score", "score": 1.45, "confidence": 0.5659, "certainty": 0.1435,
                 "legend": {"0": "low", "1": "medium", "2": "high"},
                 "probabilities": {"0": 0.1206, "1": 0.3136, "2": 0.5659},
                 "level_fit": {"0": 0.1082, "1": 0.2814, "2": 0.5078}, "fit_mass": 0.8974}
  },
  "usage": {"input_tokens": 178, "output_tokens": 0}
}
```

- `choice`: the most probable option and the distribution over all options. `confidence` is the probability of the
  chosen option. `certainty` is 1 minus the normalized entropy.
- `noul`: the probability of "yes".
- `score`: each level is judged in its own yes/no prompt (`level_fit`), then normalized into `probabilities`. `score`
  is the expected level. `fit_mass` is the unnormalized sum, which is low when no level fits.

`POST /api/alpha/decisions` (the OpenRouter route) takes the same body and returns the same answers. A `model` field in
the request is accepted and ignored.

### Switching from Jev

| You call Jev through | Replace the URL | Key |
| --- | --- | --- |
| TypeSafe: `https://api.typesafe.ai/v1/systemone` | `http://<host>:<port>/v1/systemone` | `WEBJEV_API_KEY`, or none on localhost |
| OpenRouter: `https://openrouter.ai/api/alpha/decisions` | `http://<host>:<port>/api/alpha/decisions` | `WEBJEV_API_KEY`, or none on localhost |

The behavior Jev clients rely on is kept:

- A `choice` question with exactly one option is answered with that option at probability 1, without calling the
  model. Browser agents ask such questions often, for example when a page has a single input field.
- The option order you send is the order in the answer.

One difference: `"independent": false` (several answers packed into one prompt) is rejected with HTTP 422. Leave it at
its default, `true`.

## How a decision is computed

Nothing is generated. For each question the adapter builds a prompt with the upstream prompt code. The prompt ends at
the answer slot `Answer: (`. vLLM then runs one prefill and returns the logits of the option-label tokens. The server
runs with `--logprobs-mode processed_logits`, and each request restricts sampling to those tokens with
`allowed_token_ids`. The adapter applies a softmax over the options at the model's temperature (1.0).

- **Several questions in one request** become separate prompts. They share the rendered state, and vLLM prefix caching
  (on by default) prefills that shared state once.
- **One `score` question** becomes one yes/no prompt per level.
- **Batching.** Each prompt is its own vLLM request with its exact token ids, so no request is padded.
- **Step size.** `--max-num-batched-tokens 16384` lets long web pages prefill in large steps.

## Measured speed

WebJev-35B-A3B on one A100 80GB with this `serve.sh` configuration, 1 client sending requests one at a time. The table
shows the median time per request as seen by the client.

| Benchmark | Requests | Median per request |
| --- | ---: | ---: |
| JevBench (public items) | 231 | 146 ms |
| Nimble (eval split) | 324 | 150 ms |
| SemIf (external items) | 252 | 72 ms |
| scienthoon support tickets (3 questions each) | 291 | 228 ms |
| kev transfer-v4 (dev) | 764 | 74 ms |
| kev decision-v7 (dev) | 1,204 | 144 ms |
| MMLU-Pro (10-way) | 1,000 | 145 ms |
| typed-decisions (test, 5 questions each) | 400 | 402 ms |

Web pages are much longer prompts. In the [125-task real-website evaluation](../liveweb/), the browser agent made 2,713
decisions. Three A100 80GB replicas served them, each with 10 concurrent browser episodes. Measured by the agent, a
decision took 1.18 s at the median, 2.89 s at p90 and 5.02 s at p99. A decision is 1 to 4 prompts that share the
page state.

## Check a model directory without a server

```bash
.venv/bin/python smoke.py /path/to/WebJev-35B-A3B            # built-in examples
.venv/bin/python smoke.py /path/to/WebJev-35B-A3B examples.json out.json
```

## Credits

The prompt format and the readout of option-label logits come from the upstream training framework
[Mapika/decider](https://github.com/Mapika/decider) (Apache-2.0, commit `c4daaac`). WebJev-35B-A3B was fine-tuned with
that framework and reads prompts built by its `decider.prompt` and `decider.systemone` modules, which this adapter
imports unchanged. The API shape follows Jev (TypeSafe AI). WebJev is an independent project by Lexmount and is not
affiliated with or endorsed by TypeSafe AI.
