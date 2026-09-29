# Evaluation

WebJev is evaluated at two levels. They answer different questions, so we report both.

| Level | Question | Where |
| --- | --- | --- |
| **Real-website tasks** | Put the model inside a browser agent on live websites. Does it finish the task? | [`liveweb/`](liveweb/) |
| **Single-step decisions** | Given one state and one question, does the model pick the right option? | [`benchmarks/`](benchmarks/) |

Both evaluations call the model through the same Jev-compatible API. [`serving/`](serving/) runs WebJev-35B-A3B on vLLM behind that API, so any
harness that already talks to Jev only needs a new base URL and key.

## Real-website tasks (125 tasks, end to end)

We hand-picked 125 tasks on live websites from [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web),
[WebGym](https://github.com/microsoft/webgym) and [WebVoyager](https://github.com/MinorJerry/WebVoyager). **Every task is
verifiable.** Right before the browser session closes, the harness captures evidence from the live page: the final URL,
the open tabs, page text and values, and the agent's final answer. The task's deterministic verifier then scores that
evidence. No LLM decides whether a run succeeded, so the evaluation keeps real websites without the noise of an LLM
judge: the same evidence always gets the same verdict. Each run uses the same browser agent, the same Lexmount cloud
browser and the same task list. Only the model that picks each action (operation and target element) changes. The full
method is in [`liveweb/README.md`](liveweb/README.md).

Success rate = successful tasks / gradable tasks. A task is not gradable when the browser environment failed or the
verifier itself raised an error.

| Subset | WebJev-35B-A3B | Jev 1.13 |
| --- | ---: | ---: |
| **All 125 tasks** | **38.52%** (47/122) | 16.67% (20/120) |
| Online-Mind2Web (75) | **38.36%** (28/73) | 16.90% (12/71) |
| WebGym (26) | **32.00%** (8/25) | 8.00% (2/25) |
| WebVoyager (24) | **45.83%** (11/24) | 25.00% (6/24) |

120 tasks were gradable for both models. On 30 of them WebJev succeeded and Jev failed; on 3 Jev succeeded and
WebJev failed (both succeeded on 17, both failed on 70). Jev ran on 2026-09-24 and WebJev on 2026-09-29. Both runs used the same agent build, harness, tasks and verifiers.
Live websites can change between days.

## Single-step decision benchmarks (8 public benchmarks)

Each item gives the model a state and one or more typed questions: `choice` (pick one option), `noul` (yes/no) or
`score` (pick a level). An item counts as correct when the model's top answer equals the gold label. The mean is the
unweighted mean of the 8 accuracies.

| Benchmark | Questions | WebJev-35B-A3B | Jev 1.13 |
| --- | ---: | ---: | ---: |
| JevBench (public items) | 231 | **87.88%** | 85.71% |
| Nimble (eval split) | 324 | **92.90%** | 92.59% |
| SemIf (external items) | 252 | 98.02% | **98.41%** |
| scienthoon support tickets | 873 | **76.29%** | 74.91% |
| kev transfer-v4 (dev) | 764 | 85.08% | **85.21%** |
| kev decision-v7 (dev) | 1,468 | **86.31%** | 83.31% |
| MMLU-Pro (10-way) | 1,000 | 69.40% | **83.40%** |
| typed-decisions (test) | 2,000 | 73.90% | **74.05%** |
| **Mean of 8** | | 83.72% | **84.70%** |

On general typed decisions WebJev is close to Jev. It wins 4 of the 8 benchmarks, and most of the gap in the mean comes
from MMLU-Pro, which tests stored knowledge more than reading a state. [`benchmarks/README.md`](benchmarks/README.md)
describes each benchmark, where it comes from and how to rerun it.
