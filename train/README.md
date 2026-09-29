# Training WebJev-35B-A3B

WebJev-35B-A3B is [Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) fine-tuned
into a typed-decision model: given a state and questions with listed options, it returns a probability for
every option of every question in one forward pass. Training is one supervised epoch over a single mixture of
3.19M decision records (1.21B tokens), including 64K decisions taken on real web pages. It runs on one node
with 8 x A100 80GB.

## Quick start

```bash
cd train
./setup.sh                                  # pinned upstream checkout + training and data environments
.venv/bin/hf download Qwen/Qwen3.5-35B-A3B-Base --revision 0f0813072d2358973511097385626f21fcb6d422 \
    --local-dir models/Qwen3.5-35B-A3B-Base
bash data/build_all.sh models/Qwen3.5-35B-A3B-Base    # builds the data mixture: work/mixture/items.pkl
./train.sh                                  # starts in the background; ./train.sh --check only checks
```

`train.sh` prints the run directory, `runs/<run-id>/`. The run continues after you log out, and exports the
model to `models/<run-id>/` when the epoch is complete.

| What | Where |
|---|---|
| Training log | `runs/<run-id>/training.log` |
| Metrics per update (loss, gradient norm, learning rate, throughput, memory) | `runs/<run-id>/curve.jsonl`, TensorBoard in `runs/<run-id>/tensorboard` |
| State (phase, step, ETA) | `runs/<run-id>/status.json`, `runs/<run-id>/pipeline-status.json` |
| Checkpoints (every 2,000 updates and the last; the newest two are kept) | `runs/<run-id>/checkpoints/step-NNNNNN/` |
| Exported model | `models/<run-id>/` |

**Resume** after an interruption with `./train.sh --resume runs/<run-id>`; training continues from the newest
complete checkpoint with the same data order. If training crashes, the run restarts itself from the newest
checkpoint up to twice; create a file named `STOP` in the run directory to prevent that.

**Export** a checkpoint by hand with
`.venv/bin/python src/export.py --run runs/<run-id> --base models/Qwen3.5-35B-A3B-Base --out models/<name> [--step N]`.
The directory has the layout of the released WebJev-35B-A3B checkpoint (safetensors, tokenizer, prompt package and
decision settings) and is served by [`../evaluation/serving`](../evaluation/serving) with the same API as Jev.

## Recipe

| | |
|---|---|
| Base model | Qwen/Qwen3.5-35B-A3B-Base, revision `0f08130` (Apache-2.0) |
| Objective | cross-entropy over the listed options at each answer slot; the option-letter logits of the LM head are read, nothing is generated |
| Trained parameters | full fine-tuning of every non-expert weight: attention, shared experts, routers, embeddings, norms, LM head; **2,448,355,968** of 34,660,610,688 language-model parameters. The 32,212,254,720 routed-expert parameters stay frozen. No LoRA, no RL |
| Optimizer | upstream Muon for the transformer-block matrices (momentum 0.95, Nesterov, 5 Newton-Schulz steps, update scaled by 0.2 * sqrt(max dimension)) and AdamW for the rest (betas 0.9 / 0.95, eps 1e-8); FP32 master weights, no weight decay, gradient clipping at 1.0 |
| Learning rate | 1e-5, constant (`warmup_cosine` is available in the config) |
| Batches | length-bucketed batches of at most 8,192 padded tokens, 16 batches per optimizer update; one epoch, every record exactly once |
| Precision and memory | bf16 weights on every GPU; the FP32 optimizer state is sharded across the 8 ranks and kept on the CPU between updates; routed experts run the grouped-MM kernels in checkpointed blocks of 1,024 token rows |
| Data | [data/README.md](data/README.md): general decisions from public datasets, teacher-written contrastive pairs, Open-Jev, knowledge multiple choice, Nimble, and our live-web decisions ([Hugging Face](https://huggingface.co/datasets/Lexmount/WebJev)) |

All settings are in [`configs/webjev-35b-a3b.yaml`](configs/webjev-35b-a3b.yaml).

## Files

```
setup.sh, train.sh          environment setup, launcher
configs/                    the training configuration
src/launch.py               input and GPU checks, run directory, background start / resume
src/supervisor.py           train (with automatic resume), then export
src/train.py                the training loop (torchrun, 8 ranks)
src/model_runtime.py        model loading, frozen routed experts, bounded expert blocks
src/sharded_recipe_optim.py sharded CPU-resident state around the upstream Muon/AdamW updates
src/batch_schedule.py       balanced assignment of whole records to ranks, checkpoint policy
src/export.py               checkpoint -> full model directory, frozen-expert check
data/                       data sources, synthesis and processing scripts
tests/                      optimizer equivalence (GPU) and CPU checks
```

Tests: `.venv/bin/python -m pytest tests/test_cpu.py -q` on any machine, and
`.venv/bin/torchrun --standalone --nproc_per_node 2 tests/test_optimizer_equivalence.py` on GPUs (the sharded
optimizer must match the upstream optimizers bit for bit, including after a resume).

## Acknowledgements

The training framework, prompt format and data recipe come from [Mapika/decider](https://github.com/Mapika/decider)
(Apache-2.0), used unchanged at commit `c4daaac`; see [NOTICE.md](NOTICE.md).
