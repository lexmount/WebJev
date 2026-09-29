# Single-step decision benchmarks

These are 8 public benchmarks of **typed decisions**: the model reads a state and answers one or more questions.
A question is `choice` (pick one option), `noul` (yes/no) or `score` (pick a level). They measure general decision
quality outside the browser. The browser itself is measured end to end in [`../liveweb/`](../liveweb/).

Every item is sent as the exact body of a Jev request (`{"state", "questions"}`). The same file goes unchanged to Jev
and to WebJev-35B-A3B (served by [`../serving/`](../serving/)). Scoring is exact match against the gold label; no LLM
judge is involved.

## Results

| Benchmark | Questions | WebJev-35B-A3B | Jev 1.13 | Only WebJev right / only Jev right | Paired p |
| --- | ---: | ---: | ---: | ---: | ---: |
| JevBench (public items) | 231 | **87.88%** (203) | 85.71% (198) | 11 / 6 | 0.33 |
| Nimble (eval split) | 324 | **92.90%** (301) | 92.59% (300) | 11 / 10 | 1.0 |
| SemIf (external items) | 252 | 98.02% (247) | **98.41%** (248) | 3 / 4 | 1.0 |
| scienthoon support tickets | 873 | **76.29%** (666) | 74.91% (654) | 58 / 46 | 0.28 |
| kev transfer-v4 (dev) | 764 | 85.08% (650) | **85.21%** (651) | 52 / 53 | 1.0 |
| kev decision-v7 (dev) | 1,468 | **86.31%** (1,267) | 83.31% (1,223) | 88 / 44 | 0.0002 |
| MMLU-Pro (10-way) | 1,000 | 69.40% (694) | **83.40%** (834) | 35 / 175 | <0.0001 |
| typed-decisions (test) | 2,000 | 73.90% (1,478) | **74.05%** (1,481) | 231 / 234 | 0.93 |
| **Mean of 8** | | 83.72% | **84.70%** | | |

- **Accuracy** is correct questions divided by questions. The mean is the unweighted mean of the 8 accuracies.
- **Paired columns.** Both models answered the same questions. The two counts are the questions only one of them
  answered correctly. The p-value is a two-sided exact McNemar test on those counts.
- **Where the models differ.** The two models are statistically tied on 6 of the 8 benchmarks. WebJev is clearly
  better on kev decision-v7. Jev is clearly better on MMLU-Pro. MMLU-Pro measures stored college-level knowledge more
  than reading a state, and most of the gap in the mean comes from it.
- **How the runs were made.** WebJev-35B-A3B: [`../serving/`](../serving/) on one A100 80GB, 1 client, one item at a
  time, 2026-09-29. Jev 1.13: OpenRouter's Decisions API, model `typesafe/jev-1.13` (served as
  `typesafe/jev-1.13-20260917`), 2026-09-21.
- **Recorded results.** [`results/scores.json`](results/scores.json) holds both columns. Our model's raw answers are in
  [`results/webjev-35b-a3b/`](results/webjev-35b-a3b/). Each row is `{"id", "answers"}`, with the answers exactly as
  the server returned them. `python score.py results/webjev-35b-a3b` recomputes the WebJev column.

## The benchmarks

We chose benchmarks that meet two rules:

1. **The items exist as files.** They can go unchanged to both models in the `{state, questions}` shape, and the gold
   label ships with each item.
2. **Jev has a published result on them.** Our own Jev run can then be checked against it.

| Benchmark | Items / questions | Types (choice / noul / score) | What it measures | Published Jev result | Our Jev run |
| --- | --- | --- | --- | --- | --- |
| [JevBench](https://github.com/fstandhartinger/jevbench) public items | 231 / 231 | 139 / 74 / 18 | General typed decisions: intent, extraction, tool choice, policy; a hard tier with long policies, multi-hop, time and number reasoning, traps | 0.866 on the 231 public items (leaderboard per-item results) | 0.857 |
| [Nimble](https://github.com/bespokelabsai/nimble) eval split | 324 / 324 | 146 / 114 / 64 | Minimal-pair evidence checks: one fact changes and the answer flips | 302/324 = 0.932 (Nimble README) | 0.926 |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) external items | 252 / 252 | 252 / 0 / 0 | Is a claim supported, contradicted or undecided by the facts? | 0.965 on 144 authored items, 1.00 on 108 perturbations (kev model card) | 0.972 / 1.000 |
| [scienthoon](https://github.com/scienthoon/jev-ood-calibration) support tickets | 291 / 873 | 291 / 291 / 291 | Ticket triage: queue, is the customer angry, priority | queue 0.897, angry 0.914, priority 0.447 (scienthoon, cited by kev) | 0.897 / 0.914 / 0.436 |
| [kev](https://github.com/jaredpalmer/kev) transfer-v4 dev | 764 / 764 | 444 / 280 / 40 | Transfer to MMLU, Emotion, TweetEval-offensive, QNLI, PAWS, SciQ and generated policy rules | 0.857 (kev model card) | 0.852 |
| kev decision-v7 dev | 1,204 / 1,468 | 756 / 472 / 240 | Text classification: news topic, review sentiment and stars, Banking77 intents (77 options), question type, entity type, BoolQ, MNLI, policy rules | 0.845 per record (kev model card) | 0.833 per question |
| MMLU-Pro 10-way sample | 1,000 / 1,000 | 1,000 / 0 / 0 | College-level knowledge and reasoning, up to 10 options | 0.840 (kev model card) | 0.834 |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test | 400 / 2,000 | 600 / 600 / 800 | Operations: stop or escalate an agent run, handle a support request, approve an invoice, triage a security alert | 0.727 (dataset card, TypeSafe API) | 0.741 |

Our Jev runs agree with the published numbers to within 1.5 points. For decision-v7, kev reports accuracy per
record (1,204 records) and we report it per question (1,468 questions).

### Details

- **JevBench public items.** Files `datasets/public/{easy,original,hard}.jsonl` of
  [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) at `fd51755`. There are 48 easy, 72
  standard and 111 hard items. The leaderboard has 534 items and publishes 231 of them. Example: state *"The mug I
  received arrived smashed into pieces."*, question *"Which intent does the user's message express?"*, options
  `track_order` / `cancel_order` / `change_address` / **`report_damage`** / `billing_question`.
- **Nimble eval split.** `data/eval.jsonl` of Bespoke Labs' [nimble](https://github.com/bespokelabsai/nimble) at
  `f136b3f`. The context and the counterfactual differ in one factual sentence, and the right answer changes with it.
  Example: a desk-lamp listing where the packing sheet and the listing disagree about the charger; the question asks
  for the highest severity and route under an ordered rubric. All labels are synthetic: a model checked them and no
  person reviewed them (Nimble README).
- **SemIf external items.** Human-written items from [SemIf](https://github.com/TheoLeeCJ/SemIf) (`ca3ba65`), frozen by
  kev in `evals/external/semif-v1/development.jsonl`: 144 authored items and 108 perturbations. Example: *"The
  optician ordered replacement lenses. The workshop confirms they have not yet been fitted to the customer's
  glasses."*, claim *"the replacement lenses have been fitted"*: **contradicted**.
- **scienthoon support tickets.** Tickets from
  [scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration) (`e092d3f`), frozen by kev in
  `evals/external/scienthoon-v1/development.jsonl`. Each ticket has three questions: the queue (`choice`), whether the
  customer is angry (`noul`) and the priority (`score`). Some tickets mix Korean into English text.
- **kev transfer-v4 dev.** `evals/v4/transfer-v4/development.jsonl` of [kev](https://github.com/jaredpalmer/kev) at
  `e0bcf50`. The sources and their item counts: composition_holdout 96, emotion 116, legacy_holdout 80, mmlu 116,
  paws 80, qnli 80, sciq 116 and tweet_offensive 80.
- **kev decision-v7 dev.** `evals/v7/decision-v7/development.jsonl` of kev. The sources and their item counts: agnews
  300, amazon 80, banking77 116, boolq 80, compositional 128, dbpedia14 116, imdb 80, legacy_policy 96, mnli 116, sst5
  80, trec 116 and yelp 160. Example: *"Is my currency okay to add money with?"*, question *"Which banking intent best
  describes this customer message?"*, 77 options.
- **MMLU-Pro 10-way.** A 1,000-question sample of the [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
  test split, drawn by [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) (seed 42) and frozen by kev
  in `evals/external/ekzhang-mmlupro-v1/records.jsonl`.
- **typed-decisions test.** [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
  (by codelion), config `all`, split `test`, revision `468b146`. The data has 4 workflows with 500 questions each.
  The gold label is the mean of three samples from a teacher model of about 4B parameters, so the score measures
  agreement with that teacher. The teacher's agreement with itself is 0.735, which is roughly the ceiling.

## Run it yourself

Python 3.10+ with `pandas` and `pyarrow`, which read the typed-decisions parquet. Everything is written under
`$EVAL_ROOT`, which defaults to `./work`.

```bash
./fetch_sources.sh        # clone the pinned sources, download the pinned typed-decisions parquet
python prepare.py         # -> work/bench/*.jsonl; checks each file against the SHA-256 of the inputs we evaluated
python score.py results/webjev-35b-a3b      # score our recorded answers: the WebJev column above
```

`prepare.py` prints `identical to the evaluated input` for each benchmark when your inputs are byte-identical to ours.

Run a model yourself with [`run.py`](run.py). It sends each item to any Jev-compatible endpoint, caches the answers
under `work/pred/<name>/`, and only re-sends missing or failed items when rerun:

```bash
# WebJev-35B-A3B served by ../serving on this machine
python run.py --name webjev --url http://127.0.0.1:8200/v1/systemone
# Jev 1.13 through OpenRouter (the route used for the table above) or through TypeSafe's API
OPENROUTER_API_KEY=... python run.py --name jev --url https://openrouter.ai/api/alpha/decisions \
    --model typesafe/jev-1.13-20260917 --key-env OPENROUTER_API_KEY --workers 6
TYPESAFE_API_KEY=... python run.py --name jev-typesafe --url https://api.typesafe.ai/v1/systemone \
    --model jev-1.13.0 --key-env TYPESAFE_API_KEY --workers 6

python score.py webjev jev --json work/scores.json
```

`--limit 5` sends only the first 5 items of each benchmark, which is enough for a smoke test. Rerunning WebJev with
`../serving` gives the same answers up to near-ties: the batching of vLLM can flip a question whose top two options
are within a few hundredths of each other.

### Scoring rules

- `choice`: the returned `choice` must equal the gold option.
- `noul`: the answer is "yes" when the returned probability is at least 0.5.
- `score`: the most probable level.
- An unanswered or failed item counts as wrong for all its questions.
- Questions with fewer than two options are not scored. None of these 8 benchmarks has one.

## Files

| File | What it does |
| --- | --- |
| [`fetch_sources.sh`](fetch_sources.sh) | Clones jevbench, nimble and kev at the pinned commits; downloads the pinned typed-decisions parquet and checks its SHA-256 |
| [`prepare.py`](prepare.py) | Converts the sources into `work/bench/<benchmark>.jsonl` (one item format, see [`common.py`](common.py)) |
| [`run.py`](run.py) | Sends items to a Jev-compatible endpoint (WebJev or Jev) and caches the answers |
| [`score.py`](score.py) | Accuracy per benchmark, mean of 8, paired comparison of two models |
| [`common.py`](common.py) | Item format, gold and predicted labels |
| [`results/`](results/) | Recorded scores of both models and the raw answers of WebJev-35B-A3B |

## Sources and licenses

This repository does not redistribute benchmark data. `fetch_sources.sh` downloads every file from its upstream,
and each keeps its own license:

| Source | License |
| --- | --- |
| JevBench | MIT |
| kev | Apache-2.0 |
| SemIf | MIT, as recorded in kev's manifest |
| scienthoon/jev-ood-calibration | MIT, as recorded in kev's manifest |
| MMLU-Pro | MIT |
| typed-decisions | Apache-2.0 |
| Nimble | No license file at the pinned commit |
| MMLU-Pro sample indices (ekzhang/openjev-sglang) | No license file |

The published Jev numbers above come from each benchmark's own reports and from the
[kev model card](https://github.com/jaredpalmer/kev). WebJev is an independent project by Lexmount and is not
affiliated with or endorsed by TypeSafe AI.
