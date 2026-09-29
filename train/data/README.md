# Training data

WebJev-35B-A3B is trained on one mixture of typed decisions. Every record is a state (text or JSON), one or
more questions, the listed options of each question, and the index of the correct option. The model learns to
put probability on the correct option; it is never trained to generate text.

We do not redistribute the public datasets. This directory has the scripts that rebuild every public-source
component from its original source at pinned revisions, and the description of how each component was made.
Our own component, the live-web decisions, is published on Hugging Face.

| Component | Built by | Records | Questions | Tokens |
|---|---|---:|---:|---:|
| 1. General decisions (public datasets, teacher-written data, programmatic augmentation) | [`general/`](general) | 1,587,715 | 2,022,356 | 470,606,867 |
| 2. Live-web decisions (ours) | [`web.py`](web.py) | 64,122 | 64,122 | 327,573,009 |
| 3. Open-Jev typed decisions | [`open_jev.py`](open_jev.py) | 294,573 | 294,573 | 118,597,787 |
| 4. Knowledge multiple choice: Nemotron-RL-knowledge-mcqa | [`knowledge_mcqa.py`](knowledge_mcqa.py) | 600,189 | 600,189 | 149,748,150 |
| 4. Knowledge multiple choice: OpenScienceReasoning-2 | [`knowledge_mcqa.py`](knowledge_mcqa.py) | 641,762 | 641,762 | 143,792,001 |
| 4. Nimble evidence-checking decisions | [`knowledge_mcqa.py`](knowledge_mcqa.py) | 5,362 | 5,362 | 2,153,574 |
| **Total** | [`assemble.py`](assemble.py) | **3,193,723** | **3,628,364** | **1,212,471,388** |

Tokens are the model-input tokens of the rendered records (Qwen3.5 tokenizer, no padding). The counts are those
of the data the released model was trained on. A rebuild from the pinned sources applies the same rules, but
is not guaranteed to reproduce the counts exactly.

## How the records are rendered

All components go through the same upstream prompt builder (`decider.prompt.build` of
[Mapika/decider](https://github.com/Mapika/decider)), which is also the format the model is served with: the
state after `Context:`, each question with its options labeled `(A)`, `(B)`, ..., and one answer slot per
question whose option-letter logits are read. Options are shuffled and the gold index follows them. At most 255
options per question and a context budget of 16,384 tokens.

## 1. General decisions

Built with the upstream data recipe of Mapika/decider at commit `c4daaac`, in its single-run `full` mode
(seed 6). [`general/build_general.sh`](general/build_general.sh) runs the steps below in order.

- **Public datasets** ([`general/convert_tasks.py`](general/convert_tasks.py)). All 99 tasks registered by the
  upstream package are converted with the upstream loaders into the unified format: classification, natural
  language inference, sentiment and emotion, intent and topic, question answering and multiple choice,
  retrieval relevance, preference and rating, toxicity and safety, tool and function selection, web actions
  (Mind2Web), agent trajectories, and small game environments. 70 tasks provide training rows (983,309 records);
  29 tasks are held out entirely, and every task keeps its own evaluation split out of training. The Hugging Face
  revision of each of the 81 source datasets is pinned in
  [`general/source_revisions.json`](general/source_revisions.json). Two sources are read from verified local
  copies ([`general/sources.py`](general/sources.py)): the TREC test split (the parquet conversion checked row
  by row against the original file) and the eleven original Mind2Web training files (checked against their
  LFS SHA-256). The Super Mario Bros. task is generated with the upstream emulator code
  ([`general/build_mario.py`](general/build_mario.py)).
- **Existing teacher data.** The teacher-written files shipped with the upstream repository are used as they
  are: 3,043 custom questions, 7,449 routing messages, 6,399 terse routing messages, 4,771 shell commands and
  1,502 situations, plus 669 option descriptions.
- **Teacher-written contrastive pairs** ([`general/generate_contrastive.py`](general/generate_contrastive.py)).
  With the upstream prompts, task families and domains, a teacher model (`deepseek-v4.1-flash` through an
  OpenAI-compatible API) writes a base state, a changed state in which one fact is edited, a shared question
  and the two answers, which must differ. Two fresh requests then answer the two states independently, without
  the labels and without the other state; a pair is kept only if both answers equal the written labels exactly.
  Of 6,000 candidates, 5,595 pairs pass. The pairs of 6 held-out domains (626) are used only as probes; the
  other 4,969 pairs (9,938 records) are repeated three times by the recipe, 29,814 training records in total.
  The families are claim verification, answerability, paraphrase, entailment, policy, eligibility,
  guardrails, moderation, rubric rating, reference resolution and conditional routing; the domains range from
  airline support and insurance claims to DevOps alerts and web-agent observations. Because the endpoint gives
  no log-probabilities, the upstream probability filters are replaced by this exact-agreement check; the check
  is the teacher agreeing with itself, not human review.
- **Programmatic augmentation** ([`general/build_mixture.py`](general/build_mixture.py) runs the upstream
  mixture builder unchanged). On top of the converted tasks it adds the following parts:

  | Part | Records | What it teaches |
  |---|---:|---|
  | general | 983,309 | every training task in the unified format |
  | wide | 64,000 | full label sets (up to 255 options) and small label sets padded with unrelated labels |
  | padded | 25,000 | distractor options |
  | described | 39,984 | options given as descriptions or rubrics, often under opaque names |
  | json | 32,000 | JSON states with several records, questions addressed by path |
  | json_indexed | 20,000 | index selection inside long JSON arrays |
  | single | 30,000 | one question, or a reordered subset, of multi-question records |
  | custom | 18,030 | the teacher-written custom questions, split and combined |
  | isolated | 236,619 | one yes/no question per rating level or option |
  | commands | 9,264 | shell-command risk and scope |
  | routing | 9,702 | queues with a generic bucket and a catch-all |
  | rules | 90,000 | rule-conditioned decisions over JSON records, each with a twin whose label flips |
  | contrastive | 29,814 | the teacher-written contrastive pairs |
  | **total** | **1,587,722** | |

- **Tokenization** ([`general/tokenize_items.py`](general/tokenize_items.py)): the upstream settings, i.e.
  "none of the above" added with probability 0.1 where it applies, schema-first layout with probability 0.5,
  one RNG seeded 0; run in parallel with the serial result (the RNG state of every chunk is checked, and chunks
  are compared field by field with the upstream serial function).
- **Clean-up** ([`general/finalize.py`](general/finalize.py)): 7 records that the JSON-state builder emitted
  without any question are removed (1,587,715 records remain), and portable parquet copies are written.

## 2. Live-web decisions (ours)

Published as the Hugging Face dataset **[Lexmount/WebJev](https://huggingface.co/datasets/Lexmount/WebJev)**; the dataset card
there documents the collection, labeling and audits in full.

Each record is one decision a browser agent faced on a real website: the page state it observed (URL, title,
visible text and the numbered interactive elements), the task, and either the choice of the next operation
(click, type, select, scroll, wait, go back, done, ...) or the choice of the element, input field or dropdown
option to act on. The 64,122 records are 31,050 next-operation decisions (configuration `action_prediction`) and
33,072 target decisions (configuration `element_grounding`). [`web.py`](web.py) downloads both configurations and
renders every row without changing or dropping any (state-first layout, one RNG seeded 0). The RNG only shuffles
the options of each question, so a rebuild contains exactly the decisions we trained on.

## 3. Open-Jev typed decisions

From [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) (eleven configurations) and
[ZefanCai/Open-Jev-v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1), train splits only.
[`open_jev.py`](open_jev.py) admits the thirteen control families (browser control, reasoning, entity
alignment, information retrieval, phone, amount and email extraction, context retention, citation, mailroom,
customer, sponsor segments, silent failures), seven game and control families (drone control, painting
geometry, snake, tic-tac-toe, tile platformer, T-Rex runner, ViZDoom), WANLI decisions and the community policy
scenarios. The workflow-controls family is left out because a single visible workflow question does not fully
specify the action priority.

- Only rows with a hard one-hot label; soft-label rows are skipped, never coerced to one answer.
- Labels are re-derived before use: browser-control and reasoning-control rows from their visible rules
  ([`rule_audit.py`](rule_audit.py); no source program is executed), community rows from their visible
  numbered rules (the dataset authors' checker, pinned), and WANLI rows against the original human-labeled
  WANLI train file.
- The datasets' own held-out splits are respected (IDs, groups, seeds, premises, scenario families, semantic
  contexts and source instances of the calibration, validation, test and OOD splits, and the WANLI test set).
- Each original question contributes one record. A rating question becomes, by a fixed hash, either the
  original multi-level question, the yes/no question for its correct level, or the yes/no question for an
  adjacent wrong level.
- Repeated IDs, repeated visible decisions (every copy is dropped when their labels disagree), repeated
  tokenized inputs and decisions already present in components 1-2 are removed.

## 4. Knowledge multiple choice and Nimble

[`knowledge_mcqa.py`](knowledge_mcqa.py) reads
[nvidia/Nemotron-RL-knowledge-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa) (all four
train shards), [nvidia/OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2)
(train) and [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) (`data/train.jsonl`).

- A question is used only if it ends in a contiguous labeled A/B/C/... option block (2-255 options) with one
  unambiguous gold option; the OpenScience reasoning traces are never read. The question is rendered as the
  state, with the fixed question "Which option correctly answers the question?".
- A question that appears with two different gold answers is dropped entirely; repeated questions are kept
  once; questions already present in components 1-2 are skipped.
- Nimble: all 2,676 train rows; rating rows become one yes/no question per level, giving 5,362 records.

## Exclusions

The builders of components 3 and 4 drop every candidate that matches an evaluation item exactly after text
normalization ([`eval_guard.py`](eval_guard.py)).

## Rebuilding

```bash
bash data/build_all.sh models/Qwen3.5-35B-A3B-Base
```

`build_all.sh` downloads the pinned sources ([`download_sources.py`](download_sources.py), SHA-256 checked),
builds the four components into `work/packs/`, indexes components 1-2 for the duplicate checks
([`index_components.py`](index_components.py)) and mixes everything into `work/mixture/items.pkl`
([`assemble.py`](assemble.py)): later components drop inputs that already exist in the mixture, and the result
is shuffled once. Every step is resumable and runs on CPU.

Only one step calls an LLM: the teacher-written contrastive pairs (section 1). To generate them, also set
`TEACHER_BASE_URL` and `TEACHER_API_KEY` for any OpenAI-compatible endpoint (our build used `deepseek-v4.1-flash`
and recorded 7,161 generation and 12,000 verification responses). Without them this step is skipped, and the
mixture is built without the contrastive part.

Each public source keeps its own license and terms of use; the license of this repository's code does not
extend to the data.
