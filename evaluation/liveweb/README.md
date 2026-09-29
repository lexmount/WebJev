# Real-website evaluation: 125 live tasks

Single-step benchmarks measure one decision on a frozen page. This evaluation measures what users actually get: the
same browser agent is run end to end on **125 real websites**, and only the decision model is swapped. Each task is
scored by a **deterministic verifier over evidence captured from the live page** -- no LLM acts as a judge. The tasks
run on real websites, and the verdict never depends on an LLM's opinion: the same evidence always gives the same
verdict.

## Results

| Task subset | WebJev-35B-A3B | Jev 1.13 |
|---|---:|---:|
| **All 125 tasks** | **38.52%** (47 / 122) | 16.67% (20 / 120) |
| Online-Mind2Web (75 tasks) | **38.36%** (28 / 73) | 16.90% (12 / 71) |
| WebGym (26 tasks) | **32.00%** (8 / 25) | 8.00% (2 / 25) |
| WebVoyager (24 tasks) | **45.83%** (11 / 24) | 25.00% (6 / 24) |

**Success rate = successes / gradable tasks.** A task is gradable when the episode ran and the verifier could score it.
Episodes lost to the environment (the cloud browser failed, or the page was an anti-bot wall at evidence time) and
episodes the verifier could not score (e.g. evidence capture timed out) are excluded from the denominator and reported
separately: 3 of 125 for WebJev-35B-A3B, 5 of 125 for Jev 1.13.

**Paired view.** On the 120 tasks gradable for both models, WebJev-35B-A3B succeeded where Jev failed on **30**
tasks, and Jev succeeded where WebJev-35B-A3B failed on **3**; 17 tasks were solved by both and 70 by neither.

Per-task outcomes, step counts and wall times of both runs are in [`results/`](results/) (`per_task.csv`,
`per_task.json`, `summary.json`). The Jev run was recorded on 2026-09-24 and the WebJev-35B-A3B run on 2026-09-29, with
the same agent build, tasks, harness and verifier. Live websites change from day to day, so a re-run will not
reproduce every single task, but the protocol is fixed.

## The 125 tasks

The tasks are **hand-picked from three public live-web benchmarks**:

| Source | Tasks | Upstream |
|---|---:|---|
| Online-Mind2Web | 75 | [OSU-NLP-Group/Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) (MIT) |
| WebGym | 26 | [microsoft/webgym](https://github.com/microsoft/webgym) |
| WebVoyager | 24 | [MinorJerry/WebVoyager](https://github.com/MinorJerry/WebVoyager) (Apache-2.0) |

They start on 76 different websites -- government services (gov.uk, irs.gov, usps.com, state DMVs and job boards), shopping and
listings (Best Buy, CarMax, Craigslist, LandWatch), health (Mayo Clinic, Healthline), weather, sports, finance
calculators, travel, research (arXiv) and computation (Wolfram|Alpha). Task types range from reaching a specific page
state (a filtered listing, a sorted table, a chart range, a form result) to finding a fact and reporting it.

How the set was built:

- **Only tasks that can be verified.** A task is kept only if its success can be decided from the live page and the
  agent's answer. Where the original wording did not say which site to use, which end state to leave, or which fields
  to report, the task text was made explicit so that task and check agree. Every task file keeps the original task
  text and, when the text was changed, the reason (`source.original_task_name`, `source.intent_modified_reason`).
- **Stable sites.** Tasks whose sites showed anti-bot walls or regional redirects from the cloud browser were left
  out. Many checks for values that change over time (the current product, today's weather, the last item of a list)
  read the value from the live page at evidence time instead of freezing it.
- **One task per upstream item.** No upstream task appears twice.

Each task is one JSON file in [`tasks/`](tasks/): `task_id`, `task_name` (what the agent is told), `website` (start
URL), `source` (benchmark and upstream id) and `evaluator` (the checks). A few answer patterns also accept non-English
spellings of the same value; those characters are stored as JSON `\u` escapes.

## How a task is scored

```
agent finishes --> evidence capture --> evidence.json --> browser released
                   (live page)               |
                                   verifier (offline) --> judge.json
```

1. **Evidence is captured from the live page before the browser is released**, through an independent CDP
   connection, never through the agent's own channel (the agent runtime calls the harness just before it closes the
   browser). The captured values are what the task's checks need: the
   final URL and its parameters, open tabs, page title and text, visible text, values of DOM elements, control
   states (selected options, ticked filters), the accessibility tree, browser storage (e.g. a store saved as "my
   store"), or the site's own API response (e.g. the server-side cart). The page the agent ended on is decided by
   browser facts (the foreground tab), not by what the agent believes. On a local Chrome shared by parallel runs,
   evidence is confined to the run's own browser context.
2. **A health gate runs first.** An anti-bot or verification wall, a browser error page or an empty page at evidence
   time is the environment's failure: the episode goes to the infra bucket and is retried, instead of being scored 0.
3. **Checks are declarative and deterministic.** Each check applies one metric to one piece of evidence:
   `url_pattern_match`, `url_matches`, `tabs_match`, `include_exclude`, `count_in_range`, `json_object_match`,
   `set_match` and `answer_match`. `answer_match` judges the final answer by content, not by wrapping: a value
   counts whether it is given as JSON or in a sentence, and numbers are accepted with thousands separators, currency
   symbols or as English number words; synonyms are never guessed.
4. **Combining checks.** A task passes when its checks pass: all of them for 123 tasks; for 2 tasks, either the
   answer check or the page-state check.
5. **Broken capture is not an agent failure.** Selectors that extract fields carry a canary; if the canary is missing
   on the page the check was written for, the verdict is a verifier error (excluded), not a 0. A page-specific check
   declares its prerequisite (`requires`), so an agent that never reached that page scores 0 instead of producing a
   verifier error.

The same `evidence.json` always yields the same `judge.json`, so a check can be revised and re-scored without
re-running the episode. [`tests/`](tests/) holds counterexample suites for part of the tasks: wrong end states
synthesized from real captured evidence (a missing filter, a look-alike domain, a hallucinated answer, a stale
answer) that the verifier must reject.

### Example

`vts-021` asks: *"On apple.com, open the Tech Specs page for the currently offered MacBook Air and leave the
specifications displayed."* (original Online-Mind2Web wording: *"Find technical specs for the latest Macbook Air on
Apple."*). The verifier requires the final URL to be the MacBook Air specs page (regional prefixes allowed) and, only
if that holds, the page text to contain the spec sections:

```json
{
 "conj": "and",
 "checks": [
  {"name": "target_page", "metric": "url_pattern_match",
   "result": {"type": "final_url"},
   "expected": {"type": "const",
    "value": "^https?://(?:www\\.)?apple\\.com/(?:[a-z]{2}(?:-[a-z]{2})?/)?macbook-air/specs/?(?:[?#]|$)"}},
  {"name": "target_content", "metric": "include_exclude",
   "result": {"type": "page_text"},
   "expected": {"type": "const", "value": {"must_include": ["MacBook Air", "Chip", "Display"]}},
   "requires": ["target_page"]}
 ]
}
```

Answer tasks use `answer_match` on the final answer; e.g. `vts-131` (*"Calculate the determinant of a 6x6 Hilbert
matrix."*) accepts `1/186313420339200000` and its equivalent decimal and scientific forms.

## The agent

Every model drives the same agent: the runtime of [`apps/browser-agent`](../../apps/browser-agent), used as the
**decision model alone** so that the decision model is the only variable.

- **Each step**, the runtime reads the page (URL, title, visible text, and up to 250 interactive elements with
  labels and context) and asks the decision model two typed questions: which operation (click, type, select, scroll,
  wait, press Enter, go back, Escape, DONE, BLOCKED -- only the operations the page allows) and which element. The
  runtime executes the most likely choice in the browser.
- **Text.** A decision model chooses; it does not write. Text to type and the final answer are written by a fixed
  text helper, DeepSeek V4.1 Flash (`deepseek-v4.1-flash`), from the pages the agent saw; it is the same for every
  decision model.
- **No help.** The service features that assist users are off: no LLM planner (the start URL and the task come from
  the task file), no stall hints, no recovery after BLOCKED, no completion review, no pause for a human login.
- **Budget.** 900 seconds and 60 actions per task; one process per task; 10 parallel episodes per model server; two
  retries for episodes that ended in the infra bucket.
- **Browser.** The reported runs used [Lexmount Browser](https://browser.lexmount.com/), cloud browser
  infrastructure for AI agents: every episode gets its own isolated session on demand, which is what makes many
  parallel episodes practical, so it is the recommended backend. A local Chrome is supported as well (see below);
  results on it can differ, because the network location changes what sites serve (and how often they show anti-bot
  pages) and one machine runs fewer episodes in parallel.
- **Time zone.** Relative dates in the agent's prompts are resolved in `Asia/Shanghai` (`BROWSER_AGENT_TIMEZONE`),
  as in the reported runs; the harness sets it.
- **Endpoints of the reported runs.** WebJev-35B-A3B served with vLLM behind the Jev-compatible decisions API of
  [`evaluation/serving`](../serving); Jev 1.13 through OpenRouter (`typesafe/jev-1.13-20260917`).

## Run it yourself

Requirements: Python 3.14, a browser (a [Lexmount Browser](https://browser.lexmount.com/) API key and project ID,
or a local Chrome), an OpenAI-compatible endpoint for the text helper, and the decision endpoints (a WebJev-35B-A3B
server from [`evaluation/serving`](../serving); an OpenRouter or TypeSafe key for Jev).

```bash
cd evaluation/liveweb
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt -r ../../apps/browser-agent/requirements.txt
cp .env.example .env          # fill in the browser, text-helper and decision-model keys
$EDITOR models.conf           # one row per decision model: URL, model name, key variable, parallel episodes
WEBJEV_AGENT_PYTHON=.venv/bin/python ./run_eval.sh
```

`run_eval.sh` runs every model of `models.conf` in parallel on all 125 tasks, scores every episode with the verifier
and prints the table above; results go to `runs/<RUN_TAG>/<model>/<task_id>/`. Useful variants:

```bash
# one model, two tasks
MODELS=WebJev-35B-A3B RUN_TAG=smoke ./run_eval.sh \
  --tasks vts-021-apple-current-macbook-air-specs,vts-131-webvoyager-wolfram-alpha-35
# resume: the same RUN_TAG skips tasks that already have a result; only infra episodes run again
RUN_TAG=20260929 ./run_eval.sh
# re-score offline (no browser) and summarize
.venv/bin/python -m verifier.judge runs/<RUN_TAG>/WebJev-35B-A3B --force
.venv/bin/python harness/summarize.py --run WebJev-35B-A3B=runs/<RUN_TAG>/WebJev-35B-A3B \
  --run Jev-1.13=runs/<RUN_TAG>/Jev-1.13
# verifier tests (offline)
.venv/bin/python -m pytest tests -q
```

**Browser choice.** `BROWSER=lexmount` (with `LEXMOUNT_API_KEY` and `LEXMOUNT_PROJECT_ID`) runs every episode in
its own Lexmount Browser session; `workers` in `models.conf` is the number of sessions each model uses at the same
time. `BROWSER=local` uses your own Chrome instead: start it with remote debugging and a separate profile directory,
e.g. on macOS

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 --user-data-dir="$HOME/.webjev-chrome"
```

and set `BROWSER=local` (and `CHROME_CDP_URL` if you use another port). Each episode opens its own browser context
(fresh cookies and storage) in that Chrome and closes it at the end. Because one Chrome serves every episode, the
runner caps parallel episodes at `LOCAL_CHROME_WORKERS` (default 2). Expect numbers that differ from the reported
runs: your network location, anti-bot pages and the lower parallelism all change what the agent sees.

## Files

| Path | What it is |
|---|---|
| `tasks/` | the 125 task definitions with their checks |
| `verifier/` | evidence capture (`cdp.py`, `evidence.py`), getters, metrics and the offline judge (`judge.py`) |
| `harness/runner.py` | runs one model on the tasks: parallel episodes, resume, infra retries |
| `harness/episode.py` | one task: the agent runtime with the decision model alone, evidence capture before the browser closes |
| `harness/buckets.py` | how an episode ended (agent_run / no_answer / timeout / infra) and the retry rule |
| `harness/cleanup.py` | releases an episode's cloud session if its worker process died |
| `harness/summarize.py` | success rates per subset and the paired comparison |
| `run_eval.sh`, `models.conf`, `.env.example` | one-command run and its configuration |
| `results/` | per-task outcomes of the reported WebJev-35B-A3B and Jev 1.13 runs |
| `tests/` | task-file checks and verifier counterexample suites |

## Notes

- A deterministic check only accepts what it describes. Like any fixed rubric, it can reject an unusual but valid
  end state or phrasing; both models are scored by exactly the same checks.
- The task texts derive from the upstream benchmarks; see their repositories for their licenses.
