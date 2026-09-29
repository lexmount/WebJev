"""Deterministic verifier for the live-web tasks: evidence capture (getters) x checks (metrics) -> verdict.

    verifier.evidence.collect_evidence(cdp_url, task, task_dir)   before the browser session is deleted -> evidence.json
    verifier.judge.judge_task_dir(task_dir)                       any time later, offline                -> judge.json
    python -m verifier.judge RESULTS_DIR                          score every episode of a result directory

No LLM is involved in scoring.
"""
