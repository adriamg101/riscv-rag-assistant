"""Evaluate models with eval/questions.json.

Examples:
    python -m eval.run_eval --models gemini,groq,local --pause 5
    python -m eval.run_eval --mode two_stage          (no LLM: retrieval and abstention)
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.config import ProviderConfig, load_providers
from app.llm import LLMError, ProviderRateLimitedError, generate, timeout_seconds
from app.rag import RagAnswer, answer_from_chunks
from app.retrieve import MODES, TOP_K, RetrievedChunk, retrieval_mode, retrieve
from eval.metrics import abstention_correct, answer_correct, citation_correct, retrieval_hit

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_JSON = Path("eval/results.json")
RESULTS_MD = Path("eval/results.md")
# Questions kept out of system tuning. Results are stored separately.
HOLDOUT_PATH = Path("eval/holdout.json")
HOLDOUT_JSON = Path("eval/holdout_results.json")
HOLDOUT_MD = Path("eval/holdout_results.md")
DEFAULT_PAUSE_SECONDS = 5.0
MAX_RATE_LIMIT_WAIT_SECONDS = 120.0
RETRIEVAL_ONLY = "retrieval-only"
METRICS = ("retrieval_hit", "answer_correct", "citation_correct", "abstention_correct")

Sleep = Callable[[float], None]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def warm_up(provider: ProviderConfig) -> None:
    """Load the local model so its startup does not count towards measured latency."""
    try:
        generate("Reply with: OK", [{"text": "OK"}], provider)
        print(f"{provider.name}: warm-up complete")
    except LLMError as error:
        print(f"{provider.name}: warm-up failed ({error})")


def ask_with_retry(
    question: str,
    chunks: Sequence[RetrievedChunk],
    provider: ProviderConfig,
    sleep: Sleep = time.sleep,
) -> tuple[RagAnswer | None, str | None]:
    """On a 429, wait as instructed (up to a maximum) and retry once."""
    providers = {provider.name: provider}
    try:
        return answer_from_chunks(question, chunks, providers, provider.name), None
    except ProviderRateLimitedError as error:
        wait = (error.until - datetime.now()).total_seconds()
        if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
            return None, str(error)
        sleep(max(wait, 0.0) + 1.0)
    except LLMError as error:
        return None, str(error)
    try:
        return answer_from_chunks(question, chunks, providers, provider.name), None
    except LLMError as error:
        return None, str(error)


def score_row(
    question: dict,
    chunks: Sequence[RetrievedChunk],
    response: RagAnswer | None,
    error: str | None,
    uses_llm: bool,
) -> dict:
    expected = question["expected_section"]
    answerable = question["answerable"]
    row = {
        "id": question["id"],
        "answerable": answerable,
        "retrieval_hit": retrieval_hit(expected, {chunk.section_id for chunk in chunks}),
        "answer_correct": None,
        "citation_correct": None,
        "abstention_correct": None,
        "abstained": None,
        "latency_ms": None,
        "answer": None,
        "raw_answer": None,
        "rejection": None,
        "error": error,
    }
    if response is None:
        return row

    abstained = response["abstained"]
    row.update(
        abstained=abstained,
        abstention_correct=abstention_correct(answerable, abstained),
        answer=response["answer"],
        raw_answer=response["raw_answer"],
        rejection=response["rejection"],
    )
    if uses_llm:
        row["latency_ms"] = response["latency_ms"]
        section_ids = {
            chunk.section_id
            for chunk in chunks
            if chunk.source_url in {source["source_url"] for source in response["sources"]}
        }
        row["answer_correct"] = answer_correct(
            response["answer"], question["must_contain"], answerable, abstained
        )
        row["citation_correct"] = citation_correct(section_ids, expected, answerable, abstained)
    return row


def summarize(rows: Sequence[dict]) -> dict:
    metrics = {}
    for name in METRICS:
        values = [row[name] for row in rows if row[name] is not None]
        metrics[name] = round(sum(values) / len(values), 4) if values else None
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    return {
        "metrics": metrics,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
        },
        "errors": sum(row["error"] is not None for row in rows),
    }


def evaluate(
    questions: Sequence[dict],
    retrieved: Mapping[str, Sequence[RetrievedChunk]],
    provider: ProviderConfig | None,
    mode: str,
    pause: float = DEFAULT_PAUSE_SECONDS,
    sleep: Sleep = time.sleep,
) -> dict:
    """Run one model (or retrieval only when provider is None) over every question."""
    rows = []
    called_llm = False
    for question in questions:
        chunks = retrieved[question["id"]]
        if provider is None:
            response, error = answer_from_chunks(question["question"], chunks, {}), None
        else:
            if called_llm and pause > 0:
                sleep(pause)
            response, error = ask_with_retry(question["question"], chunks, provider, sleep)
            called_llm = error is not None or (response is not None and bool(response["model"]))
        rows.append(score_row(question, chunks, response, error, provider is not None))
        print(f"{provider.name if provider else RETRIEVAL_ONLY} {question['id']}", flush=True)

    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": provider.name if provider else RETRIEVAL_ONLY,
        "model": provider.model if provider else RETRIEVAL_ONLY,
        "retrieval_mode": mode,
        "top_k": TOP_K,
        "n_questions": len(rows),
        **summarize(rows),
        "rows": rows,
    }


def merge_runs(previous: Sequence[dict], new: Sequence[dict]) -> list[dict]:
    """Replace only repeated runs, so each model can be evaluated separately."""
    key = lambda run: (run["provider"], run["model"], run["retrieval_mode"])  # noqa: E731
    merged = {key(run): run for run in previous}
    merged.update({key(run): run for run in new})
    return list(merged.values())


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def markdown_table(runs: Sequence[dict]) -> str:
    lines = [
        "| Model | Retrieval | Section hit | Answer | Citations | Abstention "
        "| Mean latency | Errors | Date |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        metrics = run["metrics"]
        mean = run["latency_ms"]["mean"]
        latency = "—" if mean is None else f"{mean:.0f} ms"
        lines.append(
            f"| {run['model']} | {run['retrieval_mode']} | {_percent(metrics['retrieval_hit'])} "
            f"| {_percent(metrics['answer_correct'])} | {_percent(metrics['citation_correct'])} "
            f"| {_percent(metrics['abstention_correct'])} | {latency} | {run['errors']} "
            f"| {run['date'][:10]} |"
        )
    return "\n".join(lines) + "\n"


def save_results(runs: Sequence[dict], json_path: Path, md_path: Path) -> list[dict]:
    previous = []
    if json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))["runs"]
    merged = merge_runs(previous, runs)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"runs": merged}, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(payload, encoding="utf-8", newline="\n")
    md_path.write_text(markdown_table(merged), encoding="utf-8", newline="\n")
    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG assistant evaluation")
    parser.add_argument("--models", default="", help="comma-separated providers")
    parser.add_argument("--mode", choices=MODES, help="defaults to RETRIEVAL_MODE")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SECONDS)
    parser.add_argument("--warmup", default="local", help="providers to warm up")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--output-md", type=Path, default=RESULTS_MD)
    parser.add_argument("--holdout", action="store_true", help="use the holdout question set")
    args = parser.parse_args(argv)
    if args.holdout:
        args.questions, args.output_json, args.output_md = HOLDOUT_PATH, HOLDOUT_JSON, HOLDOUT_MD
    return args


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    providers = load_providers() if names else {}
    missing = [name for name in names if name not in providers]
    if missing:
        raise SystemExit(f"Providers not configured: {', '.join(missing)}")
    if names:
        print(f"LLM timeout: {timeout_seconds():.0f} s")

    mode = args.mode or retrieval_mode()
    questions = load_questions(args.questions)[: args.limit]
    retrieved = {q["id"]: retrieve(q["question"], mode=mode) for q in questions}

    warmups = {name.strip().lower() for name in args.warmup.split(",")}
    runs = []
    for provider in [providers[name] for name in names] or [None]:
        if provider is not None and provider.name in warmups:
            warm_up(provider)
        runs.append(evaluate(questions, retrieved, provider, mode, args.pause))

    merged = save_results(runs, args.output_json, args.output_md)
    print(markdown_table(merged))


if __name__ == "__main__":
    main()
