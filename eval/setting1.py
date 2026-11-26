#!/usr/bin/env python3
"""Setting I: Full-Context, search disabled.

输入默认: research/DelegSearch/data/delegsearch_bench.jsonl
输出: <output-dir>/raw.jsonl; 汇总用 summarize.py --setting 1

CLI 示例:
    python research/DelegSearch/eval/setting1.py \
        --models gpt-5-mini,claude-sonnet-4.6 \
        --judge deepseek-v4-flash \
        --n-runs 3 --parallel 32

Default input: research/DelegSearch/data/delegsearch_bench.jsonl
Output: <output-dir>/raw.jsonl; summarize with summarize.py --setting 1

CLI example:
    python research/DelegSearch/eval/setting1.py \
        --models gpt-5-mini,claude-sonnet-4.6 \
        --judge deepseek-v4-flash \
        --n-runs 3 --parallel 32
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 允许既 `python research/Eval/eval_setting1.py` 又 `python -m` 运行
# Allow running via either `python research/Eval/eval_setting1.py` or `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it

import common as ec  # noqa: E402


SYSTEM_PROMPT_SETTING1 = (
    "You are a helpful research assistant. Answer the user's question based "
    "on the retrieved documents provided below. Base your answer on the "
    "information contained in these documents. When you use information "
    "from a document, cite its doc_id in square brackets, e.g. [web1]. "
    "Give a clear, accurate, and concise final answer."
)


USER_TEMPLATE_SETTING1 = (
    "Question:\n{query}\n\n"
    "Retrieved documents:\n\n{docs_text}\n\n"
    "Answer the question above, citing the doc_ids you used."
)


def build_setting1_messages(
    query: str,
    ordered_doc_ids: List[str],
    full_docs: Dict[str, Dict[str, Any]],
    max_doc_chars: int,
) -> List[Dict[str, Any]]:
    docs_text = ec.docs_to_prompt_text(
        ordered_doc_ids, full_docs, max_doc_chars=max_doc_chars,
    )
    user = USER_TEMPLATE_SETTING1.format(query=query, docs_text=docs_text)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SETTING1},
        {"role": "user", "content": user},
    ]


def run_one(
    client,
    model: str,
    query: str,
    ordered_doc_ids: List[str],
    full_docs: Dict[str, Dict[str, Any]],
    temperature: float,
    seed: int,
    max_doc_chars: int,
) -> Dict[str, Any]:
    """跑一次 Setting I 采样, 返回 {model_answer, text_source, failure}。

    Run one Setting I sample and return {model_answer, text_source, failure}.
    """
    messages = build_setting1_messages(
        query, ordered_doc_ids, full_docs, max_doc_chars,
    )
    try:
        resp = ec.create_chat(
            client, model, messages,
            tools=None,
            temperature=temperature,
            seed=seed,
        )
    except Exception as e:
        return {
            "model_answer": "",
            "text_source": None,
            "failure": f"call_failed:{e}",
        }
    choice = resp.choices[0]
    msg = choice.message
    text, src = ec.extract_message_text(msg)
    return {
        "model_answer": text,
        "text_source": src,
        "failure": None if text.strip() else "empty_response",
    }


def process_sample(
    client,
    judge_client,
    record: Dict[str, Any],
    models: List[str],
    variants: List[Tuple[str, List[str]]],
    full_docs: Dict[str, Dict[str, Any]],
    evidence_doc_ids: List[str],
    n_runs: int,
    temperature: float,
    judge_temperature: float,
    seed: int,
    max_doc_chars: int,
    judge_model: str,
    done_keys: set,
    output_path: str,
) -> int:
    """对单条样本按 (model × variant × run) 组合跑评测, 返回写盘条数。

    Evaluate one sample for each (model × variant × run) combination and
    return the number of records written.
    """
    rid = record.get("id") or record.get("data_id")
    query = record.get("query", "") or ""
    shaping_mode = record.get("shaping_mode", "") or ""
    answer_type = record.get("answer_type", "") or ""
    gold_answer = record.get("golden_answer", "") or ""

    written = 0
    for model in models:
        for variant_name, ordered_ids in variants:
            for run_idx in range(n_runs):
                eval_key = f"{rid}::setting1::{model}::{variant_name}::{run_idx}"
                if eval_key in done_keys:
                    continue
                run_seed = seed + run_idx * 101
                run_res = run_one(
                    client, model, query, ordered_ids, full_docs,
                    temperature=temperature, seed=run_seed,
                    max_doc_chars=max_doc_chars,
                )
                judge_res: Optional[Dict[str, Any]] = None
                if run_res["failure"] in (None, "empty_response"):
                    judge_res = ec.call_judge(
                        judge_client, judge_model,
                        query=query,
                        shaping_mode=shaping_mode,
                        answer_type=answer_type,
                        gold_answer=gold_answer,
                        gold_evidence_doc_ids=evidence_doc_ids,
                        model_answer=run_res["model_answer"] or "",
                        action_taken="direct_answer",
                        temperature=judge_temperature,
                    )
                if judge_res is None:
                    label = "wrong" if run_res["failure"] else "judge_failed"
                    cited: List[str] = []
                    rationale = ""
                else:
                    label = judge_res["label"]
                    cited = judge_res["cited_doc_ids"]
                    rationale = judge_res["rationale"]

                out_obj: Dict[str, Any] = {
                    "eval_key": eval_key,
                    "id": rid,
                    "shaping_mode": shaping_mode,
                    "answer_type": answer_type,
                    "category": record.get("category"),
                    "subtype": record.get("subtype"),
                    "model": model,
                    "variant": variant_name,
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "temperature": temperature,
                    "n_docs": len(ordered_ids),
                    "gold_evidence_doc_ids": evidence_doc_ids,
                    "model_answer": run_res["model_answer"],
                    "text_source": run_res["text_source"],
                    "failure": run_res["failure"],
                    "judge_model": judge_model,
                    "judge_label": label,
                    "judge_cited_doc_ids": cited,
                    "judge_rationale": rationale,
                }
                ec.append_jsonl(output_path, out_obj)
                done_keys.add(eval_key)
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DelegSearchBench Setting I evaluator "
            "(Full-Context, search disabled)"
        ),
    )
    ec.add_common_args(parser)
    parser.add_argument(
        "--n-variants", type=int, default=len(ec.VARIANT_NAMES),
        help=(
            f"跑几个 doc-order 变体 (1..{len(ec.VARIANT_NAMES)}), 按 "
            f"{list(ec.VARIANT_NAMES)} 顺序取前 N 个"
        ),
    )
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = str(ec._PKG_ROOT / "results" / "setting1")

    ec.set_qpm(args.qpm)

    models = ec.parse_models(args.models)
    if args.judge in models:
        parser.error("judge 模型不能出现在 --models 中")
    if not (1 <= args.n_variants <= len(ec.VARIANT_NAMES)):
        parser.error(
            f"--n-variants 必须在 [1, {len(ec.VARIANT_NAMES)}] 之间"
        )

    if not Path(args.input).exists():
        parser.error(f"--input 不存在: {args.input}")

    records = ec.load_jsonl(args.input)
    records = ec.filter_records_by_mode(records, args.only_mode)
    total_loaded = len(records)
    records = ec.apply_limits(records, args.limit, args.limit_per_mode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = str(output_dir / "raw.jsonl")

    done_keys = ec.load_done_keys(
        raw_path, key_field="eval_key",
        retry_failures=args.retry_failures,
    )
    n_retriable = (
        ec.count_retriable_rows(raw_path)
        if args.retry_failures else 0
    )

    print(f"[INFO] input           : {args.input}")
    print(f"[INFO] output_dir      : {output_dir}")
    print(f"[INFO] models          : {models}")
    print(f"[INFO] judge           : {args.judge}")
    print(
        f"[INFO] n_runs={args.n_runs}, n_variants={args.n_variants}, "
        f"temperature={args.temperature}, "
        f"judge_temperature={args.judge_temperature}, "
        f"seed={args.seed}, variant_seed={args.variant_seed}"
    )
    print(
        f"[INFO] loaded={total_loaded}, to_process={len(records)}, "
        f"done_evaluations={len(done_keys)}"
    )
    if args.retry_failures and n_retriable:
        print(
            f"[INFO] retry_failures=True: {n_retriable} historical rows "
            "will be re-executed (failure!=None or judge_label==judge_failed)"
        )

    variant_names = list(ec.VARIANT_NAMES[: args.n_variants])

    if args.dry_run:
        if not records:
            print("[INFO] nothing to process")
            return
        r0 = records[0]
        full_docs, evidence_doc_ids, buckets = ec.build_full_docs(r0)
        variants = ec.generate_doc_orders(
            full_docs, evidence_doc_ids, args.variant_seed,
        )[: args.n_variants]
        print("\n=== dry-run: buckets ===")
        print(json.dumps(
            {k: len(v) for k, v in buckets.items()},
            ensure_ascii=False, indent=2,
        ))
        print("\n=== dry-run: evidence_doc_ids ===")
        print(json.dumps(evidence_doc_ids, ensure_ascii=False))
        print("\n=== dry-run: variant doc orders ===")
        for name, ids in variants:
            print(f"[{name}] {ids}")
        print("\n=== dry-run: sample messages (variant=golden_first) ===")
        msgs = build_setting1_messages(
            r0.get("query", ""), variants[0][1], full_docs, args.max_doc_chars,
        )
        print(json.dumps(msgs, ensure_ascii=False, indent=2)[:4000])
        return

    client = ec.build_client(
        base_url=args.base_url, api_key=args.api_key,
    )
    judge_client = ec.build_client(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key or ec.DEFAULT_JUDGE_API_KEY,
    )

    def _worker(record: Dict[str, Any]) -> Tuple[str, int, Optional[str]]:
        rid = record.get("id") or record.get("data_id") or "?"
        try:
            full_docs, evidence_doc_ids, _buckets = ec.build_full_docs(record)
            variants = ec.generate_doc_orders(
                full_docs, evidence_doc_ids, args.variant_seed,
                variant_names=variant_names,
            )
            n = process_sample(
                client, judge_client, record, models, variants, full_docs,
                evidence_doc_ids,
                n_runs=args.n_runs,
                temperature=args.temperature,
                judge_temperature=args.judge_temperature,
                seed=args.seed,
                max_doc_chars=args.max_doc_chars,
                judge_model=args.judge,
                done_keys=done_keys,
                output_path=raw_path,
            )
            return rid, n, None
        except Exception as e:
            return rid, 0, f"exception:{e!r}"

    if not records:
        print("[INFO] no records to process")
        return

    total_written = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futs = [pool.submit(_worker, r) for r in records]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="setting1"):
            rid, n, err = fut.result()
            total_written += n
            if err:
                print(f"[ERR] sample {rid}: {err}", file=sys.stderr)

    print(
        f"[DONE] setting1: wrote {total_written} new rows -> {raw_path}"
    )
    print("       run: python research/DelegSearch/eval/summarize.py --setting 1 "
          f"--input-dir {output_dir}")


if __name__ == "__main__":
    main()
