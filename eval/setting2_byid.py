#!/usr/bin/env python3
"""Setting II evaluator filtered by explicit item ids.


CLI example:
    python research/DelegSearch/eval/setting2_byid.py \
        --models gpt-5-mini,claude-sonnet-4.6 \
        --judge deepseek-v4-flash \
        --n-runs 3 --parallel 32
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it

import common as ec  # noqa: E402
import setting2  # noqa: E402


DEFAULT_IDS_FILE = ec._DEFAULT_SETTING2_IDS


def _load_ids_file(path: str) -> List[str]:
    """Load one item id per line, ignoring blank lines and comments."""
    ids_path = Path(path)
    if not ids_path.exists():
        raise FileNotFoundError(f"--ids-file 不存在: {ids_path}")

    ids: List[str] = []
    seen: Set[str] = set()
    with ids_path.open("r", encoding="utf-8") as f:
        for line in f:
            item_id = line.strip()
            if not item_id or item_id.startswith("#"):
                continue
            if item_id in seen:
                continue
            ids.append(item_id)
            seen.add(item_id)
    return ids


def _parse_inline_ids(raw: str) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()
    for part in (raw or "").replace("\n", ",").split(","):
        item_id = part.strip()
        if not item_id or item_id in seen:
            continue
        ids.append(item_id)
        seen.add(item_id)
    return ids


def _merge_ids(ids_from_file: List[str], inline_ids: List[str]) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for item_id in ids_from_file + inline_ids:
        if item_id in seen:
            continue
        merged.append(item_id)
        seen.add(item_id)
    return merged


def _record_id(record: Dict[str, Any]) -> str:
    rid = record.get("id") or record.get("data_id") or ""
    return str(rid)


def _filter_records_by_ids(
    records: List[Dict[str, Any]],
    target_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Filter records by target_ids while preserving input record order."""
    target_set = set(target_ids)
    matched_ids: Set[str] = set()
    filtered: List[Dict[str, Any]] = []
    for record in records:
        rid = _record_id(record)
        if rid in target_set:
            filtered.append(record)
            matched_ids.add(rid)

    missing = [item_id for item_id in target_ids if item_id not in matched_ids]
    return filtered, missing


def _add_setting2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--keep-ratio", type=float, default=0.5,
        help=("初始 golden evidence 保留比例 (0..1); 按 evidence_spans 顺序"
              "从前往后取。默认 0.5 = 保留前一半, 隐藏后一半。"),
    )
    parser.add_argument("--min-keep", type=int, default=0,
                        help="至少保留 N 篇 golden evidence (默认 0)")
    parser.add_argument("--min-withhold", type=int, default=1,
                        help="至少隐藏 N 篇 golden evidence (默认 1)")
    parser.add_argument("--search-top-k", type=int, default=5,
                        help="模拟 search 每次返回的 top-k 文档数")
    parser.add_argument(
        "--max-model-searches",
        type=setting2._parse_max_searches,
        default="auto",
        help=("模型自主发起的 search 上限次数 (不含 Turn 0 harness 注入的那次)。"
              "接受整数 (强制固定值) 或 'auto' (默认, 按每条样本的 n_withheld / "
              "search_top_k 与 --search-budget-safety 推导)。"),
    )
    parser.add_argument(
        "--search-budget-safety", type=float, default=2.0,
        help=("auto 模式冗余因子: effective_max ≈ ceil(safety * n_withheld / "
              "top_k)。默认 2.0。"),
    )
    parser.add_argument(
        "--min-max-searches", type=int, default=1,
        help="auto 模式下 effective_max 的下界 (默认 1)",
    )
    parser.add_argument(
        "--max-max-searches", type=int, default=8,
        help="auto 模式下 effective_max 的上界 (默认 8)",
    )
    parser.add_argument(
        "--search-dedup", choices=["doc_id", "none"], default="doc_id",
        help=("同一次运行内, 若新一次 search 返回的 doc_id 完全落在过往已见 "
              "doc_id 集合里, 标记 turn.duplicate=True (仍消耗一次预算)。"),
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.judge in ec.parse_models(args.models):
        parser.error("judge 模型不能出现在 --models 中")
    if not (0.0 <= args.keep_ratio <= 1.0):
        parser.error("--keep-ratio 必须在 [0, 1] 之间")
    if args.min_withhold < 1:
        parser.error("--min-withhold 必须 >= 1 (Setting II 要求上下文不完整)")
    if args.max_model_searches != "auto" and not isinstance(
        args.max_model_searches, int
    ):
        parser.error("--max-model-searches 内部解析错误")
    if args.search_budget_safety <= 0:
        parser.error("--search-budget-safety 必须 > 0")
    if args.min_max_searches < 0:
        parser.error("--min-max-searches 必须 >= 0")
    if args.max_max_searches < args.min_max_searches:
        parser.error("--max-max-searches 必须 >= --min-max-searches")
    if not Path(args.input).exists():
        parser.error(f"--input 不存在: {args.input}")


def _build_max_searches_cfg(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    if args.max_model_searches == "auto":
        cfg = {
            "mode": "auto",
            "fixed": None,
            "safety": float(args.search_budget_safety),
            "floor": int(args.min_max_searches),
            "cap": int(args.max_max_searches),
        }
        info = (
            f"auto (safety={args.search_budget_safety}, "
            f"floor={args.min_max_searches}, cap={args.max_max_searches})"
        )
        return cfg, info

    cfg = {
        "mode": "fixed",
        "fixed": int(args.max_model_searches),
        "safety": float(args.search_budget_safety),
        "floor": int(args.min_max_searches),
        "cap": int(args.max_max_searches),
    }
    return cfg, f"fixed={args.max_model_searches}"


def _print_budget_preview(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    max_searches_cfg: Dict[str, Any],
) -> None:
    if not records:
        return

    n_withheld_list: List[int] = []
    effective_max_list: List[int] = []
    for record in records:
        _full_docs, evidence_doc_ids, _buckets = ec.build_full_docs(record)
        _kept, withheld = setting2._split_kept_withheld(
            evidence_doc_ids, args.keep_ratio, args.min_keep, args.min_withhold,
        )
        n_withheld = len(withheld)
        n_withheld_list.append(n_withheld)
        if max_searches_cfg["mode"] == "fixed":
            effective_max_list.append(int(max_searches_cfg["fixed"]))
        else:
            effective_max_list.append(setting2.derive_max_model_searches(
                n_withheld=n_withheld,
                top_k=args.search_top_k,
                safety=max_searches_cfg["safety"],
                floor=max_searches_cfg["floor"],
                cap=max_searches_cfg["cap"],
            ))

    def _stats(xs: List[int]) -> str:
        if not xs:
            return "n/a"
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        median = (
            xs_sorted[n // 2] if n % 2 == 1
            else (xs_sorted[n // 2 - 1] + xs_sorted[n // 2]) / 2
        )
        mean = sum(xs_sorted) / n
        return (f"min={xs_sorted[0]}, median={median}, "
                f"mean={mean:.2f}, max={xs_sorted[-1]}")

    ratios = [
        (args.search_top_k * effective_max) / n_withheld
        for n_withheld, effective_max in zip(n_withheld_list, effective_max_list)
        if n_withheld > 0
    ]
    ratio_mean = sum(ratios) / len(ratios) if ratios else float("nan")
    print(f"[INFO] budget preview  : n_withheld    {_stats(n_withheld_list)}")
    print(f"                         effective_max {_stats(effective_max_list)}")
    print(
        "                         theoretical recovery ratio "
        "(top_k*effective_max/n_withheld): "
        f"mean={ratio_mean:.2f}x"
    )

    if max_searches_cfg["mode"] == "auto":
        cap = max_searches_cfg["cap"]
        n_capped = sum(1 for effective_max in effective_max_list
                       if effective_max >= cap)
        if n_capped:
            print(
                f"[INFO] cap hit         : {n_capped}/{len(effective_max_list)}"
                f" samples reached max_max_searches={cap}"
            )


def _dry_run(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    max_searches_cfg: Dict[str, Any],
) -> None:
    if not records:
        print("[INFO] nothing to process")
        return

    record = records[0]
    full_docs, evidence_doc_ids, buckets = ec.build_full_docs(record)
    kept, withheld = setting2._split_kept_withheld(
        evidence_doc_ids, args.keep_ratio, args.min_keep, args.min_withhold,
    )
    withheld_set = set(withheld)
    initial_pool = [doc_id for doc_id in full_docs.keys()
                    if doc_id not in withheld_set]
    initial_ordered = setting2._initial_ordered_docs(
        initial_pool, [], _record_id(record), args.seed,
    )

    print("\n=== dry-run: selected_id ===", _record_id(record))
    print("=== dry-run: buckets ===")
    print(json.dumps({k: len(v) for k, v in buckets.items()},
                     ensure_ascii=False, indent=2))
    print("\n=== dry-run: evidence_doc_ids ===",
          json.dumps(evidence_doc_ids, ensure_ascii=False))
    print("=== dry-run: kept_golden ===",
          json.dumps(kept, ensure_ascii=False))
    print("=== dry-run: withheld_golden ===",
          json.dumps(withheld, ensure_ascii=False))
    print("=== dry-run: initial_ordered ===",
          json.dumps(initial_ordered, ensure_ascii=False))

    n_withheld = len(withheld)
    if max_searches_cfg["mode"] == "fixed":
        effective_max = int(max_searches_cfg["fixed"])
        print(f"=== dry-run: max_model_searches ===  "
              f"fixed mode -> effective_max={effective_max}")
    else:
        effective_max = setting2.derive_max_model_searches(
            n_withheld=n_withheld,
            top_k=args.search_top_k,
            safety=max_searches_cfg["safety"],
            floor=max_searches_cfg["floor"],
            cap=max_searches_cfg["cap"],
        )
        raw_est = (
            math.ceil(max_searches_cfg["safety"] * n_withheld
                      / max(1, args.search_top_k))
            if n_withheld > 0 else max_searches_cfg["floor"]
        )
        print(f"=== dry-run: max_model_searches ===  "
              f"auto mode  -> n_withheld={n_withheld}, "
              f"top_k={args.search_top_k}, "
              f"safety={max_searches_cfg['safety']}, "
              f"raw_ceil={raw_est}, "
              f"clip[{max_searches_cfg['floor']},{max_searches_cfg['cap']}] "
              f"-> effective_max={effective_max}")

    retrieved = ec.simple_retrieve(
        record.get("query", ""), list(full_docs.keys()), full_docs,
        top_k=args.search_top_k, tie_break_seed=args.seed,
    )
    print("=== dry-run: simple_retrieve(top-k using original query) ===",
          json.dumps(retrieved, ensure_ascii=False))
    print("\n=== dry-run: sample Turn 0 messages ===")
    messages, _ = setting2.build_turn0_messages(
        record.get("query", ""), initial_ordered, full_docs, args.max_doc_chars,
    )
    print(json.dumps(messages, ensure_ascii=False, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DelegSearchBench Setting II evaluator filtered by item id "
            "(Partial-Context, search enabled)"
        ),
    )
    ec.add_common_args(parser)
    _add_setting2_args(parser)
    parser.add_argument(
        "--ids-file",
        default=str(DEFAULT_IDS_FILE),
        help=("待评测 item id 列表, 每行一个 id。默认使用 "
              "data/setting2_ids.txt"),
    )
    parser.add_argument(
        "--ids",
        default="",
        help="额外指定的 item id, 逗号分隔; 会与 --ids-file 合并去重。",
    )
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = str(ec._PKG_ROOT / "results" / "setting2_byid")

    _validate_args(parser, args)
    ec.set_qpm(args.qpm)

    models = ec.parse_models(args.models)
    ids_from_file = _load_ids_file(args.ids_file)
    target_ids = _merge_ids(ids_from_file, _parse_inline_ids(args.ids))
    if not target_ids:
        parser.error("--ids-file / --ids 没有提供任何 item id")

    all_records = ec.load_jsonl(args.input)
    available_modes = sorted({
        (r.get("shaping_mode") or "") for r in all_records
        if r.get("shaping_mode")
    })
    if args.only_mode:
        requested = [
            mode.strip() for mode in args.only_mode.split(",") if mode.strip()
        ]
        unknown = [mode for mode in requested if mode not in available_modes]
        if unknown:
            print(
                "[WARN] --only-mode 中含未在数据里出现的 mode: "
                f"{unknown}. 已知 mode: {available_modes}. "
                "这些将过滤到 0 条。",
                file=sys.stderr,
            )

    records, missing_ids = _filter_records_by_ids(all_records, target_ids)
    selected_before_mode = len(records)
    records = ec.filter_records_by_mode(records, args.only_mode)
    total_loaded = len(records)
    records = ec.apply_limits(records, args.limit, args.limit_per_mode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = str(output_dir / "raw.jsonl")

    done_keys = ec.load_done_keys(
        raw_path, key_field="eval_key", retry_failures=args.retry_failures,
    )
    n_retriable = (
        ec.count_retriable_rows(raw_path)
        if args.retry_failures else 0
    )

    max_searches_cfg, max_searches_info = _build_max_searches_cfg(args)

    print(f"[INFO] input           : {args.input}")
    print(f"[INFO] output_dir      : {output_dir}")
    print(f"[INFO] ids_file        : {args.ids_file}")
    print(
        f"[INFO] target_ids      : requested={len(target_ids)}, "
        f"matched={selected_before_mode}, missing={len(missing_ids)}"
    )
    if missing_ids:
        print(
            "[WARN] ids not found in input (first 20): "
            f"{missing_ids[:20]}",
            file=sys.stderr,
        )
    print(f"[INFO] models          : {models}")
    print(f"[INFO] judge           : {args.judge}")
    print(
        f"[INFO] n_runs={args.n_runs}, temperature={args.temperature}, "
        f"judge_temperature={args.judge_temperature}, seed={args.seed}"
    )
    print(
        f"[INFO] keep_ratio={args.keep_ratio}, "
        f"min_keep={args.min_keep}, min_withhold={args.min_withhold}, "
        f"search_top_k={args.search_top_k}, "
        f"max_model_searches={max_searches_info}, "
        f"search_dedup={args.search_dedup}"
    )

    mode_dist_before = Counter(
        (r.get("shaping_mode") or "?") for r in all_records
    )
    mode_dist_after = Counter(
        (r.get("shaping_mode") or "?") for r in records
    )
    print(f"[INFO] available_modes : {sorted(mode_dist_before.keys())}")
    if args.only_mode:
        print(f"[INFO] --only-mode      : {args.only_mode}")
    per_mode_str = ", ".join(
        f"{mode}={mode_dist_after.get(mode, 0)}"
        for mode in sorted(mode_dist_before.keys())
    )
    print(f"[INFO] mode_distribution: {per_mode_str}")
    print(
        f"[INFO] matched_after_mode={total_loaded}, "
        f"to_process={len(records)}, done_evaluations={len(done_keys)}"
    )
    if args.retry_failures and n_retriable:
        print(
            f"[INFO] retry_failures=True: {n_retriable} historical rows "
            "will be re-executed (failure!=None or judge_label==judge_failed)"
        )

    _print_budget_preview(records, args, max_searches_cfg)

    if args.dry_run:
        _dry_run(records, args, max_searches_cfg)
        return

    client = ec.build_client(base_url=args.base_url, api_key=args.api_key)
    judge_client = ec.build_client(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key or ec.DEFAULT_JUDGE_API_KEY,
    )

    def _worker(record: Dict[str, Any]) -> Tuple[str, int, Optional[str]]:
        rid = _record_id(record) or "?"
        try:
            n = setting2.process_sample(
                client, judge_client, record, models,
                n_runs=args.n_runs,
                temperature=args.temperature,
                judge_temperature=args.judge_temperature,
                seed=args.seed,
                keep_ratio=args.keep_ratio,
                min_keep=args.min_keep,
                min_withhold=args.min_withhold,
                search_top_k=args.search_top_k,
                max_doc_chars=args.max_doc_chars,
                max_searches_cfg=max_searches_cfg,
                search_dedup=args.search_dedup,
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
        futs = [pool.submit(_worker, record) for record in records]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="setting2_byid"):
            rid, n, err = fut.result()
            total_written += n
            if err:
                print(f"[ERR] sample {rid}: {err}", file=sys.stderr)

    print(f"[DONE] setting2_byid: wrote {total_written} new rows -> {raw_path}")
    print("       run: python research/DelegSearch/eval/summarize.py --setting 2 "
          f"--input-dir {output_dir}")


if __name__ == "__main__":
    main()
