#!/usr/bin/env python3
"""

Summarize the Setting I / Setting II evaluation metrics by shaping_mode.

Usage:
    python research/DelegSearch/eval/summarize.py \
        --setting 1 \
        --input-dir research/DelegSearch/eval/results/setting1

    python research/DelegSearch/eval/summarize.py \
        --setting 2 \
        --input-dir research/DelegSearch/eval/results/setting2_byid

Setting I output (one row per shaping_mode × model):
    mode, sample_count, model,
    first_pass1, first_pass3,
    middle_pass1, middle_pass3,
    last_pass1, last_pass3,
    overall_pass1, position_robustness

Setting II output (one row per shaping_mode × model):
    mode, sample_count, model,
    search_rate, direct_answer_rate, abstain_rate,
    pass1, pass_k,
    retrieval_hit_rate,
    correct_given_search, correct_given_direct,
    avg_searches, terminated_at_max_rate,
    retrieval_hit_any_rate, correct_rate_given_terminated

pass1 is the mean accuracy per run (correct counts as 1);
pass_k is the proportion of samples with at least one correct result in k runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VARIANT_NAMES = ("golden_first", "golden_middle", "golden_last")
CORRECT_LABEL = "correct"


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_int(x: Optional[int]) -> str:
    return "0" if x is None else str(x)


def load_raw(path: Path) -> List[Dict[str, Any]]:
    """按 eval_key 去重取最新一次落盘的记录。

    因为 --retry-failures 会让同一个 eval_key 在 raw.jsonl 里出现旧+新两行,
    这里用 dict 保序覆盖 (相同 key 的后写者胜出), 保证下游统计不会重复计数。
    没有 eval_key 的兜底: 用行号造一个唯一占位, 与其它行不冲突。

    Deduplicate by eval_key and retain the latest persisted record.

    Because --retry-failures can leave both old and new raw.jsonl rows for the
    same eval_key, overwrite values in an insertion-ordered dict (the later
    row wins) so downstream statistics do not double-count records. As a
    fallback for rows without eval_key, create a unique placeholder from the
    row number that cannot conflict with other rows.
    """
    if not path.exists():
        raise SystemExit(f"[ERR] raw.jsonl 不存在: {path}")
    by_key: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            k = obj.get("eval_key")
            if not isinstance(k, str):
                k = f"__row_{idx}__"
            by_key[k] = obj  # 覆盖旧值 = 保留最新
            # Overwrite the old value to retain the latest one.
    return list(by_key.values())


# ============================================================================
# Setting I
# ============================================================================

class _S1Cell:
    """一个 (mode, model, variant) 的聚合桶.

    An aggregation bucket for one (mode, model, variant) tuple.
    """
    def __init__(self) -> None:
        # run-level (pass@1)
        self.correct_runs = 0
        self.total_runs = 0
        # sample-level (pass@k): 对每个 sample_id 记录是否 "至少一次 correct"
        # Sample-level (pass@k): record whether each sample_id is correct at least once.
        self.per_sample_any_correct: Dict[str, bool] = {}

    def add_run(self, sample_id: str, label: str) -> None:
        self.total_runs += 1
        is_correct = label == CORRECT_LABEL
        if is_correct:
            self.correct_runs += 1
        prev = self.per_sample_any_correct.get(sample_id, False)
        self.per_sample_any_correct[sample_id] = prev or is_correct

    def pass1(self) -> Optional[float]:
        return self.correct_runs / self.total_runs if self.total_runs else None

    def passk(self) -> Optional[float]:
        n = len(self.per_sample_any_correct)
        if n == 0:
            return None
        return sum(1 for v in self.per_sample_any_correct.values() if v) / n


def summarize_setting1(
    raw_path: Path, csv_out: Path,
) -> None:
    records = load_raw(raw_path)
    # (mode, model, variant) -> _S1Cell
    cells: Dict[Tuple[str, str, str], _S1Cell] = defaultdict(_S1Cell)
    # (mode) -> set(sample_id)
    sample_count_by_mode: Dict[str, set] = defaultdict(set)
    # 顺序稳定
    # Keep the order stable.
    models_order: List[str] = []
    seen_models: set = set()

    for r in records:
        mode = r.get("shaping_mode") or "unknown"
        model = r.get("model")
        variant = r.get("variant")
        sample_id = r.get("id") or r.get("data_id")
        label = r.get("judge_label")
        if not (model and variant and sample_id):
            continue
        key = (mode, model, variant)
        cells[key].add_run(str(sample_id), label or "wrong")
        sample_count_by_mode[mode].add(str(sample_id))
        if model not in seen_models:
            seen_models.add(model)
            models_order.append(model)

    modes = sorted(sample_count_by_mode.keys())
    header = [
        "mode", "sample_count", "model",
        "first_pass1", "first_pass3",
        "middle_pass1", "middle_pass3",
        "last_pass1", "last_pass3",
        "overall_pass1", "position_robustness",
    ]

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for mode in modes:
            n_samples = len(sample_count_by_mode[mode])
            print(f"\n=== [Setting I] mode={mode} "
                  f"(samples={n_samples}) ===")
            print(", ".join(header))
            for model in models_order:
                per_variant_p1: List[Optional[float]] = []
                per_variant_pk: List[Optional[float]] = []
                for v in VARIANT_NAMES:
                    c = cells.get((mode, model, v))
                    if c is None:
                        per_variant_p1.append(None)
                        per_variant_pk.append(None)
                    else:
                        per_variant_p1.append(c.pass1())
                        per_variant_pk.append(c.passk())
                valid_p1 = [x for x in per_variant_p1 if x is not None]
                overall_p1 = (
                    sum(valid_p1) / len(valid_p1) if valid_p1 else None
                )
                robustness = (
                    max(valid_p1) - min(valid_p1)
                    if len(valid_p1) >= 2 else None
                )
                row = [
                    mode, str(n_samples), model,
                    _fmt(per_variant_p1[0]), _fmt(per_variant_pk[0]),
                    _fmt(per_variant_p1[1]), _fmt(per_variant_pk[1]),
                    _fmt(per_variant_p1[2]), _fmt(per_variant_pk[2]),
                    _fmt(overall_p1), _fmt(robustness),
                ]
                w.writerow(row)
                print(", ".join(row))
    print(f"\n[OK] saved -> {csv_out}")


# ============================================================================
# Setting II
# ============================================================================

class _S2Cell:
    def __init__(self) -> None:
        self.total_runs = 0
        self.correct_runs = 0
        self.action_counts: Dict[str, int] = defaultdict(int)
        # 分子 = search_refine 且 retrieval_hit=True 的次数
        # Numerator: number of search_refine runs with retrieval_hit=True.
        # 分母 = search_refine 的次数
        # Denominator: number of search_refine runs.
        self.search_refine_runs = 0
        self.retrieval_hits = 0
        self.correct_given_search = 0
        self.correct_given_direct = 0
        self.direct_answer_runs = 0
        # 多轮 agent loop 新增计数
        # Additional counters for the multi-turn agent loop.
        # action_summary 细分 (direct_answer / search_and_commit /
        # action_summary breakdown (direct_answer / search_and_commit /
        # terminated_at_max / abstain)
        self.action_summary_counts: Dict[str, int] = defaultdict(int)
        self.n_searches_sum = 0            # 累积 n_searches, 用于 avg_searches
        # Accumulated n_searches, used for avg_searches.
        self.n_searches_available = 0      # 有 n_searches 字段的 run 数
        # Number of runs with an n_searches field.
        self.terminated_at_max_runs = 0
        self.correct_given_terminated = 0
        # retrieval_hit_any: 至少一轮 search 命中被隐藏 golden 的 run 数
        # retrieval_hit_any: runs where at least one search hit the hidden golden item.
        # (只在 n_searches > 0 的子集内累计)
        # Accumulate only within the subset where n_searches > 0.
        self.runs_with_search = 0          # n_searches > 0 的 run 数
        # Number of runs where n_searches > 0.
        self.retrieval_hit_any_runs = 0
        # sample-level pass@k
        self.per_sample_any_correct: Dict[str, bool] = {}

    def add_run(
        self,
        sample_id: str,
        label: str,
        action: str,
        retrieval_hit: bool,
        *,
        action_summary: Optional[str] = None,
        n_searches: Optional[int] = None,
        retrieval_hit_any: Optional[bool] = None,
    ) -> None:
        self.total_runs += 1
        is_correct = label == CORRECT_LABEL
        if is_correct:
            self.correct_runs += 1
        self.action_counts[action] += 1
        if action == "search_refine":
            self.search_refine_runs += 1
            if retrieval_hit:
                self.retrieval_hits += 1
            if is_correct:
                self.correct_given_search += 1
        elif action == "direct_answer":
            self.direct_answer_runs += 1
            if is_correct:
                self.correct_given_direct += 1
        prev = self.per_sample_any_correct.get(sample_id, False)
        self.per_sample_any_correct[sample_id] = prev or is_correct

        # ---- 多轮 agent loop 新指标 ----
        # ---- New metrics for the multi-turn agent loop ----
        if action_summary is not None:
            self.action_summary_counts[action_summary] += 1
            if action_summary == "terminated_at_max":
                self.terminated_at_max_runs += 1
                if is_correct:
                    self.correct_given_terminated += 1
        if n_searches is not None:
            self.n_searches_sum += n_searches
            self.n_searches_available += 1
            if n_searches > 0:
                self.runs_with_search += 1
                if retrieval_hit_any or retrieval_hit:
                    # 老数据没有 retrieval_hit_any, 回退到 retrieval_hit
                    # Older data lacks retrieval_hit_any; fall back to retrieval_hit.
                    self.retrieval_hit_any_runs += 1

    def _rate(self, num: int, den: int) -> Optional[float]:
        return num / den if den > 0 else None

    def pass1(self) -> Optional[float]:
        return self._rate(self.correct_runs, self.total_runs)

    def passk(self) -> Optional[float]:
        n = len(self.per_sample_any_correct)
        if n == 0:
            return None
        return sum(1 for v in self.per_sample_any_correct.values() if v) / n

    def search_rate(self) -> Optional[float]:
        return self._rate(self.search_refine_runs, self.total_runs)

    def direct_rate(self) -> Optional[float]:
        return self._rate(self.direct_answer_runs, self.total_runs)

    def abstain_rate(self) -> Optional[float]:
        abstain = (
            self.total_runs - self.search_refine_runs
            - self.direct_answer_runs
        )
        return self._rate(abstain, self.total_runs)

    def retrieval_hit_rate(self) -> Optional[float]:
        return self._rate(self.retrieval_hits, self.search_refine_runs)

    def correct_given_search_rate(self) -> Optional[float]:
        return self._rate(self.correct_given_search, self.search_refine_runs)

    def correct_given_direct_rate(self) -> Optional[float]:
        return self._rate(self.correct_given_direct, self.direct_answer_runs)

    # ---- 多轮 agent loop 新增指标 ----
    # ---- Additional metrics for the multi-turn agent loop ----
    def avg_searches(self) -> Optional[float]:
        if self.n_searches_available == 0:
            return None
        return self.n_searches_sum / self.n_searches_available

    def terminated_at_max_rate(self) -> Optional[float]:
        return self._rate(self.terminated_at_max_runs, self.total_runs)

    def retrieval_hit_any_rate(self) -> Optional[float]:
        return self._rate(self.retrieval_hit_any_runs, self.runs_with_search)

    def correct_rate_given_terminated(self) -> Optional[float]:
        return self._rate(
            self.correct_given_terminated, self.terminated_at_max_runs,
        )


def summarize_setting2(
    raw_path: Path, csv_out: Path,
) -> None:
    records = load_raw(raw_path)
    cells: Dict[Tuple[str, str], _S2Cell] = defaultdict(_S2Cell)
    sample_count_by_mode: Dict[str, set] = defaultdict(set)
    models_order: List[str] = []
    seen_models: set = set()

    for r in records:
        mode = r.get("shaping_mode") or "unknown"
        model = r.get("model")
        sample_id = r.get("id") or r.get("data_id")
        label = r.get("judge_label") or "wrong"
        action = r.get("action_taken") or "abstain"
        retrieval_hit = bool(r.get("retrieval_hit"))
        action_summary = r.get("action_summary")
        # n_searches: 老数据没有, 从 action_taken 反推 (search_refine → 1)
        # n_searches: infer from action_taken for older data (search_refine → 1).
        n_searches_val = r.get("n_searches")
        if n_searches_val is None and "action_taken" in r:
            n_searches_val = 1 if action == "search_refine" else 0
        retrieval_hit_any = r.get("retrieval_hit_any")
        if retrieval_hit_any is None:
            retrieval_hit_any = retrieval_hit
        if not (model and sample_id):
            continue
        cells[(mode, model)].add_run(
            str(sample_id), label, action, retrieval_hit,
            action_summary=action_summary,
            n_searches=(
                int(n_searches_val)
                if isinstance(n_searches_val, (int, float))
                else None
            ),
            retrieval_hit_any=bool(retrieval_hit_any),
        )
        sample_count_by_mode[mode].add(str(sample_id))
        if model not in seen_models:
            seen_models.add(model)
            models_order.append(model)

    modes = sorted(sample_count_by_mode.keys())
    header = [
        "mode", "sample_count", "model",
        "search_rate", "direct_answer_rate", "abstain_rate",
        "pass1", "pass_k",
        "retrieval_hit_rate",
        "correct_given_search", "correct_given_direct",
        # ---- 多轮 agent loop 新列 ----
        # ---- New columns for the multi-turn agent loop ----
        "avg_searches", "terminated_at_max_rate",
        "retrieval_hit_any_rate", "correct_rate_given_terminated",
    ]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for mode in modes:
            n_samples = len(sample_count_by_mode[mode])
            print(f"\n=== [Setting II] mode={mode} "
                  f"(samples={n_samples}) ===")
            print(", ".join(header))
            for model in models_order:
                c = cells.get((mode, model))
                if c is None:
                    row = [
                        mode, str(n_samples), model,
                    ] + ["n/a"] * (len(header) - 3)
                else:
                    row = [
                        mode, str(n_samples), model,
                        _fmt(c.search_rate()),
                        _fmt(c.direct_rate()),
                        _fmt(c.abstain_rate()),
                        _fmt(c.pass1()),
                        _fmt(c.passk()),
                        _fmt(c.retrieval_hit_rate()),
                        _fmt(c.correct_given_search_rate()),
                        _fmt(c.correct_given_direct_rate()),
                        _fmt(c.avg_searches()),
                        _fmt(c.terminated_at_max_rate()),
                        _fmt(c.retrieval_hit_any_rate()),
                        _fmt(c.correct_rate_given_terminated()),
                    ]
                w.writerow(row)
                print(", ".join(row))
    print(f"\n[OK] saved -> {csv_out}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Setting I / Setting II raw.jsonl by shaping_mode",
    )
    parser.add_argument("--setting", type=int, choices=[1, 2], required=True)
    parser.add_argument("--input-dir", required=True,
                        help="包含 raw.jsonl 的目录")
    parser.add_argument("--csv", default="",
                        help="输出 CSV 路径; 默认 <input-dir>/summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    raw_path = input_dir / "raw.jsonl"
    csv_out = Path(args.csv) if args.csv else input_dir / "summary.csv"

    if args.setting == 1:
        summarize_setting1(raw_path, csv_out)
    else:
        raw_path = input_dir / "raw.jsonl"
        summarize_setting2(raw_path, csv_out)


if __name__ == "__main__":
    main()
