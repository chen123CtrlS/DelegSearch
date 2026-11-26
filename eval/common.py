"""
Shared evaluation infrastructure for DelegSearchBench (reused by both
Setting I and Setting II).

This module follows the evaluation protocol in the released DelegSearchBench
data (research/DelegSearch/data/) and readme.md:
  * Setting I  (Full-Context / search disabled)   — see setting1.py
  * Setting II (Partial-Context / search enabled) — see setting2_byid.py

This module provides utility functions only and has no CLI entry point.
Design principles:
  1. Remain fully decoupled from D_DifficultyCalibrator (do not import other
     stages), while using the same integration patterns (OpenAI client, QPM
     rate limiting, doc-order variants, and judge rubric) so evaluation
     metrics can be compared against the Stage D report on the same baseline.
  2. In Setting II, simulate `search` with controlled closed-set retrieval
     instead of real external search. This keeps runs reproducible and
     offline while testing refined-query quality: a poor query retrieves the
     wrong documents from {golden ∪ distractor ∪ irrelevant}, causing an
     incorrect answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import json_repair
except ImportError:  # pragma: no cover
    json_repair = None  # type: ignore

try:
    from openai import OpenAI
except ImportError as _openai_err:  # pragma: no cover
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None

_PKG_ROOT = Path(__file__).resolve().parent
_DELEGSEARCH_ROOT = _PKG_ROOT.parent
_DEFAULT_DATA = _DELEGSEARCH_ROOT / "data" / "example.jsonl"
_DEFAULT_SETTING2_IDS = _DELEGSEARCH_ROOT / "data" / "setting2_ids.txt"
_ENV_PATH = _DELEGSEARCH_ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    # requirements 未安装时也能读取简单的 KEY=VALUE 配置。
    # Read simple KEY=VALUE settings even when requirements are not installed.
    if _ENV_PATH.exists():
        for raw_line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


# ============================================================================
# 默认配置 (可被 CLI 覆盖)
# Default configuration (overridable via the CLI)
# ============================================================================

DEFAULT_BASE_URL = os.getenv(
    "EVAL_BASE_URL", "",
)
DEFAULT_API_KEY = os.getenv("EVAL_API_KEY", os.getenv("API_KEY", ""))
DEFAULT_JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", DEFAULT_BASE_URL)
DEFAULT_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", DEFAULT_API_KEY)


# 目标模型与 judge 均可在 DelegSearch/.env 中配置。
# Both target models and the judge can be configured in DelegSearch/.env.
DEFAULT_MODELS: Tuple[str, ...] = tuple(
    model.strip()
    for model in os.getenv(
        "EVAL_MODELS",
        "hunyuan-3-preview,deepseek-v4-pro,qwen3.5-flash",
    ).split(",")
    if model.strip()
)
DEFAULT_JUDGE_MODEL = (
    os.getenv(
        "JUDGE_MODEL",
        os.getenv("EVAL_JUDGE_MODEL", "deepseek-v4-flash"),
    ).strip()
    or "deepseek-v4-flash"
)

DEFAULT_N_RUNS = 3
DEFAULT_TEMPERATURE = 1.0
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_SEED = 20260702
DEFAULT_VARIANT_SEED = 20260702

DEFAULT_WORKERS = 32
DEFAULT_QPM = 1500
DEFAULT_MAX_RETRIES = 3

VARIANT_NAMES: Tuple[str, ...] = (
    "golden_first", "golden_middle", "golden_last",
)


# ============================================================================
# QPM 限速 (本进程共享一个 window, 与 Stage D 无耦合)
# QPM rate limiting (one process-wide window, decoupled from Stage D)
# ============================================================================

_rate_lock = threading.Lock()
_rate_timestamps: List[float] = []
_current_qpm = DEFAULT_QPM


def set_qpm(qpm: int) -> None:
    global _current_qpm
    _current_qpm = max(1, int(qpm))


def rate_limit_acquire() -> None:
    with _rate_lock:
        now = time.time()
        while _rate_timestamps and _rate_timestamps[0] < now - 60:
            _rate_timestamps.pop(0)
        if len(_rate_timestamps) >= _current_qpm:
            wait = _rate_timestamps[0] + 60 - now
            if wait > 0:
                time.sleep(wait)
            now = time.time()
            _rate_timestamps.pop(0)
        _rate_timestamps.append(now)


# ============================================================================
# OpenAI 兼容 client + seed 兼容层
# OpenAI-compatible client and seed compatibility layer
# ============================================================================

_models_without_seed: set = set()
_seed_cache_lock = threading.Lock()


def _safe_json_loads(text: str) -> Any:
    if json_repair is not None:
        return json_repair.loads(text)
    return json.loads(text)


def build_client(
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
) -> OpenAI:
    if OpenAI is None:
        raise ImportError(
            "openai package is required. Install: pip install openai"
        ) from _OPENAI_IMPORT_ERROR
    return OpenAI(
        base_url=base_url,
        api_key=api_key or DEFAULT_API_KEY,
    )


def _looks_like_seed_unsupported(err: Exception) -> bool:
    msg = str(err).lower()
    return "seed" in msg and (
        "unsupported" in msg or "invalid" in msg or "not supported" in msg
    )


def create_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = None,
    tool_choice: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    """chat.completions.create 的统一封装, 处理 seed 兼容 + 简单重试。

    Unified wrapper for chat.completions.create with seed compatibility and
    basic retries.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            rate_limit_acquire()
            if seed is not None and model not in _models_without_seed:
                try:
                    return client.chat.completions.create(seed=seed, **kwargs)
                except TypeError:
                    with _seed_cache_lock:
                        _models_without_seed.add(model)
                    return client.chat.completions.create(**kwargs)
                except Exception as e:
                    if _looks_like_seed_unsupported(e):
                        with _seed_cache_lock:
                            _models_without_seed.add(model)
                        return client.chat.completions.create(**kwargs)
                    raise
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e
            print(
                f"[WARN] chat.completions attempt {attempt + 1}/"
                f"{max_retries} failed on {model}: {e}",
                file=sys.stderr,
            )
            time.sleep(min(2 ** attempt, 10))
    assert last_exc is not None
    raise last_exc


# ============================================================================
# JSONL I/O + 断点续跑
# JSONL I/O and resumable execution
# ============================================================================

_write_lock = threading.Lock()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] bad JSON line, skip: {e}", file=sys.stderr)
    return records


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def _row_is_retriable_failure(obj: Dict[str, Any]) -> bool:
    """是否算 "上次没跑成功" (--retry-failures 会重跑这些行)。

    判定条件 (任一即算):
      * 顶层 `failure` 非空字符串 (基础设施/API 报错, 如 call_failed:...);
      * `judge_label` 为 None / 空 / "judge_failed" (judge 本身失败)。
    注意: `judge_label == "wrong"` 是模型答错, **不算** 失败, 不会被重跑。

    Whether a row counts as an unsuccessful previous run and should be rerun
    by --retry-failures.

    A row qualifies if either condition holds:
      * top-level `failure` is a nonempty string (an infrastructure/API error,
        such as call_failed:...);
      * `judge_label` is None, empty, or "judge_failed" (the judge failed).
    Note: `judge_label == "wrong"` means the model answered incorrectly. It is
    **not** a failure and will not be rerun.
    """
    fail = obj.get("failure")
    if fail is not None and isinstance(fail, str) and fail.strip():
        return True
    if isinstance(fail, str) is False and fail:
        # 非 str 的真值 (极少见, 但保险起见)
        # Handle a truthy non-string value (rare, but safer).
        return True
    jlab = obj.get("judge_label")
    if jlab in (None, "", "judge_failed"):
        return True
    return False


def _load_last_row_per_key(
    path: str, key_field: str = "eval_key",
) -> Dict[str, Dict[str, Any]]:
    """扫 raw.jsonl, 返回 {eval_key: 最后一次写盘的 row}。

    是 `load_done_keys` 与 `count_retriable_rows` 的公共底层, 保证两者
    对"key 是否已完成 / 需要重跑"的判定基于同一份"最新状态", 避免同一 key
    出现旧+新两行时统计口径不一致。

    Scan raw.jsonl and return {eval_key: the most recently persisted row}.

    This shared foundation for `load_done_keys` and `count_retriable_rows`
    ensures both decide whether a key is complete or needs rerunning from the
    same latest state, avoiding inconsistent counts when a key has both old
    and new rows.
    """
    last: Dict[str, Dict[str, Any]] = {}
    if not Path(path).exists():
        return last
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            k = obj.get(key_field)
            if isinstance(k, str):
                last[k] = obj  # 覆盖旧值 = 保留最新
                # Overwrite the old value to retain the latest one.
    return last


def load_done_keys(
    path: str,
    key_field: str = "eval_key",
    retry_failures: bool = False,
) -> set:
    """从已存在的 raw.jsonl 提取已完成的 eval_key 集合, 用于断点续跑。

    eval_key 由脚本定义 (Setting I: id::model::variant::run;
    Setting II: id::model::run), 保证 (sample, model, variant/run) 幂等。

    当 `retry_failures=True` 时, 依据**每个 eval_key 的最后一次落盘状态**
    判断是否要重跑 (判定见 `_row_is_retriable_failure`) —— 这样即便同一 key
    出现"失败→成功"或"失败→成功→失败"多行, 也总是尊重最新一次的结果。

    Extract completed eval_keys from an existing raw.jsonl for resumable runs.

    Each script defines its eval_key (Setting I: id::model::variant::run;
    Setting II: id::model::run), ensuring idempotency for each
    (sample, model, variant/run) combination.

    When `retry_failures=True`, rerun decisions use the **last persisted state
    of each eval_key** (see `_row_is_retriable_failure`). Thus, even when a key
    has "failure→success" or "failure→success→failure" rows, the latest result
    always takes precedence.
    """
    if not Path(path).exists():
        return set()
    last_row = _load_last_row_per_key(path, key_field=key_field)
    if not retry_failures:
        return set(last_row.keys())
    return {
        k for k, row in last_row.items()
        if not _row_is_retriable_failure(row)
    }


def count_retriable_rows(path: str, key_field: str = "eval_key") -> int:
    """启用 --retry-failures 时会被重跑的 eval_key 数 (按最新状态判定)。

    用于启动时的 [INFO] 打印。与 `load_done_keys(retry_failures=True)`
    共享同一个 last-row 快照, 保证两者一致。

    Count eval_keys that --retry-failures will rerun, based on latest state.

    Used for the startup [INFO] message. It shares the same last-row snapshot
    as `load_done_keys(retry_failures=True)` to keep both results consistent.
    """
    if not Path(path).exists():
        return 0
    last_row = _load_last_row_per_key(path, key_field=key_field)
    return sum(1 for row in last_row.values()
               if _row_is_retriable_failure(row))


# ============================================================================
# Doc 构造 / evidence 锚点 / doc-order 变体
# Document construction, evidence anchors, and doc-order variants
# ============================================================================

def iter_evidence_doc_ids(record: Dict[str, Any]) -> List[str]:
    """按 evidence_spans 出现顺序抽取 doc_id, 回退到 golden_docs.keys()。

    Extract doc_ids in evidence_spans order, falling back to
    golden_docs.keys().
    """
    golden = record.get("golden_docs") or {}
    seen: set = set()
    out: List[str] = []
    for span in record.get("evidence_spans") or []:
        if not isinstance(span, dict):
            continue
        did = span.get("doc_id")
        if isinstance(did, str) and did in golden and did not in seen:
            out.append(did)
            seen.add(did)
    if not out:
        for did in golden.keys():
            if isinstance(did, str) and did not in seen:
                out.append(did)
                seen.add(did)
    return out


def build_full_docs(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, List[str]]]:
    """合并 golden ∪ distractor ∪ irrelevant 为一个 full_docs dict。

    Returns:
      full_docs        : {doc_id: doc_info}
      evidence_doc_ids : Setting I variant 锚点 (按 evidence_spans 顺序)
      buckets          : {"golden": [...], "distractor": [...],
                          "irrelevant": [...]}

    Merge golden ∪ distractor ∪ irrelevant into one full_docs dictionary.

    Returns:
      full_docs        : {doc_id: doc_info}
      evidence_doc_ids : Setting I variant anchors (in evidence_spans order)
      buckets          : {"golden": [...], "distractor": [...],
                          "irrelevant": [...]}
    """
    golden = record.get("golden_docs") or {}
    distractor = record.get("distractor_docs") or {}
    irrelevant = record.get("irrelevant_docs") or {}

    full_docs: Dict[str, Dict[str, Any]] = {}
    buckets: Dict[str, List[str]] = {
        "golden": [], "distractor": [], "irrelevant": [],
    }

    def _merge(name: str, src: Dict[str, Any]) -> None:
        for doc_id, info in (src or {}).items():
            if not isinstance(doc_id, str):
                continue
            if doc_id in full_docs:
                # 极少数上游 corner case: 直接跳过, 保留先入者。
                # Rare upstream corner case: skip it and retain the first item.
                continue
            full_docs[doc_id] = info if isinstance(info, dict) else {}
            buckets[name].append(doc_id)

    _merge("golden", golden)
    _merge("distractor", distractor)
    _merge("irrelevant", irrelevant)

    evidence_doc_ids = iter_evidence_doc_ids(record)
    # 保证锚点全在 full_docs 里
    # Ensure every anchor is present in full_docs.
    evidence_doc_ids = [d for d in evidence_doc_ids if d in full_docs]

    return full_docs, evidence_doc_ids, buckets


def generate_doc_orders(
    full_docs: Dict[str, Dict[str, Any]],
    evidence_doc_ids: Sequence[str],
    variant_seed: int = DEFAULT_VARIANT_SEED,
    variant_names: Sequence[str] = VARIANT_NAMES,
) -> List[Tuple[str, List[str]]]:
    """生成 doc-order 三变体 (与 Stage D 同构):
        golden_first  = evidence + shuffled non_evidence
        golden_middle = ne[:half] + evidence + ne[half:]
        golden_last   = shuffled non_evidence + evidence

    Generate three doc-order variants (isomorphic to Stage D):
        golden_first  = evidence + shuffled non_evidence
        golden_middle = ne[:half] + evidence + ne[half:]
        golden_last   = shuffled non_evidence + evidence
    """
    all_ids = list(full_docs.keys())
    ev_set = set(evidence_doc_ids)
    non_evidence = [d for d in all_ids if d not in ev_set]
    evidence = [d for d in evidence_doc_ids if d in full_docs]

    if not evidence:
        variants: List[Tuple[str, List[str]]] = []
        for idx, name in enumerate(variant_names):
            rng = random.Random(variant_seed + 1000 * (idx + 1))
            ids = list(all_ids)
            rng.shuffle(ids)
            variants.append((name, ids))
        return variants

    def _shuf(off: int) -> List[str]:
        rng = random.Random(variant_seed + off)
        ids = list(non_evidence)
        rng.shuffle(ids)
        return ids

    ne_first = _shuf(1001)
    ne_middle = _shuf(1002)
    ne_last = _shuf(1003)

    half = len(ne_middle) // 2
    return [
        ("golden_first", evidence + ne_first),
        ("golden_middle", ne_middle[:half] + evidence + ne_middle[half:]),
        ("golden_last", ne_last + evidence),
    ][: len(variant_names)]


# ============================================================================
# 文档序列化 (给模型看的文本 / tool payload)
# Document serialization (model-visible text and tool payload)
# ============================================================================

# 单文档正文的字符上限, 避免个别超长文档挤爆 context; 与 Stage D 的 tool
# payload 保持一致 (不做截断) 相比会稍微保守一点, 目的是让不同 provider 的
# context window 都能装下。可通过 --max-doc-chars CLI 覆盖。
# Character limit for each document body to prevent unusually long documents
# from exhausting the context. This is slightly more conservative than the
# untruncated Stage D tool payload so different providers' context windows can
# accommodate it. Override with the --max-doc-chars CLI option.
DEFAULT_MAX_DOC_CHARS = 4000


def _trim(text: Any, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars] + "..."


def docs_to_prompt_text(
    ordered_doc_ids: Sequence[str],
    full_docs: Dict[str, Dict[str, Any]],
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> str:
    """把 doc 集合序列化成"给 user prompt 直接读"的可读文本 (Setting I 用)。

    有意保留 url + site_name + publish_time (对 authority / publish_time
    shaping mode 是必需的信号), 但 **不暴露** authority_label
    (那是 oracle 标注, 会让模型 grep 关键词绕过判断)。

    Serialize the document collection as readable text placed directly in the
    user prompt (for Setting I).

    Deliberately retain url, site_name, and publish_time because they are
    necessary signals for authority and publish_time shaping modes, but
    **do not expose** authority_label. It is an oracle annotation that would
    let the model bypass judgment by grepping for a keyword.
    """
    blocks: List[str] = []
    for idx, doc_id in enumerate(ordered_doc_ids, start=1):
        info = full_docs.get(doc_id) or {}
        title = info.get("title") or ""
        url = info.get("url") or ""
        site = info.get("site_name") or ""
        pt = info.get("publish_time") or ""
        content = _trim(info.get("content"), max_doc_chars)
        header_parts = [f"[{doc_id}]"]
        if title:
            header_parts.append(f"title: {title}")
        if site:
            header_parts.append(f"site: {site}")
        if url:
            header_parts.append(f"url: {url}")
        if pt:
            header_parts.append(f"publish_time: {pt}")
        header = " | ".join(header_parts)
        blocks.append(f"Document {idx} {header}\n{content}")
    return "\n\n".join(blocks)


def docs_to_tool_payload(
    ordered_doc_ids: Sequence[str],
    full_docs: Dict[str, Dict[str, Any]],
    query: str,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> str:
    """把 doc 集合序列化成 role=tool 的 JSON payload (Setting II 用)。

    与 Stage D `_docs_to_tool_payload` 同构, 只保留网页元字段 + content,
    剥掉 distractor_type / target_info_point / authority_label 等内部字段。

    Serialize the document collection as a role=tool JSON payload (for
    Setting II).

    This is isomorphic to Stage D `_docs_to_tool_payload`: retain only webpage
    metadata and content, removing internal fields such as distractor_type,
    target_info_point, and authority_label.
    """
    docs_list: List[Dict[str, Any]] = []
    for doc_id in ordered_doc_ids:
        info = full_docs.get(doc_id) or {}
        docs_list.append({
            "doc_id": doc_id,
            "title": info.get("title", "") or "",
            "url": info.get("url", "") or "",
            "site_name": info.get("site_name", "") or "",
            "publish_time": info.get("publish_time", "") or "",
            "content": _trim(info.get("content"), max_doc_chars),
        })
    return json.dumps(
        {"query": query, "docs": docs_list},
        ensure_ascii=False,
    )


# ============================================================================
# 简易检索 (Setting II 的 search 工具后端)
# Simple retrieval (backend for Setting II's search tool)
# ============================================================================
#
# 说明: eval.md 只要求 "启用搜索工具", 不指定检索后端。为了做到"完全离线
# 可复现 + 能考察 refine query 质量", 这里用一个受控闭集检索:
#
#   * 候选池 = 该样本自身的 (golden ∪ distractor ∪ irrelevant) 全集
#     - 隐藏 golden 也在里面: 若模型的 refine query 写得好, 应能被召回;
#       写得差 → 被 distractor / irrelevant 挤到 top-k 之外, 答错就答错。
#   * 相似度 = query 与 (title + content) 的 token-overlap
#     - 英文按 [\w]+ 切词, 中文用 2-gram (兼顾中英文混合)
#     - 用 Jaccard-like score = 交集 tokens / query tokens (加 IDF 微权重)
#   * 完全无网络依赖, seed 固定, 结果可复现。
#
# Note: eval.md only requires enabling a search tool and does not prescribe a
# retrieval backend. To remain fully offline and reproducible while testing
# refined-query quality, this uses controlled closed-set retrieval:
#
#   * Candidate pool = all (golden ∪ distractor ∪ irrelevant) documents for
#     the sample.
#     - Hidden golden documents are included. A good refined query should
#       retrieve them; a poor one lets distractor or irrelevant documents push
#       them out of the top-k, resulting in an incorrect answer.
#   * Similarity = token overlap between the query and (title + content).
#     - Split English with [\w]+ and Chinese into 2-grams, supporting mixed
#       English and Chinese text.
#     - Use a Jaccard-like score: intersecting tokens / query tokens, with a
#       slight IDF weight.
#   * No network dependency; a fixed seed makes results reproducible.

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """英文按单词、中文按 2-gram 切分。返回小写 tokens。

    Split English into words and Chinese into 2-grams. Return lowercase tokens.
    """
    if not isinstance(text, str) or not text:
        return []
    text = text.lower()
    tokens: List[str] = []
    tokens.extend(_WORD_RE.findall(text))
    # 中文 2-gram
    # Chinese 2-grams
    cjk_re = re.compile(r"[\u4e00-\u9fff]+")
    for chunk in cjk_re.findall(text):
        for i in range(len(chunk) - 1):
            tokens.append(chunk[i : i + 2])
        # 单字兜底 (查询词只有 1 个字时)
        # Fall back to one character when the query term has only one.
        if len(chunk) == 1:
            tokens.append(chunk)
    return tokens


def _score_query_vs_doc(
    query_tokens: List[str],
    query_token_set: set,
    doc_text: str,
) -> float:
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens or not query_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    inter = query_token_set & doc_set
    if not inter:
        return 0.0
    # 覆盖率 + 简单频次奖励
    # Coverage plus a simple frequency bonus
    coverage = len(inter) / max(1, len(query_token_set))
    # doc 中出现次数 (取 log 压缩长文档偏好)
    # Occurrences in the document (use log to reduce long-document bias)
    from collections import Counter
    doc_counter = Counter(doc_tokens)
    freq = sum(doc_counter[t] for t in inter)
    freq_bonus = math.log1p(freq) / max(1.0, math.log1p(len(doc_tokens)))
    return coverage * 0.7 + freq_bonus * 0.3


def simple_retrieve(
    query: str,
    candidate_doc_ids: Sequence[str],
    full_docs: Dict[str, Dict[str, Any]],
    top_k: int = 5,
    tie_break_seed: int = 0,
) -> List[str]:
    """在 candidate_doc_ids 内基于 token 重叠对 query 做 top-k 检索。

    tie-break 用 (score, -stable_hash) 保证同分下顺序稳定可复现。
    完全无匹配的 doc 也不会被随机塞进来 (score=0 会被过滤)。

    Retrieve the top-k candidate_doc_ids for a query based on token overlap.

    Use (score, -stable_hash) as the tie-break to ensure stable, reproducible
    ordering for equal scores. Documents with no matches are not inserted
    randomly because score=0 is filtered out.
    """
    q_tokens = _tokenize(query)
    q_set = set(q_tokens)
    scored: List[Tuple[float, int, str]] = []
    for idx, doc_id in enumerate(candidate_doc_ids):
        info = full_docs.get(doc_id) or {}
        text = " ".join([
            str(info.get("title") or ""),
            str(info.get("content") or ""),
        ])
        s = _score_query_vs_doc(q_tokens, q_set, text)
        if s <= 0:
            continue
        # 稳定的 tie-break: 用 (seed, doc_id) hash
        # Stable tie-break using the hash of (seed, doc_id)
        tb = hash((tie_break_seed, doc_id)) & 0xFFFFFFFF
        scored.append((s, -tb, doc_id))
    scored.sort(reverse=True)
    return [d for _, _, d in scored[: max(0, top_k)]]


# ============================================================================
# Message 抽取工具 (extract_message_text / extract_search_tool_call)
# Message extraction utilities (extract_message_text / extract_search_tool_call)
# ============================================================================

def _blocks_to_text(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""
    parts: List[str] = []
    for blk in blocks:
        if isinstance(blk, dict):
            for k in ("text", "content", "thinking", "reasoning"):
                v = blk.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v)
                    break
        elif isinstance(blk, str):
            parts.append(blk)
    return "\n".join(p for p in parts if p)


def extract_message_text(message: Any) -> Tuple[str, Optional[str]]:
    """从 assistant message 抽 (text, source), 带 reasoning_content 回退。

    Extract (text, source) from an assistant message, falling back to
    reasoning_content.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content, "content"
    if isinstance(content, list):
        text = _blocks_to_text(content)
        if text.strip():
            return text, "content_blocks"
    for attr in ("reasoning_content", "reasoning", "thinking"):
        v = getattr(message, attr, None)
        if isinstance(v, str) and v.strip():
            return v, attr
        if isinstance(v, list):
            text = _blocks_to_text(v)
            if text.strip():
                return text, f"{attr}_blocks"
    try:
        dumped = message.model_dump() if hasattr(message, "model_dump") else None
    except Exception:
        dumped = None
    if isinstance(dumped, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = dumped.get(key)
            if isinstance(v, str) and v.strip():
                return v, f"dump_{key}"
            if isinstance(v, list):
                text = _blocks_to_text(v)
                if text.strip():
                    return text, f"dump_{key}_blocks"
    return "", None


_COMMITTED_SOURCES: frozenset = frozenset({"content", "content_blocks"})


def is_committed_content(source: Optional[str]) -> bool:
    return source in _COMMITTED_SOURCES


def extract_search_tool_call(message: Any) -> Optional[Dict[str, Any]]:
    """从 assistant message 抽第一个 name == 'search' 的 tool_call。

    Extract the first tool_call named 'search' from an assistant message.
    """
    def _coerce(cid, name, args):
        if not name or name != "search":
            return None
        return {
            "id": cid or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args or "{}"},
        }

    tcs = getattr(message, "tool_calls", None)
    if isinstance(tcs, list):
        for tc in tcs:
            fn = getattr(tc, "function", None)
            if fn is None and isinstance(tc, dict):
                fn = tc.get("function")
            if fn is None:
                continue
            name = (
                getattr(fn, "name", None)
                if not isinstance(fn, dict) else fn.get("name")
            )
            args = (
                getattr(fn, "arguments", None)
                if not isinstance(fn, dict) else fn.get("arguments")
            )
            cid = (
                getattr(tc, "id", None)
                if not isinstance(tc, dict) else tc.get("id")
            )
            c = _coerce(cid, name, args)
            if c is not None:
                return c
    try:
        dumped = message.model_dump() if hasattr(message, "model_dump") else None
    except Exception:
        dumped = None
    if isinstance(dumped, dict):
        tcs2 = dumped.get("tool_calls")
        if isinstance(tcs2, list):
            for tc in tcs2:
                if not isinstance(tc, dict):
                    continue
                fn_d = tc.get("function") or {}
                c = _coerce(
                    tc.get("id"), fn_d.get("name"), fn_d.get("arguments"),
                )
                if c is not None:
                    return c
    return None


def parse_query_arg(arguments_str: Any) -> str:
    if not isinstance(arguments_str, str):
        return ""
    try:
        parsed = _safe_json_loads(arguments_str)
    except Exception:
        return ""
    if isinstance(parsed, dict):
        q = parsed.get("query")
        if isinstance(q, str):
            return q.strip()
    return ""


def assistant_message_to_dict(
    assistant_msg: Any,
    tool_call_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    content = getattr(assistant_msg, "content", None)
    if isinstance(content, list):
        content = _blocks_to_text(content) or None
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_call_used is not None:
        msg["tool_calls"] = [tool_call_used]
    return msg


# ============================================================================
# Judge (与 Stage D 同 rubric, 精简到 category=D)
# Judge (same rubric as Stage D, limited to category=D)
# ============================================================================

JUDGE_SYSTEM = (
    "You are an impartial answer evaluator for the DelegSearchBench "
    "(a RAG-reasoning benchmark, category=\"D\": distraction discrimination).\n\n"
    "You receive:\n"
    "- query           : the user's question\n"
    "- shaping_mode    : plain / sufficiency / multi_hop / authority / "
    "publish_time / intra_doc_consistency\n"
    "- answer_type     : \"extractive\" | \"aggregative\"\n"
    "- gold_answer     : reference answer grounded in gold evidence\n"
    "- gold_evidence_doc_ids : doc_ids the gold answer is grounded in\n"
    "- model_answer    : the answer to evaluate\n"
    "- action_taken (optional) : \"direct_answer\" | \"search_refine\" | "
    "\"abstain\" (Setting II two-turn harness output)\n"
    "- refine_query (optional) : when action_taken==\"search_refine\", the "
    "query the model passed to its second search() call\n\n"
    "Decide a single label based on the semantic correctness of "
    "`model_answer` against `gold_answer`. Return STRICT JSON with exactly "
    "these keys:\n"
    "{\n"
    "  \"label\": \"correct\" | \"partial\" | \"wrong\",\n"
    "  \"cited_doc_ids\": [string, ...],\n"
    "  \"rationale\": string\n"
    "}\n\n"
    "Label rubric:\n"
    "- correct : model_answer is semantically equivalent to (or fully "
    "entails) gold_answer; all required facts are present; no contradictions; "
    "no fabricated facts.\n"
    "- partial : model_answer covers part of the required facts, or hedges "
    "on some of them, but does NOT directly contradict gold_answer.\n"
    "- wrong   : model_answer contradicts gold_answer, fabricates key facts, "
    "or refuses / abstains (\"I don't know\" / \"insufficient information\"). "
    "In Setting I the gold documents are always sufficient, so any abstain "
    "is `wrong`. In Setting II an abstain that fails to acquire the missing "
    "evidence is still `wrong`.\n\n"
    "cited_doc_ids: doc_ids the model wrote inline in square brackets like "
    "\"[web1]\"; use [] if none were cited.\n\n"
    "Output JSON only, no surrounding text."
)


def call_judge(
    client: OpenAI,
    judge_model: str,
    *,
    query: str,
    shaping_mode: str,
    answer_type: str,
    gold_answer: str,
    gold_evidence_doc_ids: Sequence[str],
    model_answer: str,
    action_taken: Optional[str] = None,
    refine_query: Optional[str] = None,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """调用 LLM judge, 返回 {label, cited_doc_ids, rationale} 或 None (失败)。

    Call the LLM judge and return {label, cited_doc_ids, rationale}, or None
    on failure.
    """
    payload: Dict[str, Any] = {
        "query": query,
        "shaping_mode": shaping_mode,
        "answer_type": answer_type,
        "gold_answer": gold_answer,
        "gold_evidence_doc_ids": list(gold_evidence_doc_ids),
        "model_answer": model_answer,
    }
    if action_taken is not None:
        payload["action_taken"] = action_taken
    if refine_query is not None:
        payload["refine_query"] = refine_query

    for attempt in range(max_retries):
        try:
            rate_limit_acquire()
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",
                     "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            parsed = _safe_json_loads(text)
            if (
                isinstance(parsed, dict)
                and parsed.get("label") in ("correct", "partial", "wrong")
            ):
                cited = parsed.get("cited_doc_ids")
                if not isinstance(cited, list):
                    cited = []
                parsed["cited_doc_ids"] = [str(x) for x in cited]
                if not isinstance(parsed.get("rationale"), str):
                    parsed["rationale"] = ""
                return {
                    "label": parsed["label"],
                    "cited_doc_ids": parsed["cited_doc_ids"],
                    "rationale": parsed["rationale"],
                }
        except Exception as e:
            print(
                f"[WARN] judge {judge_model} attempt {attempt + 1}/"
                f"{max_retries} failed: {e}",
                file=sys.stderr,
            )
            time.sleep(min(2 ** attempt, 10))
    return None


# ============================================================================
# CLI 公共参数
# Shared CLI arguments
# ============================================================================

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input", default=str(_DEFAULT_DATA),
        help=f"输入 jsonl (默认: {_DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="评测结果输出目录; raw.jsonl 会写在这里 (各脚本有各自默认值)",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=("被评目标模型, 逗号分隔; 默认沿用 Stage D panel。"
              "所有模型均通过 OpenAI 兼容接口 (--base-url) 调用。"),
    )
    parser.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                        help="LLM judge 模型 (三分类打标)")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                        help="每个 (sample, model[, variant]) 跑几次 (采样)")
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_TEMPERATURE,
                        help="被评模型的 temperature")
    parser.add_argument("--judge-temperature", type=float,
                        default=DEFAULT_JUDGE_TEMPERATURE,
                        help="judge 的 temperature (默认 0.0 求稳)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="base seed, 第 i 次 run 使用 seed+i")
    parser.add_argument("--variant-seed", type=int,
                        default=DEFAULT_VARIANT_SEED,
                        help="doc-order 变体的洗牌 seed")
    parser.add_argument("--parallel", type=int, default=DEFAULT_WORKERS,
                        help="并发 worker 数")
    parser.add_argument("--qpm", type=int, default=DEFAULT_QPM,
                        help="全局 QPM 限速")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条 (调试, 0 表示全部)")
    parser.add_argument("--limit-per-mode", type=int, default=0,
                        help=("每个 shaping_mode 各取前 N 条 (调试, 0 表示不限)。"
                              "适合冒烟测试: 少量数据也能覆盖全部能力模式。"
                              "与 --limit 同时给出时, 先按 mode 各取 N, 再对"
                              "结果整体截断到 --limit 条。"))
    parser.add_argument(
        "--only-mode", default="",
        help=(
            "逗号分隔的 shaping_mode 过滤 (空 = 不过滤)。已知 mode: "
            "plain / sufficiency / multi_hop / authority / publish_time / "
            "intra_doc_consistency。例如 --only-mode multi_hop,authority "
            "只跑多跳与权威两类。拼写错误或数据里不存在的 mode 会 warn "
            "但不阻塞运行 (匹配 0 条)。"
        ),
    )
    parser.add_argument("--max-doc-chars", type=int,
                        default=DEFAULT_MAX_DOC_CHARS,
                        help="单文档正文截断上限")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="被测模型的 OpenAI 兼容后端 URL")
    parser.add_argument("--api-key", default=None,
                        help="被测模型 API key; 默认读 EVAL_API_KEY")
    parser.add_argument(
        "--judge-base-url", default=DEFAULT_JUDGE_BASE_URL,
        help="judge 模型的 OpenAI 兼容后端 URL",
    )
    parser.add_argument(
        "--judge-api-key", default=None,
        help="judge 模型 API key; 默认读 JUDGE_API_KEY",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 messages / 拆分不调 API")
    parser.add_argument(
        "--retry-failures", action="store_true",
        help=("续跑时把 `failure` 非空 (call_failed/turn*_call_failed 等) 或 "
              "`judge_label` 为 judge_failed 的历史行也视为未完成, 重新执行。"
              "新行会 append 到 raw.jsonl (同 eval_key 会出现旧+新两行, "
              "summarize.py / group_by_id.py 会自动按 eval_key 去重取最新)。"
              "注意: judge_label==\"wrong\" 是模型答错, 不算失败, 不会被重跑。"),
    )


def parse_models(arg: str) -> List[str]:
    models = [m.strip() for m in (arg or "").split(",") if m.strip()]
    if len(set(models)) != len(models):
        raise ValueError(f"--models 中存在重复模型: {models}")
    if not models:
        raise ValueError("--models 至少需要 1 个模型")
    return models


def filter_records_by_mode(
    records: List[Dict[str, Any]], only_mode: str,
) -> List[Dict[str, Any]]:
    if not only_mode:
        return records
    modes = {m.strip() for m in only_mode.split(",") if m.strip()}
    if not modes:
        return records
    return [r for r in records if r.get("shaping_mode") in modes]


def apply_limits(
    records: List[Dict[str, Any]],
    limit: int = 0,
    limit_per_mode: int = 0,
) -> List[Dict[str, Any]]:
    """按 --limit / --limit-per-mode 截断待处理样本。

    * limit_per_mode > 0: 先对每个 shaping_mode 各保留前 N 条 (保持原始顺序,
      不打乱), 这样小规模测试也能覆盖全部能力模式。
    * limit > 0: 再对结果整体截断到前 N 条 (在 per-mode 之后应用)。
    两个参数均为 0 时原样返回。

    Limit pending samples according to --limit and --limit-per-mode.

    * limit_per_mode > 0: first retain the first N samples for each
      shaping_mode in original order without shuffling, so small tests still
      cover every capability mode.
    * limit > 0: then truncate the overall result to its first N samples,
      after applying the per-mode limit.
    Return the input unchanged when both parameters are 0.
    """
    out = records
    if limit_per_mode and limit_per_mode > 0:
        from collections import defaultdict
        counts: Dict[str, int] = defaultdict(int)
        kept: List[Dict[str, Any]] = []
        for r in out:
            mode = r.get("shaping_mode") or "unknown"
            if counts[mode] < limit_per_mode:
                kept.append(r)
                counts[mode] += 1
        out = kept
    if limit and limit > 0:
        out = out[:limit]
    return out


# ============================================================================
# Setting II search tool schema (Setting I 不需要 tools)
# Setting II search tool schema (Setting I does not need tools)
# ============================================================================

SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the web for documents relevant to the user's question. "
            "Returns a JSON object with a `docs` array; each doc has "
            "`doc_id`, `title`, `url`, `site_name`, `publish_time`, and "
            "`content`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query in natural language.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


__all__ = [
    "DEFAULT_MODELS", "DEFAULT_JUDGE_MODEL",
    "DEFAULT_BASE_URL", "DEFAULT_API_KEY",
    "DEFAULT_JUDGE_BASE_URL", "DEFAULT_JUDGE_API_KEY",
    "DEFAULT_N_RUNS",
    "DEFAULT_TEMPERATURE", "DEFAULT_JUDGE_TEMPERATURE",
    "DEFAULT_SEED", "DEFAULT_VARIANT_SEED", "DEFAULT_QPM",
    "DEFAULT_WORKERS", "DEFAULT_MAX_DOC_CHARS", "VARIANT_NAMES",
    "SEARCH_TOOL_SCHEMA", "JUDGE_SYSTEM",
    "_DEFAULT_DATA", "_DEFAULT_SETTING2_IDS", "_PKG_ROOT",
    "set_qpm", "rate_limit_acquire",
    "build_client", "create_chat",
    "load_jsonl", "append_jsonl", "load_done_keys", "count_retriable_rows",
    "iter_evidence_doc_ids", "build_full_docs", "generate_doc_orders",
    "docs_to_prompt_text", "docs_to_tool_payload",
    "simple_retrieve",
    "extract_message_text", "extract_search_tool_call",
    "is_committed_content", "parse_query_arg", "assistant_message_to_dict",
    "call_judge",
    "add_common_args", "parse_models", "filter_records_by_mode",
    "apply_limits",
]
