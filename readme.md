# DelegSearchBench


More details can be seen in details.pdf~

## Data Structure

![pipeline](DelegSearch/pipeline.png)

Each benchmark record has the following top-level fields:

- `id`: unique example identifier.
- `query`: question presented to the evaluated model.
- `shaping_mode`: capability category. One of `plain`, `authority`,
  `publish_time`, `intra_doc_consistency`, `sufficiency`, or `multi_hop`.
- `subtype`: primary distractor type. One of `numerical_near_miss`,
  `temporal_shift`, `entity_confusion`, `scope_mismatch`, or
  `version_mismatch`.
- `_source_lang`: source language, either `en` or `zh`.
- `golden_answer`: reference answer used by the judge.
- `evidence_spans`: supporting spans from the golden documents. Each span
  contains `doc_id`, `text`, `start`, `end`, `located`, and `match`.
- `reasoning_steps`: reference reasoning steps.
- `golden_docs`: documents containing the evidence needed to answer the query.
- `distractor_docs`: plausible but misleading documents.
- `irrelevant_docs`: documents unrelated to the answer.

The three document collections are objects keyed by document ID. Every document
contains:

```json
{
  "title": "Document title",
  "url": "https://example.com/document",
  "site_name": "example.com",
  "publish_time": "2024-01-01",
  "content": "Document text"
}
```

Documents in `distractor_docs` additionally contain diagnostic annotations:

- `distractor_type`
- `target_evidence_doc_id`
- `target_evidence_text`
- `discriminating_constraint`
- `misleading_rationale`

These annotations, together with `golden_answer`, `evidence_spans`,
`reasoning_steps`, the document collection names, and the analysis fields, are
never exposed to the evaluated model. At evaluation time, each visible document
is projected to the whitelist `doc_id`, `title`, `url`, `site_name`,
`publish_time`, and `content`.

## Evaluation

Run all commands from the repository root.

### Setup

Install the evaluation dependencies:

```bash
pip install -r research/DelegSearch/eval/requirements.txt
```

Configure the evaluated-model and judge endpoints in `.env`:

```dotenv
EVAL_API_KEY=your-eval-api-key
EVAL_BASE_URL=https://your-openai-compatible-endpoint/
EVAL_MODELS=model-a,model-b
JUDGE_API_KEY=your-judge-api-key
JUDGE_BASE_URL=https://your-openai-compatible-endpoint/
JUDGE_MODEL=judge-model
```

The evaluated models and judge must be different models. Endpoint and API-key
values can also be overridden with `--base-url`, `--api-key`,
`--judge-base-url`, and `--judge-api-key`.

### Setting I: Full Context

Setting I evaluates all 429 examples with the complete document set and no
search tool. Documents are tested in multiple order variants.

```bash
python research/DelegSearch/eval/setting1.py \
  --models model-a,model-b \
  --judge judge-model \
  --n-runs 3 \
  --parallel 32
```

The default output is `eval/results/setting1/raw.jsonl`. Summarize it with:

```bash
python research/DelegSearch/eval/summarize.py \
  --setting 1 \
  --input-dir research/DelegSearch/eval/results/setting1
```

### Setting II: Partial Context with Search

Setting II evaluates the IDs in `data/setting2_ids.txt`. Part of the golden
evidence is withheld from the initial context, and the model can search a closed
corpus containing that example's golden, distractor, and irrelevant documents.

```bash
python research/DelegSearch/eval/setting2_byid.py \
  --models model-a,model-b \
  --judge judge-model \
  --n-runs 3 \
  --parallel 32
```

The default output is `eval/results/setting2_byid/raw.jsonl`. Summarize it with:

```bash
python research/DelegSearch/eval/summarize.py \
  --setting 2 \
  --input-dir research/DelegSearch/eval/results/setting2_byid
```