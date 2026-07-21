"""Build a focused AI landmark corpus for Cite Scope.

The corpus is intentionally vertical: LLM / RAG / Agents / evaluation first,
with enough AI history papers to make cross-era questions meaningful.

Outputs:
  - data/metadata_filtered.json                 # records ready for ingest
  - data/raw_metadata/ai_landmark_candidates.json
  - data/raw_metadata/ai_landmark_skipped.json

The script uses only public arXiv HTML/PDF endpoints and keeps an HTML cache so
it can be safely resumed after network or rate-limit interruptions.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
RAW_DIR = DATA_DIR / "raw_metadata"
CACHE_DIR = RAW_DIR / "arxiv_list_cache"
METADATA_JSON = DATA_DIR / "metadata_filtered.json"
CANDIDATES_JSON = RAW_DIR / "ai_landmark_candidates.json"
SKIPPED_JSON = RAW_DIR / "ai_landmark_skipped.json"

UA = "CiteScope/1.0 (curated AI landmark corpus; mailto:dev@citescope.local)"
HTTP_TIMEOUT = 45
MAX_RETRIES = 3
LIST_SLEEP_SEC = 0.35
MAX_PDF_BYTES = 60 * 1024 * 1024


BUCKET_TARGETS = {
    "deep_learning": 80,
    "llm_transformer": 90,
    "alignment_safety_eval": 70,
    "rag_ir_memory": 90,
    "agents_reasoning": 80,
    "multimodal_generative": 50,
    "evaluation_factuality": 40,
}

BUCKET_TERMS = {
    "rag_ir_memory": [
        "retrieval", "retrieval-augmented", "rag", "dense passage", "open-domain",
        "question answering", "colbert", "contriever", "rerank", "search",
        "memory", "datastore", "knowledge-intensive", "document", "evidence",
    ],
    "agents_reasoning": [
        "agent", "agents", "tool", "tool use", "react", "reasoning",
        "chain-of-thought", "tree of thoughts", "self-refine", "reflection",
        "planning", "plan", "workflow", "autonomous", "web", "environment",
    ],
    "llm_transformer": [
        "language model", "large language model", "transformer", "pre-training",
        "pretraining", "bert", "gpt", "llama", "t5", "scaling", "mixture",
        "moe", "instruction", "token", "context", "attention",
    ],
    "alignment_safety_eval": [
        "alignment", "rlhf", "human feedback", "reward model", "preference",
        "instruction tuning", "constitutional", "safety", "harmless",
        "helpful", "jailbreak", "red teaming", "toxicity", "bias",
    ],
    "evaluation_factuality": [
        "benchmark", "evaluation", "evaluate", "hallucination", "factual",
        "truthful", "mmlu", "helm", "big-bench", "gsm8k", "humaneval",
        "faithfulness", "attribution", "verifiability",
    ],
    "multimodal_generative": [
        "vision-language", "vision language", "multimodal", "clip",
        "diffusion", "image generation", "text-to-image", "video",
        "visual instruction", "llava", "flamingo", "dall-e", "vit",
    ],
    "deep_learning": [
        "neural", "deep learning", "convolution", "recurrent", "lstm",
        "representation", "embedding", "dropout", "normalization",
        "optimization", "adam", "residual", "sequence to sequence",
    ],
    "history": [
        "learning", "neural network", "reinforcement learning", "artificial intelligence",
        "machine learning", "probabilistic", "bayesian", "support vector",
        "decision", "planning", "search", "markov",
    ],
}

LANDMARK_SEEDS = {
    "1706.03762": "llm_transformer",  # Transformer
    "1810.04805": "llm_transformer",  # BERT
    "1907.11692": "llm_transformer",  # RoBERTa
    "1909.11942": "llm_transformer",  # ALBERT
    "1910.10683": "llm_transformer",  # T5
    "2001.08361": "llm_transformer",  # scaling laws
    "2003.10555": "llm_transformer",  # ELECTRA
    "2005.14165": "llm_transformer",  # GPT-3
    "2112.11446": "llm_transformer",  # Gopher
    "2203.15556": "llm_transformer",  # Chinchilla
    "2201.08239": "llm_transformer",  # LaMDA
    "2204.02311": "llm_transformer",  # PaLM
    "2302.13971": "llm_transformer",  # LLaMA
    "2303.08774": "llm_transformer",  # GPT-4 technical report
    "2307.09288": "llm_transformer",  # Llama 2
    "2310.06825": "llm_transformer",  # Mistral 7B
    "2407.21783": "llm_transformer",  # Llama 3.1
    "2002.08909": "rag_ir_memory",  # REALM
    "2004.04906": "rag_ir_memory",  # DPR
    "2004.12832": "rag_ir_memory",  # ColBERT
    "2005.11401": "rag_ir_memory",  # RAG
    "2007.01282": "rag_ir_memory",  # Fusion-in-Decoder
    "2112.09118": "rag_ir_memory",  # Contriever
    "2112.04426": "rag_ir_memory",  # RETRO
    "2208.03299": "rag_ir_memory",  # Atlas
    "2212.10496": "rag_ir_memory",  # HyDE
    "2201.11903": "agents_reasoning",  # Chain-of-thought
    "2210.03629": "agents_reasoning",  # ReAct
    "2302.04761": "agents_reasoning",  # Toolformer
    "2303.11366": "agents_reasoning",  # Reflexion
    "2303.17651": "agents_reasoning",  # Self-Refine
    "2305.10601": "agents_reasoning",  # Tree of Thoughts
    "2305.16291": "agents_reasoning",  # Voyager
    "2304.03442": "agents_reasoning",  # Generative Agents
    "2109.07958": "evaluation_factuality",  # TruthfulQA
    "2009.03300": "evaluation_factuality",  # MMLU
    "2110.14168": "evaluation_factuality",  # GSM8K
    "2107.03374": "evaluation_factuality",  # HumanEval
    "2206.04615": "evaluation_factuality",  # BIG-bench
    "2211.09110": "evaluation_factuality",  # HELM
    "2305.11747": "evaluation_factuality",  # HaluEval
    "2203.02155": "alignment_safety_eval",  # InstructGPT
    "2204.05862": "alignment_safety_eval",  # Training language models to follow instructions
    "2212.08073": "alignment_safety_eval",  # Constitutional AI
    "2103.00020": "multimodal_generative",  # CLIP
    "2010.11929": "multimodal_generative",  # ViT
    "2006.11239": "multimodal_generative",  # DDPM
    "2112.10752": "multimodal_generative",  # latent diffusion
    "2201.12086": "multimodal_generative",  # BLIP
    "2301.12597": "multimodal_generative",  # BLIP-2
    "2304.08485": "multimodal_generative",  # LLaVA
    "2304.02643": "multimodal_generative",  # SAM
    "1207.0580": "deep_learning",  # dropout
    "1301.3781": "deep_learning",  # word2vec
    "1312.5602": "deep_learning",  # DQN
    "1409.0473": "deep_learning",  # attention for NMT
    "1409.3215": "deep_learning",  # seq2seq
    "1412.6980": "deep_learning",  # Adam
    "1502.03167": "deep_learning",  # batch norm
    "1512.03385": "deep_learning",  # ResNet
    "1607.06450": "deep_learning",  # layer norm
    "1608.06993": "deep_learning",  # DenseNet
    "1707.06347": "deep_learning",  # PPO
}

LIST_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.IR"]
LIST_YEARS = list(range(2017, 2027))


@dataclass
class Candidate:
    paper_id: str
    title: str
    authors: list[str]
    year: int
    primary_category: str
    categories: list[str]
    abstract: str = ""
    pdf_url: str = ""
    entry_id: str = ""
    bucket: str = "history"
    score: float = 0.0
    seed: bool = False
    importance_reason: str = ""
    source: str = "arxiv"
    raw_subjects: str = ""
    rank_notes: list[str] = field(default_factory=list)

    def to_record(self, pdf_path: str | None = None) -> dict:
        record = {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "published": f"{self.year}-01-01T00:00:00+00:00",
            "updated": None,
            "primary_category": self.primary_category[:32],
            "categories": list(dict.fromkeys([*self.categories, self.bucket])),
            "doi": None,
            "abstract": self.abstract,
            "pdf_url": self.pdf_url or f"https://arxiv.org/pdf/{self.paper_id}.pdf",
            "entry_id": self.entry_id or f"https://arxiv.org/abs/{self.paper_id}",
            "corpus_bucket": self.bucket,
            "importance_reason": self.importance_reason,
            "corpus_score": round(self.score, 3),
            "corpus_source": self.source,
        }
        if pdf_path:
            record["pdf_path"] = pdf_path
        return record


def _clean_text(raw: str) -> str:
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _year_from_id(paper_id: str) -> int:
    prefix = paper_id.split(".")[0]
    if len(prefix) == 4 and prefix.isdigit():
        yy = int(prefix[:2])
        return 2000 + yy if yy < 90 else 1900 + yy
    return 0


def _request(url: str, *, stream: bool = False) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": UA},
                timeout=HTTP_TIMEOUT,
                stream=stream,
            )
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                time.sleep(2.0 * attempt)
                continue
            return resp
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2.0 * attempt)
                continue
            raise
    assert last_error is not None
    raise last_error


def _fetch_cached(url: str, cache_name: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    resp = _request(url)
    resp.raise_for_status()
    text = resp.text
    cache_path.write_text(text, encoding="utf-8")
    time.sleep(LIST_SLEEP_SEC)
    return text


def _parse_authors(block: str) -> list[str]:
    authors_block = re.search(
        r"<div class='list-authors'>(.*?)</div>", block, flags=re.S
    )
    if not authors_block:
        return []
    authors = re.findall(r">(.*?)</a>", authors_block.group(1), flags=re.S)
    return [_clean_text(a) for a in authors if _clean_text(a)]


def _parse_subjects(block: str) -> tuple[str, list[str], str]:
    subjects_block = re.search(
        r"<div class='list-subjects'>(.*?)</div>", block, flags=re.S
    )
    raw = _clean_text(subjects_block.group(1)) if subjects_block else ""
    cats = re.findall(r"\(([a-z.-]+\.[A-Z]{2})\)", raw)
    primary = cats[0] if cats else ""
    return primary, list(dict.fromkeys(cats)), raw


def parse_arxiv_list(html_text: str, fallback_category: str) -> list[Candidate]:
    records: list[Candidate] = []
    for match in re.finditer(r"<dt>(.*?)</dd>", html_text, flags=re.S):
        block = match.group(0)
        id_match = re.search(r'id="(\d{4}\.\d{4,5})"', block)
        if not id_match:
            continue
        paper_id = id_match.group(1)
        title_match = re.search(
            r"<div class='list-title mathjax'><span class='descriptor'>Title:</span>(.*?)</div>",
            block,
            flags=re.S,
        )
        if not title_match:
            continue
        title = _clean_text(title_match.group(1))
        primary, categories, subjects = _parse_subjects(block)
        if not primary:
            primary = fallback_category
        if not categories:
            categories = [fallback_category]
        records.append(
            Candidate(
                paper_id=paper_id,
                title=title,
                authors=_parse_authors(block),
                year=_year_from_id(paper_id),
                primary_category=primary,
                categories=categories,
                pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
                entry_id=f"https://arxiv.org/abs/{paper_id}",
                raw_subjects=subjects,
            )
        )
    return records


def fetch_year_candidates() -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    for category in LIST_CATEGORIES:
        for year in LIST_YEARS:
            url = f"https://arxiv.org/list/{category}/{year}?show=2000"
            cache_name = f"{category.replace('.', '_')}_{year}.html"
            try:
                text = _fetch_cached(url, cache_name)
            except Exception as exc:
                print(f"[warn] list fetch failed {category} {year}: {exc}", flush=True)
                continue
            parsed = parse_arxiv_list(text, fallback_category=category)
            for item in parsed:
                candidates.setdefault(item.paper_id, item)
            print(f"[list] {category} {year}: parsed={len(parsed)} total={len(candidates)}", flush=True)
    return candidates


def fetch_recent_candidates() -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    for category in LIST_CATEGORIES:
        url = f"https://arxiv.org/list/{category}/recent?show=1000"
        cache_name = f"{category.replace('.', '_')}_recent.html"
        try:
            text = _fetch_cached(url, cache_name)
        except Exception as exc:
            print(f"[warn] recent fetch failed {category}: {exc}", flush=True)
            continue
        parsed = parse_arxiv_list(text, fallback_category=category)
        for item in parsed:
            candidates.setdefault(item.paper_id, item)
        print(f"[recent] {category}: parsed={len(parsed)} total={len(candidates)}", flush=True)
    return candidates


def fetch_seed_candidate(paper_id: str, bucket: str) -> Candidate | None:
    url = f"https://arxiv.org/abs/{paper_id}"
    cache_name = f"abs_{paper_id.replace('.', '_')}.html"
    try:
        text = _fetch_cached(url, cache_name)
    except Exception as exc:
        print(f"[warn] seed fetch failed {paper_id}: {exc}", flush=True)
        return None
    if "not found" in text.lower() or "error" in text[:500].lower():
        return None
    title_match = re.search(r"<h1 class=\"title mathjax\">(.*?)</h1>", text, flags=re.S)
    title = _clean_text(title_match.group(1).replace("Title:", "")) if title_match else ""
    authors_match = re.search(r"<div class=\"authors\">(.*?)</div>", text, flags=re.S)
    authors = re.findall(r">(.*?)</a>", authors_match.group(1), flags=re.S) if authors_match else []
    abstract_match = re.search(
        r"<blockquote class=\"abstract mathjax\">(.*?)</blockquote>", text, flags=re.S
    )
    abstract = (
        _clean_text(abstract_match.group(1).replace("Abstract:", ""))
        if abstract_match
        else ""
    )
    subjects_match = re.search(r"<td class=\"tablecell subjects\">(.*?)</td>", text, flags=re.S)
    subjects = _clean_text(subjects_match.group(1)) if subjects_match else ""
    cats = re.findall(r"\(([a-z.-]+\.[A-Z]{2})\)", subjects)
    primary = cats[0] if cats else "cs.AI"
    if not title:
        return None
    return Candidate(
        paper_id=paper_id,
        title=title,
        authors=[_clean_text(a) for a in authors if _clean_text(a)],
        year=_year_from_id(paper_id),
        primary_category=primary,
        categories=list(dict.fromkeys(cats or [primary])),
        abstract=abstract,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
        entry_id=f"https://arxiv.org/abs/{paper_id}",
        bucket=bucket,
        seed=True,
        raw_subjects=subjects,
    )


def classify_and_score(candidate: Candidate) -> Candidate:
    text = f"{candidate.title} {candidate.raw_subjects}".lower()
    bucket_scores: dict[str, float] = {}
    for bucket, terms in BUCKET_TERMS.items():
        score = 0.0
        for term in terms:
            if term in text:
                score += 8.0 if " " in term or "-" in term else 4.0
        bucket_scores[bucket] = score

    if candidate.seed:
        bucket_scores[candidate.bucket] = bucket_scores.get(candidate.bucket, 0.0) + 120.0

    bucket, best_score = max(bucket_scores.items(), key=lambda kv: kv[1])
    candidate.bucket = bucket

    year_bonus = 0.0
    if candidate.year >= 2023:
        year_bonus = 8.0
    elif 2017 <= candidate.year <= 2022:
        year_bonus = 5.0
    elif candidate.year:
        year_bonus = 2.0

    category_bonus = 0.0
    if any(cat in {"cs.CL", "cs.AI", "cs.LG", "cs.IR"} for cat in candidate.categories):
        category_bonus += 6.0
    if candidate.primary_category in {"cs.CL", "cs.AI", "cs.LG", "cs.IR"}:
        category_bonus += 4.0

    seed_bonus = 150.0 if candidate.seed else 0.0
    candidate.score = best_score + year_bonus + category_bonus + seed_bonus
    if candidate.score <= 0:
        candidate.score = year_bonus + category_bonus

    if candidate.seed:
        candidate.rank_notes.append("seed_landmark")
    if best_score > 0:
        candidate.rank_notes.append(f"matched_{candidate.bucket}")
    candidate.importance_reason = build_importance_reason(candidate)
    return candidate


def build_importance_reason(candidate: Candidate) -> str:
    label = {
        "history": "AI/ML historical context",
        "deep_learning": "deep learning foundations",
        "llm_transformer": "foundation-model and Transformer development",
        "alignment_safety_eval": "alignment, safety, or instruction tuning",
        "rag_ir_memory": "retrieval, RAG, or external memory",
        "agents_reasoning": "LLM agents, tool use, or reasoning",
        "multimodal_generative": "multimodal or generative modeling",
        "evaluation_factuality": "evaluation, factuality, or benchmark design",
    }.get(candidate.bucket, candidate.bucket)
    prefix = "Seed landmark" if candidate.seed else "Curated arXiv paper"
    return f"{prefix} for {label}; selected for Cite Scope vertical evaluation."


def build_candidate_pool() -> list[Candidate]:
    by_id = fetch_year_candidates()
    by_id.update({k: v for k, v in fetch_recent_candidates().items() if k not in by_id})

    for paper_id, bucket in LANDMARK_SEEDS.items():
        if paper_id in by_id:
            by_id[paper_id].seed = True
            by_id[paper_id].bucket = bucket
        else:
            seed = fetch_seed_candidate(paper_id, bucket)
            if seed:
                by_id[paper_id] = seed

    scored = [classify_and_score(c) for c in by_id.values()]
    scored = [c for c in scored if c.score >= 10.0 or c.seed]
    return sorted(scored, key=lambda c: (c.seed, c.score, c.year), reverse=True)


def select_candidates(candidates: Iterable[Candidate], target: int, oversample: int) -> list[Candidate]:
    by_bucket: dict[str, list[Candidate]] = {bucket: [] for bucket in BUCKET_TARGETS}
    for c in candidates:
        by_bucket.setdefault(c.bucket, []).append(c)

    selected: dict[str, Candidate] = {}
    ordered: list[Candidate] = []
    for bucket, quota in BUCKET_TARGETS.items():
        for item in sorted(by_bucket.get(bucket, []), key=lambda c: (c.seed, c.score, c.year), reverse=True)[:quota]:
            if item.paper_id not in selected:
                selected[item.paper_id] = item
                ordered.append(item)

    for item in candidates:
        if len(selected) >= target + oversample:
            break
        if item.bucket not in BUCKET_TARGETS:
            continue
        if item.paper_id not in selected:
            selected[item.paper_id] = item
            ordered.append(item)

    return ordered


def download_pdf(candidate: Candidate) -> tuple[bool, str]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PDF_DIR / f"{candidate.paper_id}.pdf"
    rel_path = out_path.relative_to(PROJECT_ROOT)
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True, str(rel_path)

    tmp = out_path.with_suffix(".pdf.part")
    try:
        resp = _request(candidate.pdf_url, stream=True)
        if resp.status_code >= 400:
            return False, f"http_{resp.status_code}"
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
            return False, f"not_pdf:{content_type[:80]}"
        written = 0
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_PDF_BYTES:
                    tmp.unlink(missing_ok=True)
                    return False, "pdf_too_large"
                handle.write(chunk)
        if written < 10_000:
            tmp.unlink(missing_ok=True)
            return False, "pdf_too_small"
        tmp.rename(out_path)
        return True, str(rel_path)
    except Exception as exc:  # pragma: no cover - network behavior
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def _download_one(candidate: Candidate) -> tuple[Candidate, bool, str]:
    ok, detail = download_pdf(candidate)
    return candidate, ok, detail


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate AI landmark corpus for Cite Scope.")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--oversample", type=int, default=180)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_pool()
    selected = select_candidates(candidates, target=args.target, oversample=args.oversample)
    write_json(CANDIDATES_JSON, [c.to_record() for c in selected])
    print(f"[candidates] scored={len(candidates)} selected_for_download={len(selected)}", flush=True)

    if args.skip_download:
        print(f"[done] candidates only -> {CANDIDATES_JSON}", flush=True)
        return 0

    ok_records: list[dict] = []
    skipped: list[dict] = []

    workers = max(1, min(args.workers, 10))
    selected_iter = iter(selected)
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        future_candidates = {}

        def submit_next() -> None:
            nonlocal exhausted
            if exhausted:
                return
            try:
                candidate = next(selected_iter)
            except StopIteration:
                exhausted = True
                return
            future = pool.submit(_download_one, candidate)
            pending.add(future)
            future_candidates[future] = candidate

        for _ in range(workers):
            submit_next()

        with tqdm(total=len(selected), desc="Downloading curated PDFs", unit="pdf") as pbar:
            while pending and len(ok_records) < args.target:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    candidate = future_candidates.pop(future)
                    try:
                        candidate, ok, detail = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        ok = False
                        detail = f"{type(exc).__name__}: {exc}"
                    if ok:
                        ok_records.append(candidate.to_record(pdf_path=detail))
                    else:
                        skipped.append({
                            **candidate.to_record(),
                            "skip_reason": detail or "download_failed",
                        })
                    pbar.update(1)
                    pbar.set_postfix(ok=len(ok_records), skipped=len(skipped), refresh=False)
                    if len(ok_records) >= args.target:
                        break
                    submit_next()

            for future in pending:
                future.cancel()

    write_json(METADATA_JSON, ok_records)
    write_json(SKIPPED_JSON, skipped)

    print(f"[done] downloaded={len(ok_records)} target={args.target} metadata={METADATA_JSON}", flush=True)
    print(f"[done] skipped={len(skipped)} skipped_json={SKIPPED_JSON}", flush=True)
    return 0 if len(ok_records) >= args.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
