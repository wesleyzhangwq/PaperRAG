"""Generate semantic evaluation questions from paper metadata using LLM.

Usage:
    cd backend
    python ../eval/scripts/gen_questions.py --output ../eval/datasets/questions_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import get_settings

CONCEPT_LOCATE_PROMPT = """You are generating evaluation questions for a RAG system that indexes academic papers.

Given the following paper metadata, generate ONE semantic question that a researcher would naturally ask to find this paper. The question must:
1. NOT quote any text from the title or abstract verbatim (no copy-paste phrases)
2. Describe the paper's contribution using your own words / paraphrasing
3. Be specific enough that only this paper (among ~100 CS papers) would answer it
4. Be in Chinese (中文)

Paper:
- ID: {paper_id}
- Title: {title}
- Category: {category}
- Abstract: {abstract}

Also generate a concise reference answer (2-3 sentences in Chinese) that summarizes the paper's key contribution and method.

Respond with ONLY a JSON object:
{{"question": "<question text>", "reference_answer": "<reference answer>"}}"""

METHOD_DETAIL_PROMPT = """You are generating evaluation questions for a RAG system.

Given the following paper, generate ONE question asking about a specific technical detail of the method — e.g., what technique is used, what loss function, what architecture component, what dataset, etc. The question must:
1. NOT quote the abstract verbatim
2. Ask about a specific aspect, not a general summary
3. Be answerable from the paper's content
4. Be in Chinese (中文)

Paper:
- ID: {paper_id}
- Title: {title}
- Category: {category}
- Abstract: {abstract}

Generate a reference answer (2-3 sentences in Chinese).

Respond with ONLY a JSON object:
{{"question": "<question text>", "reference_answer": "<reference answer>"}}"""

COMPARISON_PROMPT = """You are generating evaluation questions for a RAG system.

Given two related papers, generate ONE comparison question asking about their differences in approach, methodology, or focus. The question must:
1. NOT quote abstracts verbatim
2. Describe both papers' topics in your own words
3. Ask for specific differences (not just "compare them")
4. Be in Chinese (中文)

Paper A:
- ID: {paper_id_a}
- Title: {title_a}
- Abstract: {abstract_a}

Paper B:
- ID: {paper_id_b}
- Title: {title_b}
- Abstract: {abstract_b}

Generate a reference answer (3-4 sentences in Chinese).

Respond with ONLY a JSON object:
{{"question": "<question text>", "reference_answer": "<reference answer>"}}"""

TREND_PROMPT = """You are generating evaluation questions for a RAG system that indexes ~100 recent CS papers.

Given the following group of related papers, generate ONE synthesis question asking about trends, common themes, or contrasting approaches across these papers. The question must:
1. Ask about patterns or themes, not individual papers
2. Be answerable by examining multiple papers in the group
3. Be in Chinese (中文)

Papers in this group ({topic}):
{papers_list}

Generate a reference answer (3-5 sentences in Chinese) that references specific papers.

Respond with ONLY a JSON object:
{{"question": "<question text>", "reference_answer": "<reference answer>", "relevant_paper_ids": [<list of paper IDs relevant to the answer>]}}"""


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.7,
        max_retries=1,
        request_timeout=45,
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    # Try to find a valid JSON object by progressively shortening from the end
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON found in: {text[:200]}")
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"Unbalanced JSON in: {text[:200]}")


def _call_llm(llm: ChatOpenAI, prompt: str, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke(prompt)
            return _parse_json(resp.content)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  Retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(1)
    return {}


def load_papers(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_papers_by_topic(papers: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for p in papers:
        cat = p["primary_category"]
        groups.setdefault(cat, []).append(p)
    return groups


def find_related_pairs(papers: list[dict]) -> list[tuple[dict, dict]]:
    """Find pairs of papers in the same category for comparison questions."""
    groups = group_papers_by_topic(papers)
    pairs = []
    for cat, group in groups.items():
        if len(group) >= 2:
            shuffled = group.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                pairs.append((shuffled[i], shuffled[i + 1]))
    return pairs


def gen_concept_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 20
) -> list[dict]:
    selected = random.sample(papers, min(count, len(papers)))
    questions = []
    for i, p in enumerate(selected):
        print(f"  Concept {i+1}/{len(selected)}: {p['paper_id']}", file=sys.stderr)
        prompt = CONCEPT_LOCATE_PROMPT.format(
            paper_id=p["paper_id"],
            title=p["title"],
            category=p["primary_category"],
            abstract=p["abstract"][:600],
        )
        try:
            result = _call_llm(llm, prompt)
            questions.append({
                "qid": f"c{i+1:03d}",
                "query": result["question"],
                "expected_paper_ids": [p["paper_id"]],
                "expected_mode": "answer",
                "reference_answer": result.get("reference_answer", ""),
                "difficulty": "easy",
                "type": "concept_locate",
                "tags": ["single-paper", "semantic"],
            })
        except Exception as e:
            print(f"  SKIP {p['paper_id']}: {e}", file=sys.stderr)
        time.sleep(0.5)
    return questions


def gen_method_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 15
) -> list[dict]:
    selected = random.sample(papers, min(count, len(papers)))
    questions = []
    for i, p in enumerate(selected):
        print(f"  Method {i+1}/{len(selected)}: {p['paper_id']}", file=sys.stderr)
        prompt = METHOD_DETAIL_PROMPT.format(
            paper_id=p["paper_id"],
            title=p["title"],
            category=p["primary_category"],
            abstract=p["abstract"][:600],
        )
        try:
            result = _call_llm(llm, prompt)
            questions.append({
                "qid": f"m{i+1:03d}",
                "query": result["question"],
                "expected_paper_ids": [p["paper_id"]],
                "expected_mode": "answer",
                "reference_answer": result.get("reference_answer", ""),
                "difficulty": "medium",
                "type": "method_detail",
                "tags": ["single-paper", "semantic", "detail"],
            })
        except Exception as e:
            print(f"  SKIP {p['paper_id']}: {e}", file=sys.stderr)
        time.sleep(0.5)
    return questions


def gen_comparison_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 10
) -> list[dict]:
    pairs = find_related_pairs(papers)
    selected = pairs[:count]
    questions = []
    for i, (a, b) in enumerate(selected):
        print(
            f"  Compare {i+1}/{len(selected)}: {a['paper_id']} vs {b['paper_id']}",
            file=sys.stderr,
        )
        prompt = COMPARISON_PROMPT.format(
            paper_id_a=a["paper_id"],
            title_a=a["title"],
            abstract_a=a["abstract"][:400],
            paper_id_b=b["paper_id"],
            title_b=b["title"],
            abstract_b=b["abstract"][:400],
        )
        try:
            result = _call_llm(llm, prompt)
            questions.append({
                "qid": f"x{i+1:03d}",
                "query": result["question"],
                "expected_paper_ids": [a["paper_id"], b["paper_id"]],
                "expected_mode": "answer",
                "reference_answer": result.get("reference_answer", ""),
                "difficulty": "hard",
                "type": "comparison",
                "tags": ["pairwise", "semantic"],
            })
        except Exception as e:
            print(f"  SKIP: {e}", file=sys.stderr)
        time.sleep(0.5)
    return questions


def gen_trend_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 5
) -> list[dict]:
    topic_groups = {
        "LLM Agent 系统": [
            p for p in papers
            if any(kw in p["title"].lower() for kw in ["agent", "agentic"])
        ],
        "LLM 评估与可靠性": [
            p for p in papers
            if any(kw in p["title"].lower() for kw in ["judge", "eval", "benchmark", "diagnos"])
        ],
        "强化学习与推理": [
            p for p in papers
            if any(kw in p["title"].lower() for kw in ["reinforcement", "rl", "reasoning", "reward"])
        ],
        "高效 Transformer 与稀疏注意力": [
            p for p in papers
            if any(kw in p["title"].lower() for kw in ["sparse", "attention", "compress", "pruning", "token"])
        ],
        "联邦学习与隐私": [
            p for p in papers
            if any(kw in p["title"].lower() for kw in ["federat", "privacy", "gradient inversion"])
        ],
    }
    questions = []
    idx = 0
    for topic, group in topic_groups.items():
        if len(group) < 2 or idx >= count:
            continue
        idx += 1
        papers_list = "\n".join(
            f"- [{p['paper_id']}] {p['title']}" for p in group[:8]
        )
        print(f"  Trend {idx}/{count}: {topic} ({len(group)} papers)", file=sys.stderr)
        prompt = TREND_PROMPT.format(
            topic=topic,
            papers_list=papers_list,
        )
        try:
            result = _call_llm(llm, prompt)
            relevant_ids = result.get("relevant_paper_ids", [p["paper_id"] for p in group[:3]])
            questions.append({
                "qid": f"t{idx:03d}",
                "query": result["question"],
                "expected_paper_ids": relevant_ids,
                "expected_mode": "answer",
                "reference_answer": result.get("reference_answer", ""),
                "difficulty": "hard",
                "type": "trend_synthesis",
                "tags": ["multi-paper", "semantic", "synthesis"],
            })
        except Exception as e:
            print(f"  SKIP {topic}: {e}", file=sys.stderr)
        time.sleep(0.5)
    return questions


NEGATIVE_QUESTIONS = [
    {
        "qid": "n001",
        "query": "这些论文中有哪些关于蛋白质折叠预测（如 AlphaFold）的研究？",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "本语料库中没有关于蛋白质折叠预测的论文。",
        "difficulty": "easy",
        "type": "negative",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "qid": "n002",
        "query": "请总结语料库中关于大型语言模型在金融风控领域实际部署经验的论文。",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "语料库中没有直接讨论LLM在金融风控实际部署经验的论文。",
        "difficulty": "medium",
        "type": "negative",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "qid": "n003",
        "query": "有哪些论文讨论了气候变化建模与碳排放预测？",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "本语料库中没有关于气候变化建模或碳排放预测的论文。",
        "difficulty": "easy",
        "type": "negative",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "qid": "n004",
        "query": "请列出这些论文中关于脑机接口（BCI）硬件设计与临床试验的研究进展。",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "语料库中有涉及EEG基础模型知识蒸馏的论文(DLink)，但没有关于BCI硬件设计或临床试验的研究。",
        "difficulty": "hard",
        "type": "negative",
        "tags": ["negative", "near-miss"],
    },
    {
        "qid": "n005",
        "query": "这些论文中是否有关于自动驾驶端到端规划算法（如 UniAD, VAD）的改进工作？",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "语料库中有一篇关于自动驾驶视觉异常检测的论文(AD4AD)，但没有关于端到端规划算法改进的工作。",
        "difficulty": "hard",
        "type": "negative",
        "tags": ["negative", "near-miss"],
    },
    {
        "qid": "n006",
        "query": "请提供关于 Transformer 在蛋白质序列建模中应用的论文综述。",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "本语料库中没有关于 Transformer 在蛋白质序列建模中应用的论文。",
        "difficulty": "easy",
        "type": "negative",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "qid": "n007",
        "query": "语料库中有哪些关于推荐系统冷启动问题的研究？",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "语料库中有一篇关于contextual bandits冷启动的论文，但没有专门讨论推荐系统冷启动问题的研究。",
        "difficulty": "hard",
        "type": "negative",
        "tags": ["negative", "near-miss"],
    },
    {
        "qid": "n008",
        "query": "这些论文中有关于 RLHF（基于人类反馈的强化学习）具体实现细节的工作吗？",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "语料库中有多篇涉及RL用于LLM训练的论文，但没有专门讨论RLHF具体实现细节的工作。涉及RL的论文更关注奖励设计和验证机制。",
        "difficulty": "hard",
        "type": "negative",
        "tags": ["negative", "near-miss"],
    },
    {
        "qid": "n009",
        "query": "请总结语料库中关于3D场景生成和NeRF改进的研究。",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "本语料库中没有关于3D场景生成或NeRF改进的论文。",
        "difficulty": "easy",
        "type": "negative",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "qid": "n010",
        "query": "有哪些论文研究了大模型的数据标注成本优化和主动学习策略？",
        "expected_paper_ids": [],
        "expected_mode": "insufficient",
        "reference_answer": "本语料库中没有专门研究数据标注成本优化或主动学习策略的论文。",
        "difficulty": "medium",
        "type": "negative",
        "tags": ["negative", "out-of-scope"],
    },
]


def gen_fact_extract_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 10
) -> list[dict]:
    """Generate questions asking for specific facts from papers."""
    fact_prompt = """You are generating evaluation questions for a RAG system.

Given the following paper, generate ONE specific factual question about the paper's experimental setup, datasets used, baselines compared, or quantitative results mentioned in the abstract. The question must:
1. Ask about a concrete, verifiable fact
2. NOT quote the abstract
3. Be answerable from the paper content
4. Be in Chinese (中文)

Paper:
- ID: {paper_id}
- Title: {title}
- Category: {category}
- Abstract: {abstract}

Generate a reference answer (1-2 sentences in Chinese).

Respond with ONLY a JSON object:
{{"question": "<question text>", "reference_answer": "<reference answer>"}}"""

    selected = random.sample(papers, min(count, len(papers)))
    questions = []
    for i, p in enumerate(selected):
        print(f"  Fact {i+1}/{len(selected)}: {p['paper_id']}", file=sys.stderr)
        prompt = fact_prompt.format(
            paper_id=p["paper_id"],
            title=p["title"],
            category=p["primary_category"],
            abstract=p["abstract"][:600],
        )
        try:
            result = _call_llm(llm, prompt)
            questions.append({
                "qid": f"f{i+1:03d}",
                "query": result["question"],
                "expected_paper_ids": [p["paper_id"]],
                "expected_mode": "answer",
                "reference_answer": result.get("reference_answer", ""),
                "difficulty": "medium",
                "type": "fact_extract",
                "tags": ["single-paper", "semantic", "factual"],
            })
        except Exception as e:
            print(f"  SKIP {p['paper_id']}: {e}", file=sys.stderr)
        time.sleep(0.5)
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval questions from paper metadata")
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(PROJECT_ROOT / "data/metadata_filtered.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "eval/datasets/questions_v2.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept-count", type=int, default=20)
    parser.add_argument("--method-count", type=int, default=15)
    parser.add_argument("--comparison-count", type=int, default=10)
    parser.add_argument("--fact-count", type=int, default=10)
    parser.add_argument("--trend-count", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    papers = load_papers(Path(args.metadata))
    print(f"Loaded {len(papers)} papers", file=sys.stderr)

    llm = _get_llm()
    all_questions: list[dict] = []

    print("\n=== Concept Locate Questions ===", file=sys.stderr)
    all_questions.extend(gen_concept_questions(llm, papers, args.concept_count))

    print("\n=== Method Detail Questions ===", file=sys.stderr)
    all_questions.extend(gen_method_questions(llm, papers, args.method_count))

    print("\n=== Fact Extract Questions ===", file=sys.stderr)
    all_questions.extend(gen_fact_extract_questions(llm, papers, args.fact_count))

    print("\n=== Comparison Questions ===", file=sys.stderr)
    all_questions.extend(gen_comparison_questions(llm, papers, args.comparison_count))

    print("\n=== Trend Synthesis Questions ===", file=sys.stderr)
    all_questions.extend(gen_trend_questions(llm, papers, args.trend_count))

    print("\n=== Negative Questions (static) ===", file=sys.stderr)
    all_questions.extend(NEGATIVE_QUESTIONS)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(all_questions)} questions -> {out}", file=sys.stderr)
    by_type = {}
    for q in all_questions:
        by_type.setdefault(q["type"], []).append(q)
    for t, qs in by_type.items():
        print(f"  {t}: {len(qs)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
