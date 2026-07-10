"""Generate semantic evaluation questions from paper metadata using LLM.

Usage:
    cd backend
    python ../eval/scripts/gen_questions.py --output ../eval/datasets/questions_v2.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import time
from collections import Counter
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

FACT_EXTRACT_PROMPT = """You are generating evaluation questions for a RAG system.

Given the following paper, generate ONE specific factual question about the paper's experimental setup, datasets used, baselines compared, or quantitative results mentioned in the paper metadata. The question must:
1. Ask about a concrete, verifiable fact
2. NOT quote the title or abstract verbatim
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
4. Avoid generic wording like "这组论文体现出哪些共同趋势"; name the concrete theme being compared.

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
                return _loads_object(text[start:i + 1])
    raise ValueError(f"Unbalanced JSON in: {text[:200]}")


def _loads_object(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = ast.literal_eval(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


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


def build_generation_plan(
    *,
    total: int,
    concept_count: int,
    method_count: int,
    fact_count: int,
    comparison_count: int,
    trend_count: int,
    negative_count: int,
) -> dict[str, int]:
    plan = {
        "concept_locate": concept_count,
        "method_detail": method_count,
        "fact_extract": fact_count,
        "comparison": comparison_count,
        "trend_synthesis": trend_count,
        "negative": negative_count,
    }
    actual = sum(plan.values())
    if actual != total:
        raise ValueError(f"Question counts must sum to {total}, got {actual}: {plan}")
    return plan


def _is_placeholder_question(item: dict) -> bool:
    query = str(item.get("query") or "").strip()
    reference = str(item.get("reference_answer") or "").strip()
    return (
        not query
        or query in {"<question text>", "<question>", "..."}
        or query.startswith("<question")
        or reference in {"<reference answer>", "..."}
    )


def validate_question_set(
    questions: list[dict],
    *,
    expected_plan: dict[str, int],
) -> dict:
    qids = [str(q.get("qid") or "") for q in questions]
    if len(qids) != len(set(qids)):
        duplicates = [qid for qid, n in Counter(qids).items() if n > 1]
        raise ValueError(f"Duplicate qid(s): {duplicates[:5]}")

    required = {
        "qid",
        "query",
        "expected_paper_ids",
        "expected_mode",
        "reference_answer",
        "difficulty",
        "type",
        "tags",
    }
    for q in questions:
        missing = sorted(required - set(q))
        if missing:
            raise ValueError(f"{q.get('qid', '<missing qid>')} missing fields: {missing}")
        if _is_placeholder_question(q):
            raise ValueError(f"{q['qid']} contains placeholder query/reference")
        if q["type"] == "negative":
            if q["expected_paper_ids"] != [] or q["expected_mode"] != "insufficient":
                raise ValueError(f"{q['qid']} negative question has invalid expectation")
        else:
            if not q["expected_paper_ids"] or q["expected_mode"] != "answer":
                raise ValueError(f"{q['qid']} positive question has invalid expectation")

    by_type = Counter(q["type"] for q in questions)
    for qtype, expected in expected_plan.items():
        actual = by_type.get(qtype, 0)
        if actual != expected:
            raise ValueError(f"{qtype} expected {expected}, got {actual}")

    return {
        "total": len(questions),
        "positive": sum(1 for q in questions if q["type"] != "negative"),
        "negative": by_type.get("negative", 0),
        "by_type": dict(by_type),
        "by_difficulty": dict(Counter(q["difficulty"] for q in questions)),
    }


def _fallback_reference(paper: dict) -> str:
    title = paper.get("title", "该论文")
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        return f"该问题对应论文《{title}》。论文摘要显示，其核心内容是：{abstract[:180]}。"
    return f"该问题对应论文《{title}》，需要结合论文内容回答其方法和贡献。"


def _fallback_single_paper_question(paper: dict, qtype: str) -> tuple[str, str]:
    title = paper.get("title", "该论文")
    if qtype == "method_detail":
        query = f"论文《{title}》采用了什么核心方法或模型结构来解决其研究问题？"
    elif qtype == "fact_extract":
        query = f"论文《{title}》在实验设置、数据集、基线或结果方面给出了哪些具体事实？"
    else:
        query = f"哪篇论文主要研究了《{title}》所涉及的任务或方法方向？"
    return query, _fallback_reference(paper)


def _fallback_comparison_question(a: dict, b: dict) -> tuple[str, str]:
    title_a = a.get("title", "Paper A")
    title_b = b.get("title", "Paper B")
    query = f"《{title_a}》和《{title_b}》在研究目标、技术路线和适用场景上有什么差异？"
    answer = (
        f"该问题需要对比两篇论文：《{title_a}》和《{title_b}》。"
        "回答应分别说明两者关注的问题、采用的方法以及实验或应用侧重点。"
    )
    return query, answer


def _fallback_trend_question(topic: str, group: list[dict]) -> tuple[str, str, list[str]]:
    selected = group[: min(5, len(group))]
    ids = [p["paper_id"] for p in selected]
    titles = "、".join(f"《{p['title']}》" for p in selected[:3])
    query = f"围绕{topic}方向，这组论文体现出哪些共同趋势、方法差异和研究重点？"
    answer = f"该问题需要综合分析{titles}等论文，归纳它们在{topic}方向上的共同技术主题和差异。"
    return query, answer, ids


def group_papers_by_topic(papers: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for p in papers:
        cat = p["primary_category"]
        groups.setdefault(cat, []).append(p)
    return groups


def _dedupe_papers(papers: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for p in papers:
        pid = p.get("paper_id")
        if pid and pid not in seen:
            out.append(p)
            seen.add(pid)
    return out


def _paper_search_text(paper: dict) -> str:
    bits = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("primary_category", ""),
        paper.get("corpus_bucket", ""),
        " ".join(paper.get("categories") or []),
    ]
    return " ".join(str(b) for b in bits).lower()


def build_trend_topic_groups(papers: list[dict], count: int) -> list[tuple[str, list[dict]]]:
    """Build enough multi-paper groups for trend synthesis questions."""
    groups: list[tuple[str, list[dict]]] = []
    seen_topics: set[str] = set()

    def add(topic: str, group: list[dict]) -> None:
        if len(groups) >= count:
            return
        clean = _dedupe_papers(group)
        if len(clean) < 2 or topic in seen_topics:
            return
        groups.append((topic, clean[:8]))
        seen_topics.add(topic)

    by_bucket: dict[str, list[dict]] = {}
    for p in papers:
        bucket = p.get("corpus_bucket")
        if bucket:
            by_bucket.setdefault(str(bucket), []).append(p)
    for bucket, group in sorted(by_bucket.items(), key=lambda x: (-len(x[1]), x[0])):
        add(f"{bucket} 方向", group)

    by_category = group_papers_by_topic(papers)
    for cat, group in sorted(by_category.items(), key=lambda x: (-len(x[1]), x[0])):
        add(f"{cat} 方向", group)

    keyword_specs = [
        ("大语言模型与推理", ["llm", "language model", "reasoning", "chain-of-thought", "inference"]),
        ("智能体与工具使用", ["agent", "agentic", "tool", "web agent", "multi-agent"]),
        ("检索与信息获取", ["retrieval", "search", "rag", "information retrieval", "ranking"]),
        ("评估、基准与裁判模型", ["benchmark", "evaluation", "judge", "eval", "metric"]),
        ("多模态与视觉理解", ["multimodal", "vision", "image", "video", "visual"]),
        ("训练优化与模型压缩", ["optimization", "optimizer", "training", "compress", "pruning", "distillation"]),
        ("安全、鲁棒性与隐私", ["safety", "robust", "privacy", "attack", "alignment"]),
        ("强化学习与奖励建模", ["reinforcement", "reward", "rl", "policy"]),
        ("知识、图结构与推理", ["knowledge", "graph", "reasoning", "symbolic"]),
        ("人机交互与应用系统", ["human", "interface", "interaction", "application", "system"]),
    ]
    for topic, keywords in keyword_specs:
        matched = [p for p in papers if any(kw in _paper_search_text(p) for kw in keywords)]
        add(topic, matched)

    # Large categories can support several non-identical trend questions.
    round_idx = 1
    while len(groups) < count:
        added_this_round = False
        for cat, group in sorted(by_category.items(), key=lambda x: (-len(x[1]), x[0])):
            if len(group) < 4:
                continue
            start = ((round_idx - 1) * 5) % len(group)
            subset = group[start:start + 8]
            if len(subset) < 4:
                subset = group[:8]
            before = len(groups)
            add(f"{cat} 主题簇 {round_idx}", subset)
            added_this_round = added_this_round or len(groups) > before
            if len(groups) >= count:
                break
        if not added_this_round:
            break
        round_idx += 1

    return groups[:count]


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
    llm: ChatOpenAI, papers: list[dict], count: int = 20, sleep_sec: float = 0.5
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
        fallback_question, fallback_answer = _fallback_single_paper_question(p, "concept_locate")
        try:
            result = _call_llm(llm, prompt)
            question = result.get("question") or fallback_question
            reference_answer = result.get("reference_answer") or fallback_answer
        except Exception as e:
            print(f"  FALLBACK {p['paper_id']}: {e}", file=sys.stderr)
            question = fallback_question
            reference_answer = fallback_answer
        questions.append({
            "qid": f"c{i+1:03d}",
            "query": question,
            "expected_paper_ids": [p["paper_id"]],
            "expected_mode": "answer",
            "reference_answer": reference_answer,
            "difficulty": "easy",
            "type": "concept_locate",
            "tags": ["single-paper", "semantic"],
        })
        time.sleep(sleep_sec)
    return questions


def gen_method_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 15, sleep_sec: float = 0.5
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
        fallback_question, fallback_answer = _fallback_single_paper_question(p, "method_detail")
        try:
            result = _call_llm(llm, prompt)
            question = result.get("question") or fallback_question
            reference_answer = result.get("reference_answer") or fallback_answer
        except Exception as e:
            print(f"  FALLBACK {p['paper_id']}: {e}", file=sys.stderr)
            question = fallback_question
            reference_answer = fallback_answer
        questions.append({
            "qid": f"m{i+1:03d}",
            "query": question,
            "expected_paper_ids": [p["paper_id"]],
            "expected_mode": "answer",
            "reference_answer": reference_answer,
            "difficulty": "medium",
            "type": "method_detail",
            "tags": ["single-paper", "semantic", "detail"],
        })
        time.sleep(sleep_sec)
    return questions


def gen_comparison_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 10, sleep_sec: float = 0.5
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
        fallback_question, fallback_answer = _fallback_comparison_question(a, b)
        try:
            result = _call_llm(llm, prompt)
            question = result.get("question") or fallback_question
            reference_answer = result.get("reference_answer") or fallback_answer
        except Exception as e:
            print(f"  FALLBACK: {e}", file=sys.stderr)
            question = fallback_question
            reference_answer = fallback_answer
        questions.append({
            "qid": f"x{i+1:03d}",
            "query": question,
            "expected_paper_ids": [a["paper_id"], b["paper_id"]],
            "expected_mode": "answer",
            "reference_answer": reference_answer,
            "difficulty": "hard",
            "type": "comparison",
            "tags": ["pairwise", "semantic"],
        })
        time.sleep(sleep_sec)
    return questions


def gen_trend_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 5, sleep_sec: float = 0.5
) -> list[dict]:
    questions = []
    topic_groups = build_trend_topic_groups(papers, count)
    for idx, (topic, group) in enumerate(topic_groups, start=1):
        papers_list = "\n".join(
            f"- [{p['paper_id']}] {p['title']}: {str(p.get('abstract') or '')[:260]}"
            for p in group[:8]
        )
        print(f"  Trend {idx}/{count}: {topic} ({len(group)} papers)", file=sys.stderr)
        prompt = TREND_PROMPT.format(
            topic=topic,
            papers_list=papers_list,
        )
        fallback_question, fallback_answer, fallback_ids = _fallback_trend_question(topic, group)
        try:
            result = _call_llm(llm, prompt)
            question = result.get("question") or fallback_question
            reference_answer = result.get("reference_answer") or fallback_answer
            relevant_ids = result.get("relevant_paper_ids") or fallback_ids
        except Exception as e:
            print(f"  FALLBACK {topic}: {e}", file=sys.stderr)
            question = fallback_question
            reference_answer = fallback_answer
            relevant_ids = fallback_ids
        questions.append({
            "qid": f"t{idx:03d}",
            "query": question,
            "expected_paper_ids": relevant_ids,
            "expected_mode": "answer",
            "reference_answer": reference_answer,
            "difficulty": "hard",
            "type": "trend_synthesis",
            "tags": ["multi-paper", "semantic", "synthesis"],
        })
        time.sleep(sleep_sec)
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

EXTRA_NEGATIVE_QUESTIONS = [
    {
        "query": "这些论文中有哪些关于量子纠错码和容错量子计算硬件实现的研究？",
        "reference_answer": "本语料库中没有关于量子纠错码或容错量子计算硬件实现的论文。",
        "difficulty": "easy",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "query": "请总结语料库中关于农业病虫害遥感监测和作物产量预测的论文。",
        "reference_answer": "本语料库中没有关于农业病虫害遥感监测或作物产量预测的论文。",
        "difficulty": "easy",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "query": "这些论文是否讨论了区块链共识协议在供应链金融中的实际部署？",
        "reference_answer": "本语料库中没有关于区块链共识协议在供应链金融中实际部署的论文。",
        "difficulty": "easy",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "query": "语料库中有哪些论文专门研究医学影像分割模型在临床多中心数据上的泛化问题？",
        "reference_answer": "语料库中可能包含视觉或多模态论文，但没有专门研究医学影像分割临床多中心泛化的论文。",
        "difficulty": "medium",
        "tags": ["negative", "near-miss"],
    },
    {
        "query": "请归纳这些论文中关于数据库事务隔离级别和并发控制协议优化的工作。",
        "reference_answer": "本语料库中没有专门讨论数据库事务隔离级别或并发控制协议优化的论文。",
        "difficulty": "medium",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "query": "有哪些论文研究了机器人抓取策略在真实仓储机械臂上的 sim-to-real 迁移？",
        "reference_answer": "本语料库中没有关于仓储机械臂抓取策略 sim-to-real 迁移的论文。",
        "difficulty": "medium",
        "tags": ["negative", "near-miss"],
    },
    {
        "query": "这些论文中是否包含关于边缘设备上联邦推荐系统压缩与个性化部署的研究？",
        "reference_answer": "语料库中可能有联邦学习或推荐相关近邻主题，但没有专门讨论边缘设备联邦推荐系统压缩与个性化部署的论文。",
        "difficulty": "medium",
        "tags": ["negative", "near-miss"],
    },
    {
        "query": "请比较语料库中关于神经渲染、三维重建和实时 SLAM 融合系统的最新方法。",
        "reference_answer": "本语料库中没有足够论文支持对神经渲染、三维重建和实时SLAM融合系统进行比较。",
        "difficulty": "hard",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "query": "有哪些论文系统评估了大模型在法律判决预测中的公平性、可解释性和跨地区迁移能力？",
        "reference_answer": "本语料库中没有系统评估大模型在法律判决预测中公平性、可解释性和跨地区迁移能力的论文。",
        "difficulty": "hard",
        "tags": ["negative", "out-of-scope"],
    },
    {
        "query": "这些论文是否提出了面向电力调度的多智能体强化学习安全约束优化框架？",
        "reference_answer": "语料库中可能包含强化学习或多智能体论文，但没有面向电力调度安全约束优化的专门研究。",
        "difficulty": "hard",
        "tags": ["negative", "near-miss"],
    },
]


def make_negative_questions(count: int = 10) -> list[dict]:
    bank = NEGATIVE_QUESTIONS + EXTRA_NEGATIVE_QUESTIONS
    if count > len(bank):
        raise ValueError(f"Only {len(bank)} negative questions are available, got {count}")
    questions = []
    for i, item in enumerate(bank[:count], start=1):
        q = dict(item)
        q["qid"] = f"n{i:03d}"
        q["expected_paper_ids"] = []
        q["expected_mode"] = "insufficient"
        q["type"] = "negative"
        q["tags"] = list(dict.fromkeys(["negative", *q.get("tags", [])]))
        questions.append(q)
    return questions


def gen_fact_extract_questions(
    llm: ChatOpenAI, papers: list[dict], count: int = 10, sleep_sec: float = 0.5
) -> list[dict]:
    """Generate questions asking for specific facts from papers."""
    selected = random.sample(papers, min(count, len(papers)))
    questions = []
    for i, p in enumerate(selected):
        print(f"  Fact {i+1}/{len(selected)}: {p['paper_id']}", file=sys.stderr)
        prompt = FACT_EXTRACT_PROMPT.format(
            paper_id=p["paper_id"],
            title=p["title"],
            category=p["primary_category"],
            abstract=p["abstract"][:600],
        )
        fallback_question, fallback_answer = _fallback_single_paper_question(p, "fact_extract")
        try:
            result = _call_llm(llm, prompt)
            question = result.get("question") or fallback_question
            reference_answer = result.get("reference_answer") or fallback_answer
        except Exception as e:
            print(f"  FALLBACK {p['paper_id']}: {e}", file=sys.stderr)
            question = fallback_question
            reference_answer = fallback_answer
        questions.append({
            "qid": f"f{i+1:03d}",
            "query": question,
            "expected_paper_ids": [p["paper_id"]],
            "expected_mode": "answer",
            "reference_answer": reference_answer,
            "difficulty": "medium",
            "type": "fact_extract",
            "tags": ["single-paper", "semantic", "factual"],
        })
        time.sleep(sleep_sec)
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
    parser.add_argument("--negative-count", type=int, default=10)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    papers = load_papers(Path(args.metadata))
    print(f"Loaded {len(papers)} papers", file=sys.stderr)
    total = args.total or (
        args.concept_count
        + args.method_count
        + args.fact_count
        + args.comparison_count
        + args.trend_count
        + args.negative_count
    )
    expected_plan = build_generation_plan(
        total=total,
        concept_count=args.concept_count,
        method_count=args.method_count,
        fact_count=args.fact_count,
        comparison_count=args.comparison_count,
        trend_count=args.trend_count,
        negative_count=args.negative_count,
    )
    print(f"Generation plan: {expected_plan}", file=sys.stderr)

    llm = _get_llm()
    all_questions: list[dict] = []

    print("\n=== Concept Locate Questions ===", file=sys.stderr)
    all_questions.extend(
        gen_concept_questions(llm, papers, args.concept_count, args.sleep_sec)
    )

    print("\n=== Method Detail Questions ===", file=sys.stderr)
    all_questions.extend(
        gen_method_questions(llm, papers, args.method_count, args.sleep_sec)
    )

    print("\n=== Fact Extract Questions ===", file=sys.stderr)
    all_questions.extend(
        gen_fact_extract_questions(llm, papers, args.fact_count, args.sleep_sec)
    )

    print("\n=== Comparison Questions ===", file=sys.stderr)
    all_questions.extend(
        gen_comparison_questions(llm, papers, args.comparison_count, args.sleep_sec)
    )

    print("\n=== Trend Synthesis Questions ===", file=sys.stderr)
    all_questions.extend(
        gen_trend_questions(llm, papers, args.trend_count, args.sleep_sec)
    )

    print("\n=== Negative Questions (static) ===", file=sys.stderr)
    all_questions.extend(make_negative_questions(args.negative_count))

    summary = validate_question_set(all_questions, expected_plan=expected_plan)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(all_questions)} questions -> {out}", file=sys.stderr)
    print(f"  positive: {summary['positive']}", file=sys.stderr)
    print(f"  negative: {summary['negative']}", file=sys.stderr)
    for t, n in summary["by_type"].items():
        print(f"  {t}: {n}", file=sys.stderr)
    for d, n in summary["by_difficulty"].items():
        print(f"  difficulty/{d}: {n}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
