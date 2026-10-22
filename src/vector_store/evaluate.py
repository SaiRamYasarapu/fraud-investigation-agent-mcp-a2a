"""
Retrieval evaluation: how well does each search setup find the right fraud case?

data/eval_queries.json holds 63 labelled questions. Each one lists the case IDs
a good search should return. Four kinds:
- paraphrase (37): one case described in different words; still shares some
  words with the case text.
- exact_id (5): a case ID such as "FC-1043" (tests exact-word matching).
- lookalike (9): a legitimate case that resembles fraud, such as a password
  reset followed by normal bill payments (tests telling look-alikes apart).
- hard_paraphrase (12): a fraud type described with almost no words in common
  with the case text, such as "launder" for a money mule (tests real meaning
  matching; every case of that type counts as relevant).

Every query runs through four setups, and each is scored on the top 5 results:
- Recall@5: share of the relevant cases that appear in the top 5.
- MRR: 1 / position of the first relevant case (1.0 = it came first).
- nDCG@5: rewards relevant cases near the top more than lower down.

Setups: vector only, BM25 only, hybrid (vector + BM25 fused by RRF), and hybrid
plus rerank. Offline, the reranker keeps the hybrid order, so the last two
match; set RERANK_MODE=bedrock to score the live Bedrock reranker.

Usage:
    python -m src.vector_store.evaluate
    python -m src.vector_store.evaluate --misses   # also list queries each setup got wrong
    python -m src.vector_store.evaluate --stats    # also add 95% ranges and paired tests
"""
from __future__ import annotations

import argparse
import json
import math

from src.vector_store.hybrid import HybridCaseRetriever
from src.vector_store.rerank import RerankedCaseRetriever, rerank_mode
from src.vector_store.retriever import DATA_DIR, get_case_retriever
from src.vector_store.significance import bootstrap_ci, paired_test

EVAL_PATH = DATA_DIR / "eval_queries.json"
K = 5


def recall_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    """Share of the relevant items found in the top k (capped so a query with
    more than k relevant items can still score 1.0)."""
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(len(relevant), k)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant item, or 0 if none was returned."""
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    """Discounted gain of the top k (1 per relevant item, divided by log2 of
    position + 1), scaled by the best gain possible for this query."""
    dcg = sum(1.0 / math.log2(pos + 1) for pos, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


class BM25Only:
    """BM25 lane of the hybrid retriever on its own, returning case IDs."""

    def __init__(self, hybrid: HybridCaseRetriever):
        self.hybrid = hybrid

    def ranked_ids(self, query: str, k: int) -> list[str]:
        return [self.hybrid.cases[i]["case_id"] for i, _ in self.hybrid.bm25.top(query, k)]


class Retrieves:
    """Wraps any retriever with .retrieve(query, top_k) so it returns case IDs."""

    def __init__(self, retriever):
        self.retriever = retriever

    def ranked_ids(self, query: str, k: int) -> list[str]:
        return [r.case_id for r in self.retriever.retrieve(query, top_k=k)]


def build_setups() -> dict:
    vector = get_case_retriever()
    hybrid = HybridCaseRetriever(vector_retriever=vector)
    return {
        "vector": Retrieves(vector),
        "bm25": BM25Only(hybrid),
        "hybrid": Retrieves(hybrid),
        "hybrid+rerank": Retrieves(RerankedCaseRetriever(base=hybrid)),
    }


def evaluate(setups: dict, queries: list[dict], k: int = K) -> dict:
    """Average Recall@k, MRR and nDCG@k per setup, overall and per query kind,
    plus the IDs of the queries each setup missed (no relevant case in the top k).
    The per-query scores are kept too, in query order, for the significance checks."""
    report = {}
    for name, setup in setups.items():
        rows = []
        for q in queries:
            ranked = setup.ranked_ids(q["query"], k)
            relevant = set(q["relevant"])
            rows.append({
                "id": q["id"], "kind": q["kind"],
                "recall": recall_at_k(ranked, relevant, k),
                "mrr": reciprocal_rank(ranked, relevant),
                "ndcg": ndcg_at_k(ranked, relevant, k),
            })
        groups = {"all": rows}
        for kind in sorted({r["kind"] for r in rows}):
            groups[kind] = [r for r in rows if r["kind"] == kind]
        report[name] = {
            group: {m: round(sum(r[m] for r in rs) / len(rs), 3) for m in ("recall", "mrr", "ndcg")} | {"n": len(rs)}
            for group, rs in groups.items()
        }
        report[name]["misses"] = [r["id"] for r in rows if r["recall"] == 0]
        report[name]["per_query"] = rows
    return report


def print_report(report: dict) -> None:
    groups = [g for g in next(iter(report.values())) if g not in ("misses", "per_query")]
    for group in groups:
        n = next(iter(report.values()))[group]["n"]
        print(f"\n{group} ({n} queries)")
        print(f"  {'setup':<15}{'Recall@5':>10}{'MRR':>8}{'nDCG@5':>9}")
        for name, res in report.items():
            r = res[group]
            print(f"  {name:<15}{r['recall']:>10.3f}{r['mrr']:>8.3f}{r['ndcg']:>9.3f}")


COMPARISONS = [("bm25", "vector"), ("hybrid", "vector"), ("hybrid", "bm25"), ("hybrid+rerank", "hybrid")]


def print_significance(report: dict, metric: str = "ndcg") -> None:
    """95% range for each setup's average, then paired tests between setups."""
    scores = {name: [r[metric] for r in res["per_query"]] for name, res in report.items()}
    print(f"\n{metric} with 95% bootstrap ranges")
    for name, values in scores.items():
        ci = bootstrap_ci(values)
        print(f"  {name:<15}{ci.mean:>7.3f}  [{ci.low:.3f}, {ci.high:.3f}]")
    print(f"\npaired tests on {metric} (difference = first minus second)")
    for a, b in COMPARISONS:
        if a in scores and b in scores:
            t = paired_test(scores[a], scores[b])
            if t.low == t.high == 0:
                verdict = "identical"
            else:
                verdict = "real difference" if t.significant else "could be luck"
            print(f"  {a} vs {b}: {t.mean_diff:+.3f} [{t.low:+.3f}, {t.high:+.3f}]  "
                  f"p = {t.p_value:.3f}  ({verdict})")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Score retrieval setups on labelled queries.")
    parser.add_argument("--misses", action="store_true", help="list queries with no relevant case in the top 5")
    parser.add_argument("--stats", action="store_true", help="add 95%% ranges and paired tests on nDCG@5")
    args = parser.parse_args()

    with open(EVAL_PATH) as f:
        queries = json.load(f)
    print(f"{len(queries)} queries, rerank mode: {rerank_mode()}")
    report = evaluate(build_setups(), queries)
    print_report(report)
    if args.stats:
        print_significance(report)
    if args.misses:
        print()
        for name, res in report.items():
            print(f"{name} missed: {', '.join(res['misses']) or 'none'}")
