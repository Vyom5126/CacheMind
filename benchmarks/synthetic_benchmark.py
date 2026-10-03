"""
Synthetic-trace benchmark of the hot-cache eviction policies.

Runs the real cache classes (SCRLCache, ForcedExpertCache) through the same
hit / pool / insert logic as app.py, but with synthetic unit vectors instead
of MySQL, BM25, the embedding model and the cross-encoder. It measures the
eviction policy only; it says nothing about answer quality or latency.

Run from the project root:  python benchmarks/synthetic_benchmark.py
"""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cache.scrl_cache import SCRLCache
from cache.forced_expert_cache import ForcedExpertCache

# same constants as app.py
CACHE_SIZE, HIT_THRESHOLD, INSERT_THRESHOLD, INSERT_COUNT = 15, 7, 0.5, 5
DIM = 384
N_TOPICS, DOCS_PER_TOPIC, SIGMA = 40, 10, 0.7
N_QUERIES, PHASE, ZIPF_ALPHA = 2000, 250, 1.2
SEEDS = range(10)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_corpus(rng):
    centres = [unit(rng.normal(size=DIM)) for _ in range(N_TOPICS)]
    docs = {}
    for t, c in enumerate(centres):
        for j in range(DOCS_PER_TOPIC):
            docs[f"{t}_{j}"] = unit(c + SIGMA * unit(rng.normal(size=DIM)))
    return centres, docs


def make_trace(rng, drifting):
    ranks = np.arange(1, N_TOPICS + 1)
    p = 1 / ranks ** ZIPF_ALPHA
    p /= p.sum()
    order = rng.permutation(N_TOPICS)
    trace = []
    for i in range(N_QUERIES):
        if drifting and i % PHASE == 0 and i > 0:
            order = rng.permutation(N_TOPICS)     # popularity reshuffles
        trace.append(int(order[rng.choice(N_TOPICS, p=p)]))
    return trace


def run(cache, centres, docs, trace, rng, np_seed):
    np.random.seed(np_seed)                        # Hedge sampling
    hits = 0
    for topic in trace:
        q = unit(centres[topic] + SIGMA * unit(rng.normal(size=DIM)))
        hot = cache.request(q)
        if len(hot) >= HIT_THRESHOLD:
            hits += 1
            continue
        # "cold retrieval": the topic's chunks stand in for the BM25 top-10
        pool = {d: (s, None) for d, s in hot}
        for j in range(DOCS_PER_TOPIC):
            d = f"{topic}_{j}"
            s = float(docs[d] @ q)
            if s >= INSERT_THRESHOLD and (d not in pool or s > pool[d][0]):
                pool[d] = (s, docs[d])
        ranked = sorted(pool.items(), key=lambda kv: kv[1][0], reverse=True)[:HIT_THRESHOLD]
        new = [(d, e, s) for d, (s, e) in ranked if e is not None][:INSERT_COUNT]
        for d, e, s in new:
            cache.insert(d, e, ce_score=s)         # similarity stands in for CE score
        if new:
            cache.update_retrieved([e for _, e, _ in new])
    return hits / len(trace)


def main():
    policies = {
        "SemLRU only": lambda: ForcedExpertCache(capacity=CACHE_SIZE, hit_threshold=HIT_THRESHOLD, forced_expert=0),
        "LFU only":    lambda: ForcedExpertCache(capacity=CACHE_SIZE, hit_threshold=HIT_THRESHOLD, forced_expert=1),
        "RDGE only":   lambda: ForcedExpertCache(capacity=CACHE_SIZE, hit_threshold=HIT_THRESHOLD, forced_expert=2),
        # same three experts, weights frozen at 1/3 each (no learning)
        "Uniform mix":  lambda: SCRLCache(capacity=CACHE_SIZE, hit_threshold=HIT_THRESHOLD, learning_rate=0.0),
        "SCRL (Hedge)": lambda: SCRLCache(capacity=CACHE_SIZE, hit_threshold=HIT_THRESHOLD),
    }
    for label, drifting in (("static", False), ("drifting", True)):
        print(f"\n{label} trace: {N_QUERIES} queries, {len(list(SEEDS))} seeds")
        for name, make in policies.items():
            rates = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                centres, docs = make_corpus(rng)
                trace = make_trace(rng, drifting)
                rates.append(run(make(), centres, docs, trace, np.random.default_rng(1000 + seed), seed))
            print(f"  {name:13s} hit rate {100*np.mean(rates):5.1f}%  (sd {100*np.std(rates):.1f})")


if __name__ == "__main__":
    main()
