# CacheMind

A two-tier RAG system with a **learned semantic cache**. A small in-memory
cache answers repeated or paraphrased questions without going back to the
document store, and it learns on its own which eviction rule to trust.

## How it works

```
question ─► embed (MiniLM, 384-d) ─► hot cache (15 chunks, cosine ≥ 0.5)
                                        │
                     ≥ 7 matches ◄──────┴──────► < 7 matches
                         HIT                        MISS
                          │                          │
                          │       spaCy clean + lemmatize the question
                          │       BM25 over MySQL (top 10)
                          │       cross-encoder relevance score
                          │       keep chunks with cosine ≥ 0.5
                          │       cache the best 5 (evict if full)
                          ▼                          ▼
                     LLM (Groq, Llama 3.3 70B) answers from ≤ 7 chunks
```

- **Hot tier:** `SCRLCache`, 15 chunk embeddings in memory. A query is a hit
  when at least 7 cached chunks are similar to it.
- **Cold tier:** every chunk in MySQL, searched with BM25. The query is
  cleaned with spaCy (stop words removed, each word kept along with its
  lemma), so "networks" also matches "network". No vector database and no
  up-front embedding of the corpus.
- **Selective admission:** a BM25 result enters the cache only if its cosine
  similarity to the query is at least 0.5, and at most 5 enter per miss.
- **On a hit** we skip the BM25 scan, the cross-encoder and up to 10 embedding
  calls. The LLM is still called on every question.

### Learned eviction

When the cache is full, three experts each nominate a victim:

| Expert | Evicts |
|---|---|
| SemLRU | the chunk least similar to recent queries |
| LFU | the least frequently used chunk |
| RDGE | like SemLRU, but protects chunks close to the latest cold-tier results |

One expert is chosen by its weight (followed directly once its weight is above 0.6).
The evicted chunk goes into that expert's **ghost list** (FIFO, 15 entries).
If a ghost chunk is later fetched again, that eviction was a mistake and the
expert's weight is reduced with a Hedge / multiplicative-weights update
(the LeCaR approach). Our addition is **graded regret**: the penalty is the
chunk's cross-encoder relevance multiplied by a decay for how long ago it was
evicted, so losing an important chunk recently costs more than losing a weak
one long ago. Weights are clipped to [0.01, 0.99] so no expert is ever
switched off. If all three experts nominate the same chunk, nobody is blamed.

## Results

`benchmarks/synthetic_benchmark.py` runs the real cache classes on a
synthetic workload (40 topics, 2,000 queries, 10 seeds) with the same hit /
admission logic as the app. Hit rate:

| Policy | Static topics | Drifting topics |
|---|---|---|
| SemLRU only | 4.0% | 2.5% |
| LFU only | 29.8% | 5.1% |
| RDGE only | 8.1% | 8.0% |
| Uniform mix (equal weights, no learning) | 16.3% | 15.3% |
| **SCRL (Hedge), learned weights** | **20.3%** | **18.6%** |

LFU wins when topics stay fixed but collapses under drift. The learned
policy stays steady in both and is 2.3× the best single expert under drift;
learning the weights adds 3–4 points over an equal mix.

These numbers measure the eviction policy only, on synthetic vectors. They
are not answer quality or latency on real documents.

## Project structure

```
app.py                      Streamlit app (query flow, hit/miss, live cache stats)
core/
  config.py                 Settings from .env
  main.py                   PDF ingestion: LlamaParse -> chunks -> MySQL
  cold_storage.py           MySQL store + BM25 with lemmatized queries
  embedding_model.py        all-MiniLM-L6-v2 wrapper (normalized embeddings)
  chatbot.py                Groq LLM prompt chain
cache/
  scrl_cache.py             The learned cache (lookup, insert, evict, regret)
  experts.py                SemLRU, LFU and RDGE nominations
  hedge.py                  Expert weights and updates
  history.py                Per-expert ghost queues
  embed_index.py            Cosine-similarity index over cached chunks
  entry.py, structures.py   Cache entry and helper data structures
  forced_expert_cache.py    Single-expert baseline used by the benchmark
benchmarks/
  synthetic_benchmark.py    Eviction-policy benchmark
presentation/
  cachemind.tex             Beamer slides
documents/                  Put your PDFs here (not tracked)
```

## Setup

Tested with Python 3.13. Requires a local MySQL server.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create the database:

```sql
CREATE DATABASE cold_storage;
```

Copy `.env.example` to `.env` and fill in:

| Variable | Needed for |
|---|---|
| `GROQ_API_KEY` | answering questions |
| `LLAMA_CLOUD_API_KEY` | PDF ingestion only |
| `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | cold storage |

## Usage

1. Put PDF files in `documents/` and ingest them:
   ```bash
   python -m core.main
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```
   Each answer shows whether it was a cache hit or miss, the chunks used,
   the BM25 candidates, what was admitted or evicted, and why. The sidebar
   shows the hit rate, expert weights and eviction counts as they change.
3. Run the benchmark (no MySQL or API keys needed):
   ```bash
   python benchmarks/synthetic_benchmark.py
   ```

Tunable constants (`CACHE_SIZE`, `HIT_THRESHOLD`, `INSERT_THRESHOLD`,
`INSERT_COUNT`) are at the top of `app.py`.

## Limitations and next steps

- The LLM is called on hits too; the cache saves retrieval work, not LLM calls.
- End-to-end latency and cost savings on real documents have not been measured yet.
- Cache lookup is a linear scan, fine for 15 slots but not for large caches.
- Planned: recency in SemLRU, softer expert sampling, evaluation on real query logs.

## Credits

- The expert-weighting, ghost-list and regret design follows **LeCaR**:
  G. Vietri et al., "Driving Cache Replacement with ML-based LeCaR",
  USENIX HotStorage 2018. `cache/structures.py` (DequeDict, HeapDict) is
  adapted from the LeCaR codebase.
