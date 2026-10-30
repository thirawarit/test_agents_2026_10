# md-rag

Local RAG over mixed Thai/English markdown files. Ask for a summary of a topic
across your files, and get an answer in the language of the question, with
citations at file and heading level. Models run in Ollama on this machine.

## Setup

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
uv sync --python 3.12
ollama pull bge-m3
ollama pull gemma3:latest
cp config.example.toml config.local.toml
```

Set `paths.docs_dir` in `config.local.toml` to your markdown folder. That file
is ignored by git, and paths in it are relative to the file itself.

## Usage

```bash
uv run md-rag doctor                    # check config, Ollama, models, index
uv run md-rag index                     # embed new and changed files only
uv run md-rag index --full              # rebuild everything
uv run md-rag ask "สรุปแนวทางการสำรองข้อมูล"
uv run md-rag ask --show-score "MinIO"  # also print the cutoff decision to stderr
```

The config is read from `--config PATH`, then `$MD_RAG_CONFIG`, then
`./config.local.toml`.

`ask` prints the answer and a source list. Warnings go to stderr.

`--show-score` adds one line to stderr, and the answer and exit code stay the
same:

```
score: best match 0.498, retrieval.min_relevance 0.50, passed on rare term MinIO
```

| Exit code | Meaning |
| --- | --- |
| 0 | Answer has at least one valid citation |
| 1 | Error, or a failed `doctor` check |
| 3 | Answer has no valid citation, so it can't be traced to a document |
| 4 | Topic not covered by the documents: the reply says so and lists no sources |

## How it works

```
retrieve -> summarize -> agent <-> tools -> cite
```

1. **Ingest:** finds `.md` files, skips `REQ.local.md`, and normalizes the text.
2. **Chunk:** splits by heading and keeps the heading path. A section longer
   than `chunk_words` is split on sentences. Code fences and tables stay whole.
3. **Index:** stores vectors in NumPy and chunks plus BM25 tokens in JSON, in
   `.rag_index/`. A rerun embeds only changed files.
4. **Retrieve:** refuses the question if its best dense cosine is below
   `retrieval.min_relevance` and it has no rare literal term from the documents.
   Otherwise it fuses dense and BM25 results, picks varied chunks with MMR, and
   expands them to whole sections.
5. **Summarize:** uses one call if the sections fit, otherwise map-reduce with
   citation numbers preserved.
6. **Agent:** a pass-through while the tool registry is empty.
7. **Cite:** resolves `[n]` markers in code, and removes any that match no passage.

The graph is fixed, so the model can't skip retrieval.

## Tuning

- `retrieval.min_relevance` is 0.50. Use `ask --show-score` to see how your
  questions score. Check this value on your own documents: scores depend on the
  corpus, and two uncovered questions that scored below 0.50 on the fixtures
  scored above it on a larger synthetic corpus. Recheck it if `embed_model`
  changes. Setting it to 0 turns the check off.
- `retrieval.rare_term_max_fraction` is 0.05, and it lets a low-scoring question
  through if it has a rare literal term from the documents. Being in few chunks
  isn't enough, because on a small corpus ordinary words are in few chunks too,
  so a term must also pass one of these checks:
  - Latin terms must be identifier-shaped in the documents: `MinIO`, `SEV3`,
    `pg_size_pretty`, `deploy.sh`, `UTC`. Capitalized names such as `Singapore`
    and plain words such as `canary` don't count.
  - Thai terms must be uncommon, with a count of at most
    `retrieval.rare_term_max_thai_freq` in PyThaiNLP's bundled word list. Every
    other content word in the question must also be in the documents.

  Setting `rare_term_max_fraction` to 0 turns this rule off.
- `retrieval.mmr_keep` is the main control over focus. Lower values keep the
  answer on topic but can drop files from a cross-file summary. On the test
  fixtures, 6 removed off-topic facts but also lost one of the three files
  covering the topic, compared with the default of 24.
- `ollama.num_ctx` is 16384. Prompts are planned to use at most 80% of it, minus
  `num_predict`, because Ollama cuts an overflowing prompt without an error and
  the system instructions can be lost with it. A warning appears if a cut looks
  likely. It's a hint, not proof: Ollama reports no overflow flag, so a healthy
  prompt can occasionally trigger it.
- Changing the embedding model or chunk settings triggers a full rebuild on the
  next `index`.

## Local-only safeguards

- Ollama hosts that aren't loopback are refused, as are cloud model tags. Ollama
  is also asked whether a model really runs locally, which catches aliases of
  cloud models.
- Model clients ignore proxy variables and don't follow redirects, so a prompt
  can't be routed off this machine by the environment.
- LangSmith tracing is forced off.
- `.rag_index/` contains your document text and is ignored by git.

### Warning: future tools

The tool registry in `src/md_rag/tools.py` is empty, so nothing can leave the
machine through a tool today. **Any tool you add can change that.** The model
writes a tool's arguments, and it has read your documents, so a tool that reaches
a network can send document text elsewhere. The checks above only cover Ollama
traffic and can't stop this. A document could also contain text that tries to
steer the model into calling a tool.

Keep any new tool local, or review what it can send before you register it.
gemma3 doesn't support tools either, so you'd need a generation model that does.
`doctor` reports this and fails if tools are registered without support.

## Known limits

- Citations prove that a passage was retrieved, not that it supports the claim.
  An answer can be wrong and still carry a valid citation, so check important
  facts against the source.
- The relevance cutoff is a single score, and it makes mistakes both ways:
  - A question next to a covered topic can pass. A Thai sick leave question
    scored 0.64 against the annual leave section, so it gets an answer that says
    sick leave isn't covered but still cites other sections and exits 0.
  - A question can pass because it contains a rare term, even when the rest of
    it isn't covered. "How big is the SEV1 bonus?" passes on `SEV1`. The Thai word
    list comes from textbooks, so everyday workplace words such as `ทีมงาน` can
    count as rare too.
  - A covered question can be refused. This mostly affects loosely worded Thai
    questions against English text, and bare words that aren't identifier-shaped
    such as `Singapore` or `canary`. If a question is refused, try a full question
    using terms from the documents.
- `gemma3` sometimes includes off-topic facts or blurs details when summarizing
  Thai. See `retrieval.mmr_keep` above.
- Long code fences or tables are kept whole, so their embedding can be cut off at
  the embedding model's context limit. BM25 still covers their full text.

## Tests

```bash
uv run pytest -m "not integration"   # no Ollama needed
uv run pytest -m integration         # needs Ollama with both models
```
