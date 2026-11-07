# What Would Lee Kuan Yew Do? - RAG Persona Chatbot

A text chatbot that answers questions (life, geopolitics, history, policy) **in the voice and reasoning style of Lee Kuan Yew**, grounded in his public speeches and writings via Retrieval-Augmented Generation (RAG). Includes an evaluation harness (RAG vs. no-RAG baseline).

> ⚠️ **Disclaimer:** This is an AI simulation for an educational challenge. Outputs are *not* real quotes, may extrapolate, and do not represent the real person.

## Architecture

```
Speeches / articles (.txt, with TITLE/YEAR/URL header)
        │  chunk (800 chars, 120 overlap)
        ▼
 all-MiniLM-L6-v2 embeddings ──► ChromaDB (cosine)
                                       ▲
 User question ── embed ── top-5 ──────┘
        │
        ▼
 Persona system prompt + numbered CONTEXT + chat history ──► Claude ──► answer + cited sources
```

Design choices
- **Source citation**: every chunk keeps title/year/URL; the UI lists retrieved sources.
- **Persona guardrails**: no invented quotes, flags extrapolation, states it cannot know post-March-2015 events, never claims to be the real person, treats retrieved text as data (basic prompt-injection resistance).
- **Multilingual**: replies in the user's language (e.g. Bahasa Indonesia, English).

## Quick start

```bash
git clone <this-repo> && cd lky-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY

# 1) Add public speech URLs to data/sources.csv (title,year,url), then:
python -m src.scrape_speeches   # writes data/raw/*.txt
# (or drop your own .txt files into data/raw/ - optionally with a TITLE:/YEAR:/URL: header)

python -m src.ingest            # build the vector DB
python app.py                   # open http://127.0.0.1:7860
```

## Data & copyright
- Use **publicly available** speeches (e.g. National Archives of Singapore, Singapore Government press releases).
- Memoirs/books are copyrighted: use only copies you legally own, for local use. `data/raw/*` is git-ignored - **full texts are never committed**.

## Evaluation

```bash
python -m eval.run_eval         # -> eval/results.json, eval/results.md
```

25 questions across policy, geopolitics, history, life, philosophy, post-2015 "trap" questions and robustness/prompt-injection tests. An LLM judge scores **faithfulness** (RAG only), **persona consistency** and **honesty/safety**, and the same questions are run **without RAG** as a baseline.

**Results:** _paste the table from `eval/results.md` here after running._

## Limitations
- Quality depends on the corpus you ingest; small corpus = more extrapolation.
- LLM-as-judge is itself imperfect; treat scores as relative, not absolute.
- A simulation cannot capture the real person's actual judgement; it may over-generalise.

## Project layout
```
app.py                  Gradio UI
src/config.py           settings
src/scrape_speeches.py  download public speeches -> data/raw
src/ingest.py           chunk + embed + store
src/chat.py             retrieval + persona prompt + generation
eval/questions.json     test set
eval/run_eval.py        RAG vs baseline evaluation
```
