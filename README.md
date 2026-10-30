# Laboratoare LLM — Modele Generative

Repo cu laboratoarele de la cursul **„Modele Generative”**: evaluare de modele LLM locale și un sistem *Retrieval-Augmented Generation* (RAG) pe Codul muncii al Republicii Moldova.

> ⚠️ **Proiect educațional.** Răspunsurile generate de modele nu reprezintă consultanță juridică.

## Structura

```text
.
├── laborator_1/                    # Evaluare modele LLM prin Ollama
│   ├── laborator_1.ipynb           # notebook-ul laboratorului (40 celule)
│   ├── benchmarks/                 # rezultate per model (gemma4, llama3.2)
│   ├── ollama_eval_aggregate.json  # metrici agregate per model
│   ├── ollama_eval_item_level.json # rezultate per întrebare
│   └── LL1. Evaluare modele LLM-1.pdf  # enunțul laboratorului
├── laborator_2/                    # RAG pe Codul muncii al RM (proiect uv separat)
│   ├── notebooks/rag_codul_muncii.ipynb  # notebook principal (123 celule)
│   ├── pyproject.toml / uv.lock    # dependențe + versiuni exacte
│   ├── README.md                   # documentația detaliată a laboratorului 2
│   ├── .env.example                # model de configurare (fără chei)
│   ├── data/processed/             # articole.csv, chunks.jsonl, text curățat
│   ├── data/raw/                   # sursa legii (ignorată de git, se salvează local)
│   └── cache/                      # embeddings .npy (ignorate de git, se regenerează)
├── .gitignore
└── README.md
```

## Laborator 1 — Evaluare modele LLM

Notebook-ul `laborator_1/laborator_1.ipynb` interoghează modele locale prin **Ollama** (`http://localhost:11434`) pe un set de întrebări de cultură generală/istorie și matematică și măsoară:

- acuratețea factuală (exactă și parțială),
- acuratețea la matematică și rata pașilor de rezolvare,
- viteza de generare (tps), latența primului token (TTFT),
- perplexitatea și numărul de tokeni generați.

Rezultatele se salvează în `ollama_eval_aggregate.json`, `ollama_eval_item_level.json` și `benchmarks/`.

**Rulare:**

```bash
# pornește Ollama cu modelele necesare
ollama serve
# apoi rulează notebook-ul
jupyter lab laborator_1/laborator_1.ipynb
```

Dependențe: `torch`, `transformers`, `ollama`, `pandas`, `matplotlib`, `seaborn`, `ipywidgets`, `numpy`.

## Laborator 2 — RAG pe Codul muncii al RM

Sistem RAG scris explicit în Python, **fără LangChain / LlamaIndex**, care compară trei arhitecturi de retrieval:

| Metoda | Retrieval | Ce demonstrează |
|---|---|---|
| **Naive RAG** | doar căutare densă (embeddings + Qdrant) | baseline-ul clasic |
| **Hybrid RAG** | BM25 + căutare densă, fuzionate cu **RRF** | combinarea căutării lexicale cu cea semantică |
| **Advanced RAG** | Hybrid + **re-ranking cu cross-encoder** | rafinarea relevanței |

Stack: Python 3.12 + `uv`, embeddings `intfloat/multilingual-e5-base`, re-ranker `BAAI/bge-reranker-v2-m3`, Qdrant în memorie, `rank-bm25`, LLM `gpt-oss:120b` (Ollama Cloud / OpenRouter), chat **Gradio**.

Documentația completă (instalare, configurare `.env`, obținerea legii de pe legis.md, parametri, evaluare) este în [`laborator_2/README.md`](laborator_2/README.md).

**Rulare rapidă:**

```bash
cd laborator_2
uv sync
uv run python -m ipykernel install --user \
    --name rag-codul-muncii --display-name "RAG Codul Muncii (uv)"
uv run jupyter lab
# deschide notebooks/rag_codul_muncii.ipynb cu kernelul „RAG Codul Muncii (uv)"
```

Ai nevoie de o cheie API (`OLLAMA_API_KEY` sau `OPENROUTER_API_KEY`) în `laborator_2/.env` — vezi `.env.example`. Fără cheie rulează capitolele 1–10 (pregătire, indexare, retrieval); generarea și chat-ul au nevoie de LLM.

## Cerințe

- Python 3.10+ pentru laborator_1, Python **3.12** pentru laborator_2
- [`uv`](https://docs.astral.sh/uv/) pentru laborator_2
- Ollama local (laborator_1) și/sau cheie Ollama Cloud / OpenRouter (laborator_2)
- ~3–5 GB spațiu liber pentru modelele Hugging Face; GPU opțional (pe CPU re-rankerul e lent)

## Fișiere ignorate de git

`.env` (chei API), `.venv/`, `cache/` și `*.npy` (embeddings regenerate), `data/raw/` (sursa HTML a legii), `.ipynb_checkpoints/`, modele mari (`*.gguf`, `*.safetensors`, `*.bin/pt/pth/ckpt/onnx`). Se versionează: notebook-urile, `benchmarks/`, rezultatele `ollama_eval_*.json`, `data/processed/`, `pyproject.toml` + `uv.lock`, `.env.example`.

## Observație: repo imbricat

`laborator_2/` conține propriul depozit git (`.git/`). Dacă vrei un monorepo gestionat din rădăcină, fie șterge `laborator_2/.git` și comite totul în repo-ul părinte, fie declară-l ca submoudul:

```bash
git submodule add <url> laborator_2
```

## Surse

- Codul muncii al Republicii Moldova nr. 154/2003 — [legis.md](https://www.legis.md/cautare/getResults?doc_id=8238&lang=ro)
