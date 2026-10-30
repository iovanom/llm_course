# RAG pe Codul muncii al Republicii Moldova

Laborator pentru cursul **„Modele Generative”**: un sistem *Retrieval-Augmented Generation* (RAG) construit pe **Codul muncii al RM (Legea nr. 154/2003)**, care compară trei arhitecturi de retrieval:

| Metoda | Retrieval | Ce demonstrează |
|---|---|---|
| **Naive RAG** | doar căutare densă (embeddings + Qdrant) | baseline-ul clasic |
| **Hybrid RAG** | BM25 + căutare densă, fuzionate cu **RRF** (Reciprocal Rank Fusion) | combinarea căutării lexicale cu cea semantică |
| **Advanced RAG** | Hybrid RAG + **re-ranking cu cross-encoder** | rafinarea relevanței cu un model mai precis |

Tot procesul este scris explicit în Python, **fără LangChain / LlamaIndex**, ca fiecare pas (curățare, chunking, embeddings, RRF, re-ranking, prompt) să fie vizibil și verificabil.

> ⚠️ **Disclaimer:** proiect educațional. Răspunsurile generate nu reprezintă consultanță juridică.

---

## Cuprins

1. [Arhitectura](#arhitectura)
2. [Stack tehnologic](#stack-tehnologic)
3. [Structura proiectului](#structura-proiectului)
4. [Cerințe](#cerințe)
5. [Instalare cu uv](#instalare-cu-uv)
6. [Configurare (.env)](#configurare-env)
7. [Obținerea Codului muncii](#obținerea-codului-muncii)
8. [Rularea notebook-ului](#rularea-notebook-ului)
9. [Parametri configurabili](#parametri-configurabili)
10. [Conținutul notebook-ului](#conținutul-notebook-ului)
11. [Evaluare](#evaluare)
12. [Chat-ul Gradio](#chat-ul-gradio)
13. [Probleme frecvente](#probleme-frecvente)
14. [Limitări](#limitări)
15. [Ce se predă](#ce-se-predă)

---

## Arhitectura

```text
                ┌──────────────────────────────────────────────┐
  legis.md ──►  │ Achiziție → Curățare → Parsare pe articole   │
  (HTML/TXT)    │            → Chunking structural             │
                └───────────────────────┬──────────────────────┘
                                        │ chunks.jsonl
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        Embeddings (multilingual-e5-base)          Tokenizare + BM25
                    │                                       │
           Qdrant (în memorie)                        BM25Okapi
                    │                                       │
                    └──────────────┬────────────────────────┘
                                   ▼
          ┌────────────────────────────────────────────────┐
          │ Naive:  doar dens                              │
          │ Hybrid: dens + BM25 → RRF                      │
          │ Rerank: RRF → cross-encoder (bge-reranker-v2-m3)│
          └───────────────────────┬────────────────────────┘
                                  ▼ top-k fragmente
                     LLM (Ollama Cloud / OpenRouter)
                                  ▼
                  Răspuns cu citarea articolelor + Chat Gradio
```

---

## Stack tehnologic

| Componentă | Alegere |
|---|---|
| Limbaj / mediu | Python 3.12, gestionat cu [`uv`](https://docs.astral.sh/uv/) |
| Notebook | JupyterLab |
| Embeddings | `intfloat/multilingual-e5-base` (dim. 768, prefixe `query:` / `passage:`) |
| Re-ranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| Bază vectorială | Qdrant, mod **în memorie** (`qdrant-client`) |
| Căutare lexicală | `rank-bm25` (BM25Okapi) |
| Fuziune | RRF implementat manual (`k = 60`) |
| LLM | `gpt-oss:120b` prin Ollama Cloud (implicit) sau `openai/gpt-oss-120b` prin OpenRouter |
| Client LLM | `openai` (API compatibil OpenAI) |
| Interfață | Gradio, afișat inline în notebook |

---

## Structura proiectului

```text
rag-codul-muncii/
├── pyproject.toml          # dependențele (generat de uv)
├── uv.lock                 # versiunile exacte (generat de uv)
├── .env                    # cheile API – NU se urcă în git
├── .env.example            # model pentru .env, fără chei
├── .gitignore
├── README.md
├── data/
│   ├── raw/                # codul_muncii.html / codul_muncii.txt + meta.json
│   └── processed/          # codul_muncii_clean.txt, articole.csv, chunks.jsonl
├── cache/                  # embeddings salvate (.npy)
└── notebooks/
    └── rag_codul_muncii.ipynb
```

Notebook-ul detectează singur rădăcina proiectului: dacă rulează din `notebooks/`, folosește directorul părinte.

---

## Cerințe

- Python **3.12** (uv îl poate instala automat)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) instalat:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- O cheie API pentru **Ollama Cloud** sau **OpenRouter**
- ~3–5 GB spațiu liber (modelele Hugging Face + PyTorch); mai mult dacă se instalează PyTorch cu CUDA
- Conexiune la internet la prima rulare (descărcarea modelelor)
- GPU **opțional** – totul rulează și pe CPU, doar mai lent (în special re-rankerul)

---

## Instalare cu uv

### Varianta A – proiect nou de la zero

```bash
# 1. Proiect nou cu Python 3.12
uv init rag-codul-muncii --python 3.12
cd rag-codul-muncii
rm -f main.py hello.py
mkdir -p data/raw data/processed cache notebooks

# 2. Dependențele principale
uv add requests beautifulsoup4 lxml unidecode \
       numpy pandas matplotlib tqdm python-dotenv \
       sentence-transformers qdrant-client rank-bm25 \
       openai gradio

# 3. Dependențele de dezvoltare
uv add --dev jupyterlab ipykernel ipywidgets

# 4. Kernel Jupyter legat de mediul uv
uv run python -m ipykernel install --user \
    --name rag-codul-muncii --display-name "RAG Codul Muncii (uv)"
```

Apoi copiază `rag_codul_muncii.ipynb` în `notebooks/`.

### Varianta B – proiectul există deja (clonat, cu `pyproject.toml` și `uv.lock`)

```bash
cd rag-codul-muncii
uv sync                    # creează .venv cu versiunile exacte din uv.lock
uv run python -m ipykernel install --user \
    --name rag-codul-muncii --display-name "RAG Codul Muncii (uv)"
```

### Opțional: PyTorch doar pentru CPU (server fără GPU)

Pe Linux, `sentence-transformers` aduce implicit `torch` cu CUDA (câțiva GB). Pentru varianta CPU adaugă în `pyproject.toml`, **înainte** de `uv add` (sau rulează apoi `uv sync`):

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

### Rolul dependențelor

| Pachet | Rol |
|---|---|
| `requests`, `beautifulsoup4`, `lxml` | descărcarea și parsarea HTML-ului |
| `unidecode` | normalizarea diacriticelor pentru BM25 |
| `numpy`, `pandas`, `matplotlib`, `tqdm` | calcul, tabele, grafice, progres |
| `python-dotenv` | citirea cheilor din `.env` |
| `sentence-transformers` | embeddings (`SentenceTransformer`) și re-ranking (`CrossEncoder`) |
| `qdrant-client` | baza vectorială în memorie |
| `rank-bm25` | căutarea lexicală BM25 |
| `openai` | client pentru Ollama Cloud / OpenRouter |
| `gradio` | chat-ul interactiv |
| `jupyterlab`, `ipykernel`, `ipywidgets` *(dev)* | notebook și bare de progres |

---

## Configurare (.env)

Creează fișierul `.env` în **rădăcina proiectului** (lângă `pyproject.toml`):

```bash
# Providerul LLM: ollama | openrouter
LLM_PROVIDER=ollama

# Cheia pentru providerul ales (e suficientă una)
OLLAMA_API_KEY=cheia_ta_ollama
OPENROUTER_API_KEY=cheia_ta_openrouter

# Opțional: suprascrie modelul implicit al providerului
# LLM_MODEL=gpt-oss:120b

# Opțional: altă sursă pentru Codul muncii
# CODUL_MUNCII_URL=https://www.legis.md/cautare/getResults?doc_id=8238&lang=ro
```

| Variabilă | Obligatorie | Implicit | Descriere |
|---|---|---|---|
| `LLM_PROVIDER` | nu | `ollama` | `ollama` sau `openrouter` |
| `OLLAMA_API_KEY` | da, dacă provider = `ollama` | – | cheie Ollama Cloud |
| `OPENROUTER_API_KEY` | da, dacă provider = `openrouter` | – | cheie OpenRouter |
| `LLM_MODEL` | nu | `gpt-oss:120b` / `openai/gpt-oss-120b` | modelul folosit |
| `CODUL_MUNCII_URL` | nu | pagina CP154/2003 pe legis.md | sursa documentului |

Endpoint-urile folosite:

| Provider | `base_url` |
|---|---|
| Ollama Cloud | `https://ollama.com/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

Dacă cheia lipsește, notebook-ul afișează un avertisment: capitolele 1–10 (pregătire, indexare, retrieval) rulează oricum; doar generarea, evaluarea răspunsurilor și chat-ul au nevoie de LLM.

Model pentru `.env.example` (acesta se predă/urcă în git):

```bash
LLM_PROVIDER=ollama
OLLAMA_API_KEY=
OPENROUTER_API_KEY=
# LLM_MODEL=
```

Recomandat în `.gitignore`:

```text
.env
.venv/
cache/
data/raw/
__pycache__/
.ipynb_checkpoints/
```

---

## Obținerea Codului muncii

Prima celulă de date (capitolul 3) **descarcă documentul doar dacă nu există deja** în `data/raw/`.

> ⚠️ Descărcarea automată de pe legis.md **probabil va eșua**: site-ul răspunde cu o verificare Cloudflare (HTTP 403), iar textul legii se încarcă prin JavaScript. În acest caz notebook-ul afișează instrucțiuni și trebuie făcută **o singură dată** salvarea manuală:

**Varianta recomandată – pagina HTML**
1. Deschide în browser [Codul muncii pe legis.md](https://www.legis.md/cautare/getResults?doc_id=8238&lang=ro).
2. Așteaptă să se încarce complet textul legii.
3. `Ctrl+S` → **„Pagină web, completă”** → salvează ca `data/raw/codul_muncii.html`.

HTML-ul păstrează indicii superiori ai articolelor (de ex. 54¹), care sunt importanți la parsare.

**Alternativa – text simplu**
1. Selectează tot textul legii din pagină și copiază-l.
2. Salvează-l ca `data/raw/codul_muncii.txt` (UTF-8).

Celula verifică apoi că fișierul conține cel puțin **300** de apariții „Articolul N”; sub acest prag consideră textul incomplet. Data descărcării și hash-ul SHA-256 se salvează în `data/raw/meta.json`.

---

## Rularea notebook-ului

```bash
cd rag-codul-muncii
uv run jupyter lab
```

1. Deschide `notebooks/rag_codul_muncii.ipynb`.
2. Selectează kernelul **„RAG Codul Muncii (uv)”**.
3. Rulează celulele în ordine (`Run → Run All Cells`).

Ce se întâmplă la prima rulare:
- se descarcă modelele Hugging Face (`multilingual-e5-base`, `bge-reranker-v2-m3`) – câteva minute;
- se calculează embeddings-urile și se salvează în `cache/`; la rulările următoare se încarcă din cache (recalculate doar dacă s-au schimbat chunk-urile);
- se generează fișierele din `data/processed/`.

Pentru predare, notebook-ul trebuie salvat **cu toate output-urile vizibile**.

---

## Parametri configurabili

Toți parametrii sunt într-o singură celulă (**2.2 Constante și parametri**):

| Parametru | Valoare | Rol |
|---|---|---|
| `EMBED_MODEL_NAME` | `intfloat/multilingual-e5-base` | modelul de embeddings |
| `RERANK_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | cross-encoderul |
| `MAX_CHUNK_CHARS` | `1200` | limita unui fragment la chunking structural |
| `FIXED_CHUNK_SIZE` / `FIXED_CHUNK_OVERLAP` | `1000` / `150` | chunking fix (baseline) |
| `TOP_K` | `5` | fragmente trimise în promptul LLM |
| `CANDIDATES` | `20` | candidați per retriever înainte de RRF |
| `RERANK_CANDIDATES` | `15` | candidați trimiși la re-ranker |
| `K_RRF` | `60` | constanta din formula RRF |
| `EVAL_DEPTH` | `10` | adâncimea la evaluare (MRR@10) |
| `LLM_TEMPERATURE` | `0.1` | răspunsuri factuale |
| `LLM_CALL_DELAY` | `0.5` | pauză între apeluri (evită rate-limit) |
| `RUN_GENERATION_EVAL` | `True` | evaluarea răspunsurilor LLM (~54 apeluri API) |
| `RUN_ABLATIONS` | `True` | experimente suplimentare (embeddings în plus) |
| `SEED` | `42` | reproductibilitate |

Pe CPU sau cu limită de apeluri API, setează `RUN_ABLATIONS = False` și/sau `RUN_GENERATION_EVAL = False`.

---

## Conținutul notebook-ului

| Cap. | Titlu | Ce face |
|---|---|---|
| 1 | Introducere și arhitectură | obiectiv, cele trei metode, disclaimer |
| 2 | Configurare | importuri, versiuni, parametri, chei API, seed, CPU/GPU |
| 3 | Achiziția datelor | descărcare condiționată, verificare, metadate |
| 4 | Curățarea textului | HTML → text, normalizare ş/ţ → ș/ț, eliminarea notelor `[... modificat prin LP ...]` |
| 5 | Parsarea structurii juridice | Titlu / Capitol / Secțiune / Articol, validări, analiză exploratorie |
| 6 | Chunking | structural pe articole/alineate cu antet contextual vs. fix; comparație |
| 7 | Embeddings | e5 cu prefixe, normalizare, cache, verificări |
| 8 | Qdrant în memorie | colecție cosinus, căutare, **verificare manuală cu NumPy** |
| 9 | BM25 | tokenizare adaptată pentru română, stopwords, index |
| 10 | Cele trei metode de retrieval | Naive, Hybrid (RRF scris manual + tabel didactic), Rerank |
| 11 | Generarea cu LLM | prompt de sistem, citarea articolelor, refuz în afara contextului |
| 12 | Evaluare | Hit@k, MRR@10, grafice, evaluarea răspunsurilor, ablații |
| 13 | Chat Gradio | reformularea întrebării, alegerea metodei, top-k, surse |
| 14 | Concluzii | rezumatul rezultatelor, limitări, direcții viitoare |

---

## Evaluare

- **Setul de test:** 18 întrebări de trei tipuri – `exact` (termeni juridici), `parafrază` (limbaj colocvial), `capcană` (în afara Codului muncii).
- **Răspunsurile corecte** sunt identificate prin cuvinte-cheie din titlul articolului (`title_keywords`), nu prin numere de articol scrise de mână. **Verifică tabelul afișat în 12.1** și, unde e nevoie, ajustează `title_keywords` sau completează `expected_override`.
- **Metrici de retrieval:** Hit@1, Hit@3, Hit@5, MRR@10 – pe metodă și pe tip de întrebare.
- **Evaluarea generării:** dacă răspunsul citează articolul corect și dacă modelul refuză întrebările capcană.
- **Ablații:** chunking fix vs. structural, valoarea `k` din RRF, chunk-uri cu vs. fără antet contextual.

---

## Chat-ul Gradio

Capitolul 13 pornește un chat **inline** în notebook:
- alegerea metodei: Naive / Hybrid / Rerank;
- slider pentru `top_k`;
- reformularea întrebării pe baza istoricului (întrebări de follow-up);
- sursele (articole + scoruri) afișate sub fiecare răspuns.

Necesită cheia API configurată în `.env`.

---

## Probleme frecvente

| Problemă | Cauză / soluție |
|---|---|
| HTTP 403 sau „Conținutul se încarcă...” la descărcare | Cloudflare / conținut JavaScript → salvare manuală (vezi [Obținerea Codului muncii](#obținerea-codului-muncii)) |
| „Prea puține articole găsite” | pagina salvată incomplet → reîncarcă pagina, așteaptă textul complet, salvează din nou |
| Număr greșit de articole la capitolul 5 | formatul HTML diferă → ajustează `ARTICLE_RE` / `HEADING_RES` |
| Avertisment „cheie API lipsă” | `.env` lipsește sau nu e în rădăcina proiectului; verifică `LLM_PROVIDER` și cheia corespunzătoare |
| Erori 429 (rate limit) | mărește `LLM_CALL_DELAY` sau pune `RUN_GENERATION_EVAL = False` |
| Re-rankerul e foarte lent | normal pe CPU; redu `RERANK_CANDIDATES` |
| Kernelul nu apare în Jupyter | rulează din nou comanda `ipykernel install` și repornește JupyterLab |
| Se descarcă PyTorch cu CUDA (GB) | folosește indexul `pytorch-cpu` și `uv sync` |

---

## Limitări

- Legea se modifică periodic: sistemul folosește versiunea salvată la data din `meta.json`.
- BM25 nu înțelege sinonime; căutarea densă poate rata termenii exacți – de aici motivația pentru Hybrid.
- Re-rankerul este lent pe CPU.
- Setul de test este mic (18 întrebări), deci rezultatele sunt orientative.
- Răspunsurile LLM pot conține erori; nu înlocuiesc un jurist.


---

*Sursa documentului:* Codul muncii al Republicii Moldova nr. 154/2003, [legis.md](https://www.legis.md/cautare/getResults?doc_id=8238&lang=ro).
