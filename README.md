# SEO Tools

Collection of standalone Streamlit apps for on-page SEO analysis — cosine similarity, entity-topic gaps, and passage-level scoring. Each script runs independently.

## Tools

- **`cosine-similarity-tools.py`** — multi-URL competitor comparison against a search term, with branded output (logo, report styling).
- **`cosine-similarity-paragraphs.py`** — paragraph-level similarity scoring within a single document.
- **`cosine-similarity-competitor-analysis.py`** — head-to-head competitor comparison for a target query.
- **`score-every-embedding.py`** — scores every passage/sentence in a document against a search term.
- **`entity-topic-gaps.py`** — extracts entities from your URL and competitor URLs with spaCy, renders word clouds, and highlights missing topics. Uses Selenium + BeautifulSoup for JS-rendered pages.

## Stack

- Streamlit UI
- `transformers` + PyTorch (`bert-base-uncased`) for embeddings
- spaCy (`en_core_web_sm`) for entity extraction
- Selenium + BeautifulSoup for page fetching (JS-heavy sites)
- WordCloud + Matplotlib for visualizations
- `cairosvg` / `PIL` for branded report rendering

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run cosine-similarity-tools.py
# or any of the other apps
```

`packages.txt` lists the system packages Streamlit Cloud needs for Selenium / Chrome.

## Relationship to other repos

- [`Noob`](https://github.com/rwongphoto/Noob) — earlier scratch versions of these ideas
- [`ai-mode-cosine-similarity`](https://github.com/rwongphoto/ai-mode-cosine-similarity) — newer, AI-search-focused analyzer using MPNet embeddings
- [`entity-gap-analysis`](https://github.com/rwongphoto/entity-gap-analysis) — production version of the entity-gap workflow, backed by Google Cloud NLP
