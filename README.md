# MemoryLens 📸

**Drop in a folder of unsorted photos — get back the story of your life.**

MemoryLens is an AI-powered personal memory reconstruction system. Given an
unordered pile of photos, it understands what's in them, groups them into
real-life events (birthdays, weddings, trips...), rebuilds a timeline, and
writes a short narrative for every event — all with pretrained models, zero
training, and no manual labels.


## 🎥 Demo



## What it does

| Step | Model / Technique | Output |
|------|-------------------|--------|
| 1. Semantic embeddings | CLIP ViT-B/32 | `embeddings/embeddings.npy` — one 512-d vector per photo |
| 2. Scene recognition | ResNet-18 · Places365 | `scene_recognition/scene_predictions.json` — Beach, Airport, Museum... |
| 3. Image captioning | BLIP | `captioning/captions.json` — one sentence per photo |
| 4. Similarity search | FAISS | image-to-image, upload-to-image and **text-to-image** search |
| 5. Event clustering | Agglomerative (or HDBSCAN) over image+caption embeddings | `event_clustering/clusters.json` |
| 6. Timeline | EXIF → file time → similarity chaining | `timeline/timeline.json` |
| 7. Story generation | Llama-3.3-70B via Groq | `story_generation/stories.json` — a title + story per event |
| 8. Web interface | Gradio | interactive Memory Book, search, and reprocessing |



## Quick start

```bash
git clone https://github.com/mariemmohsen/MemoryLens.git
cd MemoryLens
pip install -r requirements.txt

# one-time: download the Places365 pretrained weights (~45 MB)
python -m utils.download_places365
```

Create a `.env` file in the project root for story generation
(get a free key at https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_your_key_here
```

Put your photos in `data/raw_images/` (a demo dataset is included), then run
the pipeline:

```bash
python -m embeddings.generate_embeddings
python -m scene_recognition.run_scene_recognition
python -m captioning.run_captioning
python -m similarity_search.build_index
python -m event_clustering.run_clustering
python -m timeline.run_timeline
python -m story_generation.run_story_generation



Launch the app:

```bash
python -m ui.app        # then open http://127.0.0.1:7860
```

The UI can also add photos and re-run the whole pipeline itself
(**➕ Add Photos & Reprocess** tab) with a live progress bar.

## The interface

- **📖 Memory Book** — your reconstructed timeline as chapters: each event has
  an AI-written title and story, its photos, detected scene and confidence.
- **🔍 Search** — describe a moment in words (*"birthday cake with candles"*),
  pick an existing photo, or upload a new one; CLIP + FAISS find the moments
  that *feel* the same.
- **➕ Add Photos & Reprocess** — run the full pipeline from the browser.

## Design decisions worth knowing

- **Why Agglomerative clustering instead of HDBSCAN?** HDBSCAN is
  density-based: small, visually-diverse event groups (5 photos from one
  birthday) never reach the density of large near-duplicate photo pools, so it
  marks every real event as noise (0/10 recovered in our tests). A flat
  cosine-distance threshold recovers them. HDBSCAN is still available via
  `EVENT_CLUSTERING_ALGORITHM = "hdbscan"` in `config.py`.
- **Captions as a clustering signal.** Clustering runs on the *sum* of CLIP
  image embeddings and CLIP text embeddings of each photo's BLIP caption —
  words like "cake" or "christmas tree" separate events that pixels alone
  can't.
- **BLIP repetition guard.** `repetition_penalty=1.5, no_repeat_ngram_size=3`
  prevents degenerate captions ("con con con...") that occur on ~5% of images
  otherwise.
- **Timeline fallback.** When photos have no EXIF (e.g. downloaded datasets),
  events are chained by embedding similarity instead, so the timeline still
  reads as a plausible sequence.

## Project structure

```
config.py              # every path, model name and threshold in one place
data/raw_images/       # your photo collection
embeddings/            # step 1 - CLIP
scene_recognition/     # step 2 - Places365
captioning/            # step 3 - BLIP
similarity_search/     # step 4 - FAISS
event_clustering/      # step 5
timeline/              # step 6
story_generation/      # step 7 - Groq / Llama
ui/                    # step 8 - Gradio app
evaluation/            # metrics, ablation study, RESULTS.md
utils/                 # shared helpers + result visualizers
outputs/               # generated visualizations (annotated photos, cluster grids)
```

## Tech stack

Python · PyTorch · Hugging Face Transformers · FAISS · scikit-learn ·
HDBSCAN · Gradio · Pillow · NumPy · Groq API (Llama-3.3-70B)


