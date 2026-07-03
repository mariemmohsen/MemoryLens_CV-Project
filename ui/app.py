"""
MemoryLens UI - a clean Gradio interface that turns the pipeline's outputs
into a personal "memory book": an AI-reconstructed timeline of life events,
each with photos, a title and a short story, plus semantic photo search.

Run:
    python -m ui.app
then open http://127.0.0.1:7860
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    TIMELINE_OUTPUT,
    CLUSTERS_OUTPUT,
    STORY_GENERATION_OUTPUT,
    SCENE_PREDICTIONS_OUTPUT,
    CAPTIONS_OUTPUT,
    FAISS_INDEX_OUTPUT,
    FAISS_INDEX_FILENAMES_OUTPUT,
    SIMILARITY_SEARCH_TOP_K,
    CLIP_MODEL_NAME,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from similarity_search.faiss_index import SimilaritySearchIndex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_json(path: Path, default):
    if not path.exists():
        logger.warning("Missing %s - run its pipeline step first.", path)
        return default
    with open(path) as f:
        return json.load(f)


def load_events():
    """Timeline events joined with their generated stories."""
    timeline = _load_json(TIMELINE_OUTPUT, None)
    if timeline is None:
        clusters = _load_json(CLUSTERS_OUTPUT, [])
        timeline = [c for c in clusters if c["cluster_id"] != -1]

    stories = {s["cluster_id"]: s for s in _load_json(STORY_GENERATION_OUTPUT, [])}
    events = []
    for event in timeline:
        story = stories.get(event["cluster_id"], {})
        events.append({
            **event,
            "story_title": story.get("title", event["title"]),
            "story_text": story.get("story", ""),
        })
    return events


def load_all_data():
    """(Re)load every pipeline output the UI displays."""
    global EVENTS, CAPTIONS, SCENES, SCENE_CONFIDENCE, ALL_IMAGES, _search_index
    EVENTS = load_events()
    CAPTIONS = {c["image"]: c["caption"] for c in _load_json(CAPTIONS_OUTPUT, [])}
    scene_predictions = _load_json(SCENE_PREDICTIONS_OUTPUT, [])
    SCENES = {p["image"]: p["scene"] for p in scene_predictions}
    SCENE_CONFIDENCE = {p["image"]: p["confidence"] for p in scene_predictions}
    ALL_IMAGES = sorted(p.name for p in RAW_IMAGES_DIR.iterdir() if p.is_file())
    _search_index = None  # force reload of the FAISS index after reprocessing


EVENTS: list = []
CAPTIONS: dict = {}
SCENES: dict = {}
SCENE_CONFIDENCE: dict = {}
ALL_IMAGES: list = []

_search_index = None
_clip_embedder = None

load_all_data()


def get_search_index():
    global _search_index
    if _search_index is None and FAISS_INDEX_OUTPUT.exists():
        _search_index = SimilaritySearchIndex.load(
            FAISS_INDEX_OUTPUT, FAISS_INDEX_FILENAMES_OUTPUT
        )
    return _search_index


def get_clip_embedder():
    """Lazily load CLIP only if the user searches with an uploaded photo."""
    global _clip_embedder
    if _clip_embedder is None:
        from embeddings.clip_embedder import ClipEmbedder
        _clip_embedder = ClipEmbedder(model_name=CLIP_MODEL_NAME)
    return _clip_embedder


# ---------------------------------------------------------------------------
# Memory Book tab
# ---------------------------------------------------------------------------
def event_choices():
    return [
        f"{i + 1}. {e['story_title']}  ·  {e['size']} photos"
        for i, e in enumerate(EVENTS)
    ]


def show_event(choice: str):
    index = int(choice.split(".")[0]) - 1
    event = EVENTS[index]

    confidences = [SCENE_CONFIDENCE[n] for n in event["images"] if n in SCENE_CONFIDENCE]
    confidence_note = (
        f" &nbsp;·&nbsp; scene confidence {sum(confidences) / len(confidences):.0%}"
        if confidences else ""
    )
    story_card = f"""
    <div class="story-card">
      <div class="story-kicker">Chapter {index + 1} of {len(EVENTS)}</div>
      <h2 class="story-title">{event['story_title']}</h2>
      <p class="story-text">{event['story_text'] or 'Run story generation to write this chapter.'}</p>
      <div class="story-meta">{event['size']} photos &nbsp;·&nbsp; detected as “{event['title']}”{confidence_note}</div>
    </div>
    """
    gallery = [
        (str(RAW_IMAGES_DIR / name), CAPTIONS.get(name, ""))
        for name in event["images"]
    ]
    return story_card, gallery


# ---------------------------------------------------------------------------
# Find Similar tab
# ---------------------------------------------------------------------------
def search_by_name(filename: str):
    index = get_search_index()
    if index is None:
        raise gr.Error("No FAISS index found - run `python -m similarity_search.build_index` first.")
    results = index.search_by_filename(filename, k=SIMILARITY_SEARCH_TOP_K)
    return str(RAW_IMAGES_DIR / filename), [
        (str(RAW_IMAGES_DIR / name), f"{score:.0%} similar - {CAPTIONS.get(name, '')}")
        for name, score in results
    ]


def search_by_upload(image_path: str):
    if image_path is None:
        raise gr.Error("Upload a photo first.")
    index = get_search_index()
    if index is None:
        raise gr.Error("No FAISS index found - run `python -m similarity_search.build_index` first.")
    embeddings, _ = get_clip_embedder().embed_batch([Path(image_path)])
    if embeddings.shape[0] == 0:
        raise gr.Error("Could not read that image.")
    results = index.search(embeddings[0], k=SIMILARITY_SEARCH_TOP_K)
    return [
        (str(RAW_IMAGES_DIR / name), f"{score:.0%} similar - {CAPTIONS.get(name, '')}")
        for name, score in results
    ]


def search_by_text(query: str):
    """CLIP text-to-image search: describe a moment in words, get the photos."""
    if not query or not query.strip():
        raise gr.Error("Type a description first, e.g. \"sunset on the beach\".")
    index = get_search_index()
    if index is None:
        raise gr.Error("No FAISS index found - run `python -m similarity_search.build_index` first.")
    text_embedding = get_clip_embedder().embed_texts([query.strip()])[0]
    results = index.search(text_embedding, k=SIMILARITY_SEARCH_TOP_K * 2)
    return [
        (str(RAW_IMAGES_DIR / name), f"match {score:.2f} - {CAPTIONS.get(name, '')}")
        for name, score in results
    ]


# ---------------------------------------------------------------------------
# Home tab: add photos and (re)run the pipeline with a progress bar
# ---------------------------------------------------------------------------
def process_collection(files, progress=gr.Progress()):
    """Copy uploaded photos into the collection and run the full pipeline.

    Each stage reports to the Gradio progress bar. Story generation is
    skipped gracefully when GROQ_API_KEY isn't set, so the rest of the
    pipeline still works offline.
    """
    added = 0
    if files:
        for f in files:
            src = Path(f)
            if src.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                dest = RAW_IMAGES_DIR / src.name
                if not dest.exists():
                    shutil.copy(src, dest)
                    added += 1
    status = [f"**{added}** new photos added to the collection." if added else "No new photos uploaded - reprocessing the existing collection."]

    from embeddings.generate_embeddings import run as run_embeddings
    from scene_recognition.run_scene_recognition import run as run_scenes
    from captioning.run_captioning import run as run_captions
    from similarity_search.build_index import run as run_faiss
    from event_clustering.run_clustering import run as run_clustering
    from timeline.run_timeline import run as run_timeline

    stages = [
        ("Extracting CLIP embeddings", run_embeddings, {}),
        ("Recognizing scenes (Places365)", run_scenes, {}),
        ("Writing captions (BLIP)", run_captions, {"skip_existing": True}),
        ("Building FAISS search index", run_faiss, {}),
        ("Clustering photos into events", run_clustering, {}),
        ("Reconstructing the timeline", run_timeline, {}),
    ]
    for i, (label, stage_fn, kwargs) in enumerate(stages):
        progress(i / (len(stages) + 1), desc=label)
        try:
            stage_fn(**kwargs)
            status.append(f"✅ {label}")
        except Exception as exc:
            logger.exception("Stage failed: %s", label)
            status.append(f"❌ {label}: {exc}")

    progress(len(stages) / (len(stages) + 1), desc="Writing event stories (Llama)")
    if os.environ.get("GROQ_API_KEY"):
        try:
            from story_generation.run_story_generation import run as run_stories
            run_stories()
            status.append("✅ Writing event stories (Llama)")
        except Exception as exc:
            logger.exception("Story generation failed")
            status.append(f"❌ Writing event stories: {exc}")
    else:
        status.append("⚠️ Story generation skipped - set the GROQ_API_KEY environment variable and reprocess to write stories.")

    progress(1.0, desc="Done")
    load_all_data()

    choices = event_choices()
    return (
        "\n\n".join(status),
        gr.update(choices=choices, value=choices[0] if choices else None),
        gr.update(choices=ALL_IMAGES),
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
CSS = """
.hero { text-align: center; padding: 28px 16px 8px; }
.hero h1 { font-size: 2.4em; margin: 0; letter-spacing: -1px; }
.hero .lens { background: linear-gradient(90deg, #6366f1, #ec4899); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: var(--body-text-color-subdued); font-size: 1.05em; margin-top: 6px; }
.impact { display: flex; justify-content: center; gap: 14px; flex-wrap: wrap; margin: 14px 0 4px; }
.impact .stat { background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 10px 22px; text-align: center; }
.impact .stat b { display: block; font-size: 1.5em; }
.impact .stat span { color: var(--body-text-color-subdued); font-size: 0.85em; }
.story-card { border: 1px solid var(--border-color-primary); border-left: 5px solid #6366f1; border-radius: 14px; padding: 18px 22px; background: var(--block-background-fill); }
.story-kicker { text-transform: uppercase; letter-spacing: 2px; font-size: 0.72em; color: #6366f1; font-weight: 700; }
.story-title { margin: 4px 0 8px; font-size: 1.6em; }
.story-text { font-size: 1.05em; line-height: 1.55; margin: 0 0 10px; }
.story-meta { color: var(--body-text-color-subdued); font-size: 0.85em; }
.footer-note { text-align: center; color: var(--body-text-color-subdued); font-size: 0.85em; padding: 18px 0 8px; }
"""

HERO = f"""
<div class="hero">
  <h1>Memory<span class="lens">Lens</span></h1>
  <p>Drop in a folder of unsorted photos — get back the story of your life.<br>
  Built for the moments people can't afford to forget: family archives, fading memories, and everyone who has 10,000 photos and no idea what's in them.</p>
</div>
<div class="impact">
  <div class="stat"><b>{len(ALL_IMAGES)}</b><span>photos understood</span></div>
  <div class="stat"><b>{len(EVENTS)}</b><span>life events discovered</span></div>
  <div class="stat"><b>{len(set(SCENES.values()) - {"Other"})}</b><span>scene types recognized</span></div>
  <div class="stat"><b>0</b><span>manual labels needed</span></div>
</div>
"""

FOOTER = """
<div class="footer-note">
  CLIP embeddings · Places365 scenes · BLIP captions · FAISS search · Agglomerative event clustering · Llama-3.3 stories — all pretrained, zero training.
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MemoryLens", css=CSS, theme=gr.themes.Soft()) as app:
        gr.HTML(HERO)

        with gr.Tabs():
            with gr.Tab("📖 Memory Book"):
                choices = event_choices()
                selector = gr.Radio(
                    choices=choices,
                    value=choices[0] if choices else None,
                    label="Your reconstructed timeline",
                )
                story_html = gr.HTML()
                event_gallery = gr.Gallery(
                    label="Photos from this memory", columns=6, height=340,
                    object_fit="cover",
                )
                selector.change(show_event, inputs=selector, outputs=[story_html, event_gallery])
                if choices:
                    app.load(show_event, inputs=selector, outputs=[story_html, event_gallery])

            with gr.Tab("🔍 Search"):
                gr.Markdown("Describe a moment in plain words, pick a photo from the collection, or upload a new one — MemoryLens finds the moments that *feel* the same, not just look the same.")
                with gr.Row():
                    text_query = gr.Textbox(
                        label="Describe a moment",
                        placeholder='e.g. "sunset on the beach", "birthday cake", "people at a wedding"',
                        scale=3,
                    )
                    text_btn = gr.Button("Search by text", variant="primary", scale=1)
                with gr.Row():
                    with gr.Column(scale=1):
                        picker = gr.Dropdown(choices=ALL_IMAGES, label="...or a photo from your collection")
                        search_btn = gr.Button("Find similar")
                        query_preview = gr.Image(label="Query photo", height=200)
                        upload = gr.Image(label="...or upload a new photo", type="filepath", height=160)
                        upload_btn = gr.Button("Search with upload")
                    with gr.Column(scale=2):
                        results_gallery = gr.Gallery(
                            label="Most similar moments", columns=3, height=520,
                            object_fit="cover",
                        )
                text_btn.click(search_by_text, inputs=text_query, outputs=results_gallery)
                text_query.submit(search_by_text, inputs=text_query, outputs=results_gallery)
                search_btn.click(search_by_name, inputs=picker, outputs=[query_preview, results_gallery])
                upload_btn.click(search_by_upload, inputs=upload, outputs=results_gallery)

            with gr.Tab("➕ Add Photos & Reprocess"):
                gr.Markdown(
                    "Upload new photos (or none, to just re-run the pipeline on the current "
                    "collection), then press **Process**. Every stage of the pipeline runs "
                    "right here with a live progress bar."
                )
                uploader = gr.File(
                    label="Drop photos here",
                    file_count="multiple",
                    file_types=["image"],
                )
                process_btn = gr.Button("Process collection", variant="primary")
                process_status = gr.Markdown()
                process_btn.click(
                    process_collection,
                    inputs=uploader,
                    outputs=[process_status, selector, picker],
                )

            with gr.Tab("✨ How It Works"):
                gr.Markdown(
                    """
### From chaos to story, in seven steps — no training, all pretrained models

| | Step | Model / Technique | What it contributes |
|--|------|-------------------|---------------------|
| 1 | **Semantic embeddings** | CLIP ViT-B/32 | Every photo becomes a 512-number "meaning fingerprint" |
| 2 | **Scene recognition** | ResNet-18 · Places365 | Knows a beach from an airport from a museum |
| 3 | **Image captioning** | BLIP | Writes a sentence about every single photo |
| 4 | **Similarity search** | FAISS | Finds photos that share meaning in milliseconds |
| 5 | **Event clustering** | Agglomerative + image+caption embeddings | Groups photos into real events — birthdays, weddings, graduations |
| 6 | **Timeline** | EXIF → file time → similarity chaining | Puts the events back in a life-like order |
| 7 | **Storytelling** | Llama-3.3-70B (Groq) | Turns each event into a warm, human paragraph |

### Why it matters
- **For families** — decades of shoeboxes and camera rolls become a readable memory book.
- **For memory care** — people living with dementia respond to *stories and photos together*; MemoryLens builds exactly that from an unlabeled archive, automatically.
- **For digital legacy** — a life's photos, organized and narrated, without anyone spending a weekend tagging them.
                    """
                )

        gr.HTML(FOOTER)
    return app


if __name__ == "__main__":
    build_app().launch(allowed_paths=[str(RAW_IMAGES_DIR)])
