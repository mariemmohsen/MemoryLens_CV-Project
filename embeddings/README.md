# Embeddings

Step 2 of the MemoryLens pipeline: generate semantic image embeddings with
a pretrained CLIP model (no training).

- Model: `openai/clip-vit-base-patch32` (see `CLIP_MODEL_NAME` in `config.py`)
- Input: images from `data/raw_images/`
- Output:
  - `embeddings/embeddings.npy` — `(N, D)` L2-normalized float32 array, one row per image
  - `embeddings/image_filenames.json` — filenames in the same row order as `embeddings.npy`

Run with:

```
python -m embeddings.generate_embeddings
```
