# Evaluation & Ablation Study

Quantitative evaluation of the MemoryLens event-clustering and scene-recognition
stages. All numbers below are reproducible:

```bash
python -m evaluation.run_evaluation   # metrics for the current production run
python -m evaluation.ablation         # full ablation study + plots
```

Results are written to `evaluation/results/` (JSON, CSV, and PNG plots).

---

## 1. Dataset and pseudo-ground-truth

The demo dataset has **290 images**. Each image's true category is encoded in
its filename (see `evaluation/ground_truth.py`):

| Group | Classes | Images |
|-------|---------|--------|
| Scene stock photos | airport, restaurant, museum, beach | 200 (50 each) |
| Curated events | birthday_party, wedding_celebration, graduation_party, family_dinner, family_christmas, family_gathering, family_picnic, family_beach_day, family_vacation, kids_playing | 49 (~5 each) |
| Filler (`sample_*`, `hero`) | — (unlabeled) | 41 |

This gives **14 labeled classes over 249 images**. These are *weak* labels
derived from how the dataset was assembled, not hand-verified annotations, so
they are best described as **pseudo-ground-truth**. They remain a valid basis
for extrinsic metrics because each labeled group is a genuinely distinct scene
or event. The 41 filler images are excluded from extrinsic metrics.

## 2. Metrics

**Intrinsic** (geometry only, no labels): Silhouette (cosine, higher better),
Davies–Bouldin (lower better), Calinski–Harabasz (higher better).

**Extrinsic** (agreement with ground truth): Adjusted Rand Index (ARI),
Normalized Mutual Information (NMI), Homogeneity, Completeness, V-measure, and
Purity. Noise points (cluster −1) are treated as one extra predicted cluster;
`coverage` separately reports the fraction of images actually clustered.

## 3. Production configuration results

Fused (image + caption) embeddings, Agglomerative clustering, cosine distance
threshold 0.30, `min_cluster_size=3`:

| Metric | Value |
|--------|-------|
| Clusters found | 14 |
| Coverage (non-noise) | 84.1% |
| **Silhouette (cosine)** | **0.297** |
| Davies–Bouldin | 1.73 |
| Calinski–Harabasz | 14.7 |
| **Adjusted Rand Index (ARI)** | **0.759** |
| **NMI** | **0.801** |
| Homogeneity | 0.828 |
| Completeness | 0.776 |
| V-measure | 0.801 |
| **Purity** | **0.847** |

An ARI of 0.76 and purity of 0.85 indicate the clustering recovers the true
categories well above chance (ARI is chance-corrected: 0 = random).

## 4. Ablation: does adding caption text help?

Threshold swept over 0.15–0.55 for three embedding modalities. Best ARI per
modality:

**Full dataset** (dominated by the 200 near-identical scene photos):

| Modality | Best ARI | Best NMI | @ threshold |
|----------|----------|----------|-------------|
| Image only | **0.936** | **0.905** | 0.350 |
| Caption only | 0.477 | 0.622 | 0.350 |
| Fused (image + caption) | 0.886 | 0.879 | 0.350 |

**Event subset only** (the 49 hard event photos, which fusion targets):

| Modality | Best ARI | Best NMI | @ threshold |
|----------|----------|----------|-------------|
| Image only | 0.455 | 0.740 | 0.325 |
| Caption only | 0.358 | 0.640 | 0.325 |
| Fused (image + caption) | **0.458** | **0.749** | 0.275 |

**Honest finding.** On the *full* dataset, image-only embeddings actually score
highest: the 200 visually near-identical stock photos form four very clean
pools that captions slightly blur. On the *event subset* — the hard,
visually-diverse minority the fusion was designed for — fusion is best on both
ARI and NMI, though the margin over image-only is small. Caption-only is
consistently weakest, confirming captions are a useful *complementary* signal,
not a replacement for visual features.

**Practical implication.** For a dataset dominated by repeated scene shots,
image-only clustering is sufficient. For a real personal photo collection
(mostly diverse events, few near-duplicates) fusion is the safer choice. This
is exactly the regime MemoryLens targets, which justifies keeping fusion as the
default while documenting image-only as a strong, simpler baseline.

See `results/ablation_ari.png`, `ablation_ari_events.png`, `ablation_nmi.png`,
`ablation_silhouette.png`, `ablation_n_clusters.png`.

## 5. Algorithm comparison: Agglomerative vs HDBSCAN

Both run on fused embeddings at threshold 0.35:

| Algorithm | ARI | NMI | Clusters | Coverage |
|-----------|-----|-----|----------|----------|
| **Agglomerative** | **0.886** | **0.879** | 13 | 93% |
| HDBSCAN | 0.402 | 0.616 | 10 | 59% |

HDBSCAN's density criterion marks 41% of images as noise and scores less than
half the ARI. This quantitatively justifies choosing Agglomerative clustering
over the density-based HDBSCAN named in the original specification. (HDBSCAN
remains selectable via `EVENT_CLUSTERING_ALGORITHM = "hdbscan"`.)

## 6. Threshold sensitivity and adaptive selection

Results are sensitive to the distance threshold: ARI peaks sharply around
0.30–0.35 and collapses beyond 0.45 (everything merges into one cluster). To
avoid a hardcoded value that may not transfer to other datasets,
`select_threshold_by_silhouette()` auto-selects the threshold maximizing mean
silhouette. On this dataset it picks **0.40** (favoring cleaner geometry),
versus **0.35** as the extrinsic (ARI) optimum — a reasonable, label-free
proxy. Enable it with `EVENT_CLUSTERING_AUTO_THRESHOLD = True`.

## 7. Scene recognition accuracy

Places365 top-1 accuracy on the 200 labeled scene photos: **55.0%**.

The bottleneck is not the CNN but the hand-written mapping from Places365's 365
fine categories to MemoryLens's simple labels (`SCENE_CATEGORY_MAP` in
`scene_recognition/scene_recognizer.py`): many correct fine predictions (e.g.
"boat_deck", "coast") fall outside the mapping and collapse to "Other".
Expanding the mapping is the clearest path to higher scene accuracy.

## 8. Limitations

- **Pseudo-ground-truth**, not human annotation — labels reflect dataset
  assembly, not verified truth.
- **No timestamp/GPS/face signals** — clustering is visual + textual only.
  EXIF-based separation is implemented for the timeline stage but the demo
  photos carry no EXIF.
- **Small event classes** (~5 images) make event-subset metrics
  high-variance; a larger annotated set would tighten the estimates.
- **Scene mapping coverage** caps scene accuracy independent of the CNN.
