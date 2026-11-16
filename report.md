---
title: Homework 2 -- Multi-object video detection
author: Mykyta Yahoda & Olena Yakovenko
course: Artificial Intelligence · UCU
date: 2026-11-16
---

<!--
HOW TO USE THIS TEMPLATE
- Everything inside <!-- --> comments is guidance and never renders. Delete the comments when a section is done.
- [A] = Person A writes, [B] = Person B writes. Word budgets add up to ~1,500 (tables and captions not counted).
- Tables are pre-filled with numbers from the fact sheet. Check every number against its notebook before submitting.
- Write the prose yourselves. Answer the prompts in your own words; do not paste text from any AI tool.
- Each result you mention should name the notebook that produced it, e.g. (`02_faster_rcnn.ipynb`).
- "TODO" marks facts that are still missing (leaderboard scores, final pipeline choice).
-->

## 1. Task and data

<!--
[A] ~150 words. Answer:
- What is the task (8 classes, drone video, Kaggle metric) and what makes it hard? Pick the 2–3 reasons that drove your decisions.
- How did you split and sample the data, and why (clip-level split, frame step 8 and the IoU <= 0.5 rule)?
- How are ignored regions handled in training vs in scoring?
- How does test differ from val? Refer to the table below; say what you concluded from it.
-->

| Property | Train / val | Test |
|---|---|---|
| Clips / frames used | 56 train clips, 3,050 frames (step 8); 7 val clips, 359 labelled frames | 48 clips, 3,811 frames |
| Frame width | 1344–2720 px | 680–1920 px |
| Confident boxes under 32 px (Faster R-CNN) | ~18% (val) | ~57% |
| Camera motion, median shift per frame | ~1 px (val) | 2.4 px |

## 2. Validation design and its limits

<!--
[A] ~120 words. Answer:
- What is the single evaluator and why does every number go through it (`common/evaluate.py`, 500 boxes, crowd boxes per class)?
- What validation did you actually use (one fixed 7-clip split; test-like val for Faster R-CNN only; leave-one-clip-out only for the learned rescorer)?
- Where did val and the leaderboard disagree, and what would you do differently? (Be honest: no CV, seed noise ~0.03 for 4-epoch YOLO.)
-->

## 3. Models and tuning strategy

### 3.1 Two-stage and anchor-based detectors

<!--
[A] ~120 words. Answer:
- Why Faster R-CNN and RetinaNet as the anchor-based families? Which weights (COCO) and protocol (1280 px, 8 epochs, seed 42)?
- What was your tuning strategy (one change per run against a baseline, then a combined "tuned" run)?
- Which changes targeted small objects and why (anchor sizes from k-means in 05, RPN re-initialisation, RoI budget, GIoU, focal gamma)?
-->

### 3.2 Anchor-free and query-based detectors

<!--
[B] ~130 words. Answer:
- Why YOLO26 and RT-DETR? Which sizes (s, m, L, X)?
- What did the short 4-epoch runs test (input size, scale jitter, P2 head, mixup, denser frames, queries)?
- Why 500 RT-DETR queries; why long schedules for RT-DETR but short for YOLO?
-->

## 4. Single-model results

<!--
[A] fills the 02/04 rows and checks them; [B] fills the 07/08 rows. [B] writes ~70 words, [A] ~50 words under the table:
- Which family is best overall, which on small objects, which on large ones? What does that suggest for an ensemble?
- What did input resolution do?
Native val, best epoch, no TTA unless marked.
-->

| Family | Run (notebook) | mAP | AP50 | AP small | AP large |
|---|---|---|---|---|---|
| Faster R-CNN | baseline (`02`) | 0.2145 | 0.440 | 0.098 | 0.405 |
| Faster R-CNN | anchors_3scale (`02`) | 0.2234 | 0.461 | 0.099 | 0.410 |
| RetinaNet | baseline (`04`) | 0.1837 | 0.344 | 0.060 | 0.360 |
| RetinaNet | focal_gamma 1.5 (`04`) | 0.1865 | 0.350 | 0.055 | 0.360 |
| YOLO26s | baseline 1280, 4 ep (`07`) | 0.2173 | 0.422 | 0.142 | 0.345 |
| YOLO26s | tuned 1280 + flip (`09`) | 0.2426 | 0.466 | 0.146 | 0.399 |
| YOLO26m | m_1280 + flip (`09`) | 0.2165 | 0.437 | 0.161 | 0.366 |
| RT-DETR-L | final_960_s025, 40 ep (`08`) | 0.2467 | 0.475 | 0.105 | 0.407 |
| RT-DETR-L | final_1280 (`09`) | 0.2728 | 0.506 | 0.120 | 0.466 |

## 5. Inference and ensembling

### 5.1 Test-time augmentation and model fusion

<!--
[B] ~150 words. Answer:
- What did flip and tiles do per model, and why do you think tiles hurt?
- How does WBF work in one or two sentences, and why WBF instead of NMS?
- How did you choose the members and the IoU (family vs all models; 0.65)?
-->

| Fusion (WBF IoU 0.65, val) | mAP | Notebook |
|---|---|---|
| Best single model + flip | 0.2534 | `09` |
| B's 7 models | 0.2970 | `docs/09_run1_7models.ipynb` |
| All RT-DETR (B) | 0.3019 | `09` |
| All 18 models (B) | 0.3091 | `09` |

### 5.2 Additions to the fusion

<!--
[A] ~100 words. Answer:
- Why did RetinaNet help the fusion but Faster R-CNN hurt it (look at AP75)?
- What are size-aware weights and why one parameter p?
- What did the Faster R-CNN "score-only vote" test, and what did the learned rescorer show under leave-one-clip-out?
-->

| Fusion (val, IoU 0.65) | mAP | Notebook |
|---|---|---|
| B's 7 + RetinaNet, equal weights | 0.2998 | `experiments/ensemble_temporal` |
| same, size-aware weights p = 2 | 0.3027 | `experiments/ensemble_weights` |
| + RT-DETR 1280 + YOLO26m, size-aware | 0.3128 | `experiments/ensemble_weights` |
| + Faster R-CNN score-only vote | 0.3138 | `experiments/ensemble_weights` |
| Learned rescorer, leave-one-clip-out (vs 0.3027) | 0.2790 | `experiments/learned_rescorer` |

## 6. Temporal rescoring

<!--
[A] ~80 words + [B] 2 sentences. Answer:
- [A] How does v1 link and rescore, and why can it raise mAP without adding boxes?
- [A] What did each v2 switch do (motion compensation, centre distance, gap filling, smoothing)? When does temporal make things worse?
- [B] How does temporal_rescore.py differ (support model, ±1 frame, a = 0.1)?
-->

| Input (val) | Without | With | Notebook |
|---|---|---|---|
| Faster R-CNN multi-scale, v1 | 0.2182 | 0.2263 | `experiments/temporal_rescoring` |
| B7 + RetinaNet, v1 | 0.2998 | 0.3051 | `experiments/ensemble_temporal` |
| 10-member fusion, v1 / v2 (smoothing) | 0.3138 | 0.3191 / 0.3210 | `experiments/temporal_v2` |
| B's 7 models, `temporal_rescore.py` | 0.2970 | 0.3002 | `09`, `temporal_rescore.py` |

![One val frame without (left) and with (right) temporal rescoring](images/temporal_example.jpg)

## 7. Unsuccessful attempts

<!--
[A] ~75 words, [B] ~75 words. One or two sentences each: what you tried, the number, what you learned.
[A] candidates: anchors_3scale (best on val, worse on LB), 8 px RetinaNet anchors, learned rescorer, YOLO26m on test-like frames.
[B] candidates: tiles / SAHI, long YOLO schedules, denser training frames (step 4 / 2), P2 head, seed noise.
-->

## 8. Validation vs leaderboard

<!--
[A] ~80 words. Answer: where did val and the leaderboard agree, where not, and what do you think caused it?
Fill the TODO scores from runs/kaggle_submissions.csv.
-->

| Submission | Val mAP | Public LB |
|---|---|---|
| Faster R-CNN baseline, multi-scale | 0.2182 (test-like) | 0.2489 |
| Faster R-CNN anchors_3scale, multi-scale | 0.2248 (test-like) | 0.2144 |
| RT-DETR-L 960 + flip | 0.254 | 0.3013 |
| B's 7 models, WBF | 0.297 | 0.3113 |
| B7 + RetinaNet | 0.2998 | 0.3145 |
| B7 + RetinaNet + temporal v1 | 0.3051 | 0.3202 |
| B7 + RetinaNet, size-aware weights | 0.3027 | 0.3268 |
| B's 7, RT-DETR weight 2 | 0.2967 | 0.3278 |
| 10 members, size-aware, FRCNN vote | 0.3138 | TODO |
| same + temporal v2 | 0.3210 | TODO |
| B's 18 models | 0.3091 | TODO |
| B's 18 models + temporal_rescore | TODO | TODO |

![Validation vs public leaderboard per submission](images/val_vs_lb.png)

## 9. Final model and how to run it

<!--
Written by the owner of the chosen pipeline (TODO: decide by leaderboard); the other person follows the steps to check them. ~280 words.
Include:
- Which submission file is final, its members, fusion settings and temporal step.
- Dependencies: Python version, requirements.txt / requirements-amd.txt (and the exact install command used for ROCm).
- Data: where the competition data and data_processed/ must be.
- Exact order of notebooks / scripts with commands, and roughly how long each takes on which GPU.
- Seeds (42; note that RT-DETR runs are not bit-exact).
- Where the final CSV appears.
-->

## 10. Conclusions

<!--
[A + B together] ~60 words. Three or four sentences: what worked best, what you would do next with more time.
-->