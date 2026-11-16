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

The task is to detect objects of 8 classes in drone video clips of different resolutions, where most objects are small. The small size makes the task difficult: there are fewer pixels per object, a bounding box error of a few pixels is more harmful, and the detections are harder to classify.

The dataset consists of video clips with labeled bounding boxes, their classes, and ignored areas, where detections are not scored. Those ignored areas typically contain many objects too small to classify, so in preprocessing we fill them with grey rectangles. During training, we filled ignored regions with grey pixels and removed bounding boxes that overlapped these regions by more than 50%. During validation, we instead represented ignored regions as COCO crowd annotations for each class, so matching detections were not penalized as false positives.

The dataset is split into train, validation and test clips by the organizers. As we saw in preprocessing, the splits differ in resolution: test is 680–1920 px wide and train 1344–2720 px, so test has smaller resolutions than train. We also found that test frames contain a much higher proportion of small predicted boxes (57% under 32 px compared to 18% on validation), as well as more camera motion, blur and compression. These differences motivated our experiments with input resolution, smaller anchors and test-like augmentations.

Each clip has little movement: the camera does not move rapidly, and most of the movement comes from moving objects. We addressed that by taking not every frame from the train dataset, but only every 8th one, because we wanted consecutive sampled frames to show noticeable movement of the bounding boxes. That shrinks the dataset, makes training faster (reduced the training dataset from 24,201 to 3,050 frames) and reduces overfitting. We compared the same objects across frames using IoU and chose step 8 as the smallest tested interval satisfying our overlap criterion (IoU ≤ 0.5).

| Property | Train / val | Test |
|---|---|---|
| Clips / frames used | 56 train clips, 3,050 frames (step 8); 7 val clips, 359 labelled frames | 48 clips, 3,811 frames |
| Frame width | 1344–2720 px | 680–1920 px |
| Confident boxes under 32 px (Faster R-CNN) | ~18% (val) | ~57% |
| Camera motion, median shift per frame | ~1 px (val) | 2.4 px |

## 2. Validation design and its limits


| Metric | Definition |
|---|---|
| IoU | Overlap of a predicted and a true box: intersection area / union area |
| Precision | Share of predicted boxes that match a true box |
| Recall | Share of true boxes that are matched by a prediction |
| AP | Area under the precision–recall curve for one class at one IoU threshold, with predictions sorted by score |
| mAP@[.5:.95] (Kaggle metric) | AP averaged over IoU thresholds 0.50, 0.55, …, 0.95 and over the 8 classes |
| AP50 / AP75 | AP at a single IoU threshold of 0.5 / 0.75, averaged over classes |
| AP small / medium / large | mAP@[.5:.95] over objects with area < 32², 32²–96², > 96² px in the original frame |
| AR | Recall averaged over the IoU thresholds, with up to 200 boxes per image |
| Per-class AP | mAP@[.5:.95] for one class |




## 3. Models and tuning strategy

### 3.1 Two-stage and anchor-based detectors


Anchor-based detectors have a fixed size anchors for bounding boxes and the model moves and stretches those anchors to fit the detection.

We chose Faster R-CNN and RetinaNet as examples of two-stage and one-stage anchor-based detectors. RetinaNet does fitting the bounding boxes and classification in one step, making it faster, but less acurate. Faster R-CNN works in 2 stages. First Region Proposal Network suggests the bounding box by classifying every anchor as "object" or "background", then that anchor is classified as either of the label classes.

We tuned hyperparameters of the models one at a time to improve the inference on small objects.

**Faster R-CNN** (ResNet-50 FPN, COCO weights). Code: `src/02_faster_rcnn.ipynb`; anchor sizes from `src/05_anchor_kmeans.ipynb`; results in `runs/faster_rcnn/<run>/summary.json`.

| Run | Change vs baseline | Best epoch | mAP | AP50 | AP75 | AP small |
|---|---|---|---|---|---|---|
| baseline | anchors 32–512, one size per FPN level | 8 | 0.2145 | 0.440 | 0.184 | 0.098 |
| small_anchors | anchors 16–256 | 5 | 0.2138 | 0.452 | 0.179 | 0.097 |
| giou_loss | GIoU box loss instead of smooth L1 | 5 | 0.2169 | 0.442 | 0.187 | 0.086 |
| more_rois | 1024 RoIs per image, 50% positives, 3000/2000 test proposals | 7 | 0.2171 | 0.452 | 0.179 | 0.102 |
| small_anchors_giou | small_anchors + GIoU | 8 | 0.2152 | 0.449 | 0.184 | 0.099 |
| tuned | small_anchors + more_rois + GIoU | 3 | 0.2162 | 0.455 | 0.176 | 0.108 |
| anchors_3scale | 3 anchor sizes per level from 16 px (k-means), RPN output layers re-initialised | 5 | **0.2234** | 0.461 | 0.192 | 0.099 |

**RetinaNet** (ResNet-50 FPN, COCO weights, focal loss). Code: `src/04_retinanet.ipynb`; results in `runs/retinanet/<run>/summary.json`.

| Run | Change vs baseline | Best epoch | mAP | AP50 | AP75 | AP small |
|---|---|---|---|---|---|---|
| baseline | anchors 32–512, focal γ = 2, IoU fg/bg 0.5/0.4 | 7 | 0.1837 | 0.344 | 0.173 | 0.060 |
| focal_gamma | γ = 1.5 | 8 | **0.1865** | 0.350 | 0.177 | 0.055 |
| data_anchors | anchors 8–128 (k-means) | 6 | 0.0953 | 0.252 | 0.054 | 0.042 |
| data_anchors_s2 | anchors 16–256 | 7 | 0.1700 | 0.373 | 0.133 | 0.060 |
| iou_thresholds | IoU fg/bg 0.4/0.3 | 8 | 0.1426 | 0.270 | 0.135 | 0.050 |

### 3.2 Anchor-free and query-based detectors

We chose YOLO26 as an anchor-free detector and RT-DETR as a query-based detector to compare them with the previous anchor-based models. YOLO predicts bounding boxes and classes directly without using predefined anchors. RT-DETR uses a transformer with object queries to detect objects. We experimented with YOLO26s, YOLO26m, RT-DETR-L and RT-DETR-X.

We first ran short experiments with 4 epochs to test different input resolutions (640, 960 and 1280), scale augmentation, an additional P2 detection head for small objects, and the number of RT-DETR queries. We tested 500 queries because drone frames can contain many objects. We also experimented with mixup augmentation and using more training frames. After that, we trained selected models for more epochs to see whether their performance would improve. Longer training helped RT-DETR, while YOLO often reached its best validation performance earlier and started to overfit.

## 4. Single-model results

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

RT-DETR-L achieved the highest overall mAP (0.2728) and performed best on large objects (AP large = 0.466). However, YOLO26m performed better on small objects, reaching AP small = 0.161 compared to 0.120 for RT-DETR-L. This showed that different models have different strengths, which motivated us to combine their predictions in an ensemble. We also observed that the RT-DETR configuration trained at 1280 px achieved higher mAP than the 960 px version (0.2728 vs 0.2467). However, other training settings also differed, so we cannot attribute this improvement to resolution alone.

## 5. Inference and ensembling

### 5.1 Test-time augmentation and model fusion


We experimented with different inference methods to improve detection without retraining the models. Horizontal flip augmentation improved the results, especially for RT-DETR, while dividing images into smaller tiles reduced mAP. We also combined predictions from multiple models using Weighted Box Fusion (WBF), which merges overlapping bounding boxes instead of simply removing duplicates as NMS does. We tested different combinations of models and IoU thresholds, with 0.65 giving the best results. Combining 7 models increased mAP to 0.2970, while using all 18 models improved it further to 0.3091.

| Fusion (WBF IoU 0.65, val) | mAP | Notebook |
|---|---|---|
| Best single model + flip | 0.2534 | `09` |
| B's 7 models | 0.2970 | `docs/09_run1_7models.ipynb` |
| All RT-DETR (B) | 0.3019 | `09` |
| All 18 models (B) | 0.3091 | `09` |

### 5.2 Additions to the fusion


We also experimented with adding Faster R-CNN and RetinaNet to the ensemble. RetinaNet slightly improved mAP, while Faster R-CNN reduced it when its bounding boxes were included in the fusion. We then tested size-aware weights, giving models different importance depending on their performance on small, medium and large objects. This improved mAP from 0.2998 to 0.3027. Adding two more models increased it to 0.3128. We also used Faster R-CNN only to adjust detection confidence scores without changing box coordinates, reaching 0.3138. Finally, we tried a learned rescorer, but it performed worse on validation.

| Fusion (val, IoU 0.65) | mAP | Notebook |
|---|---|---|
| B's 7 + RetinaNet, equal weights | 0.2998 | `experiments/ensemble_temporal` |
| same, size-aware weights p = 2 | 0.3027 | `experiments/ensemble_weights` |
| + RT-DETR 1280 + YOLO26m, size-aware | 0.3128 | `experiments/ensemble_weights` |
| + Faster R-CNN score-only vote | 0.3138 | `experiments/ensemble_weights` |
| Learned rescorer, leave-one-clip-out (vs 0.3027) | 0.2790 | `experiments/learned_rescorer` |

## 6. Temporal rescoring
We developed two versions of temporal post-processing to improve detections using information from neighbouring video frames. The first version (v1) linked bounding boxes of the same class across consecutive frames using IoU ≥ 0.5 and adjusted their confidence scores based on the strongest detections in each track, without adding new boxes. This improved the 10-model ensemble mAP from 0.3138 to 0.3191.

The second version (v2) extended this approach with camera motion compensation, centre-distance linking, gap filling and Gaussian box smoothing. We tested these modifications separately, and smoothing gave the best result, increasing mAP to 0.3210. However, temporal processing can also reinforce false detections if the same mistake appears across several frames.

We also tested a separate, simpler approach in `temporal_rescore.py`, which uses RT-DETR predictions from the previous and next frames to adjust confidence scores with a weight of 0.1. This improved the 7-model ensemble from 0.2970 to 0.3002.

| Input (val) | Without | With | Notebook |
|---|---|---|---|
| Faster R-CNN multi-scale, v1 | 0.2182 | 0.2263 | `experiments/temporal_rescoring` |
| B7 + RetinaNet, v1 | 0.2998 | 0.3051 | `experiments/ensemble_temporal` |
| 10-member fusion, v1 / v2 (smoothing) | 0.3138 | 0.3191 / 0.3210 | `experiments/temporal_v2` |
| B's 7 models, `temporal_rescore.py` | 0.2970 | 0.3002 | `09`, `temporal_rescore.py` |


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
| 10 members, size-aware, FRCNN vote | 0.3138 | 0.3303 |
| same + temporal v2 | 0.3210 | 0.3335 |
| B's 18 models | 0.3091 | 0.3175 |
| B's 18 models + temporal_rescore | 0.3091 | 0.3345 |


## 9. Final model and how to run it

Our final solution uses an ensemble of 18 detectors (11 YOLO26 and 7 RT-DETR models). The combination was selected using 09_final_pipeline.ipynb, achieving 0.3091 mAP on validation before temporal rescoring.

Each model predicts on the original and horizontally flipped images, except one model that uses only the original. Predictions are combined using Weighted Box Fusion (WBF) with IoU = 0.65 and equal model weights. We then apply temporal rescoring using RT-DETR predictions from neighbouring frames, adjusting confidence scores without changing bounding boxes.

The models were trained using 07_yolo.ipynb and 08_rtdetr.ipynb with different resolutions, augmentations and training durations. The best checkpoints were selected based on validation mAP.

How to run:

- install dependencies from requirements.txt

- run src/01_preprocess.ipynb to prepare the dataset

- run src/make_train_step.py with arguments 4 and 2 to generate additional training datasets

- run src/07_yolo.ipynb and src/08_rtdetr.ipynb to train the models and save their best.pt weights

- run src/09_final_pipeline.ipynb to evaluate model combinations and generate final_config.json

- run src/ensemble_submission.py with the selected configuration to generate the ensemble predictions

- run src/temporal_rescore.py with the ensemble submission and RT-DETR support predictions to generate the final CSV


## 10. Conclusions

<!--
[A + B together] ~60 words. Three or four sentences: what worked best, what you would do next with more time.
-->