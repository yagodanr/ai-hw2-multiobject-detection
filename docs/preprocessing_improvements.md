# Preprocessing improvements: backlog

Ideas for preprocessing beyond the current `src/01_preprocess.ipynb`, ranked by expected gain vs. effort.
Each one should be run as an ablation against the current pipeline, with the same val split, metric and budget.

## Current pipeline (baseline)

- Split by dataset folders (56 train clips, 7 val clips)
- Frame step 5 (smallest step where same-object IoU between sampled frames ≤ 0.5)
- Boxes clipped to the image, degenerate boxes (< 2 px) dropped
- Ignored regions grey-filled for training; boxes > 50% inside dropped
- Evaluation GT: original val images, ignored regions as `iscrowd=1` copied for every class
- Augmentation: flip, mild scale/translate, brightness/contrast, HSV

## Priority 1: do these

### 1. Tiled training at full resolution
- **What:** cut each training frame into 512–640 px tiles with about 25% overlap, discard tiles that contain no objects, and also keep the downscaled full frame.
- **Why:** this was the most common trick among top VisDrone entries. The 2nd-place VID2019 team cut frames into 4 patches, and others trained on 896×896 patches or ran detection on 4 unscaled pieces. CroW (Varga & Zell, 2021) reports +2 to +13 mAP on VisDrone (YOLOv4: 17.7 → 28.7), mainly because removing empty tiles reduces background bias.
- **How:** an extra export step after subsampling. Recompute boxes per tile, keep boxes that are at least 30–50% visible, and keep the grey fill. Use a larger frame step to keep the image count manageable, because tiling multiplies it by about 4–6×.
- **Inference:** either full frame (as in CroW) or SAHI slices merged with NMS/WBF.

### 2. Match the training scale to the test scale
- **What:** cap the long side of training frames (for example at 1920 px) so the object-size distribution matches the test set.
- **Why (our EDA):** train/val frames reach 3840 px (median long side 1904), while test frames are at most 1920 (median 1360, some at 960 and 680). The same object covers fewer pixels at test time. This is the idea behind Scale Match for tiny-person detection.
- **How:** resize before tiling. Also evaluate on val resized to test-like sizes, so validation reflects the leaderboard.

## Priority 2: if time allows

### 3. Per-clip frame step
- **What:** apply the IoU ≤ 0.5 rule to each clip separately instead of one global step.
- **Why:** the per-clip curves in the step plot vary a lot. Fast-moving clips should keep more frames and near-static ones fewer, which gives more variety for the same budget.

### 4. Rare-class oversampling
- **What:** repeat-factor sampling (LVIS): repeat frames or tiles that contain rare classes 2–3×.
- **Why:** bus is 0.2% of objects and truck 1.2%.
- **Alternative:** copy-paste of box crops. Ultralytics' built-in copy-paste needs segmentation masks, so it would have to be custom.

## Priority 3: experiments

### 5. Filter by occlusion
- **What:** train with vs. without heavily occluded boxes (COCO field `occlusion` = 2).
- **Why:** heavily occluded boxes may be label noise. It's a clean, cheap ablation.

### 6. Mask ignored regions in the loss instead of grey-filling them
- **What:** keep the original pixels and exclude anchors or proposals that overlap ignored regions from the loss.
- **Why:** the grey fill creates artificial edges, while loss masking keeps the image natural. Compare the two on Faster R-CNN.
- **Cost:** needs a custom change in torchvision's sampler.

### 7. Multi-scale training and photometric distortion
- **What:** random scale 0.5–1.5× and stronger colour jitter.
- **Why:** nearly every top VID2019 entry used multi-scale training and testing.
- **First:** check whether test clips include night scenes. The 2nd-place team balanced day and night with a GAN, which is too heavy for us; colour jitter is the cheap substitute.

## Validation and test-condition checks (from team discussion)

### 8. Rare classes are under-validated (bus, truck)
- **Problem:** in video, one bus on screen for 200 frames gives 200 near-identical boxes. What counts is the number of *distinct* objects (unique `track_id`s) and the number of clips they appear in. If val has only a handful of buses, bus AP is noisy, and it is 1/8 of mAP.
- **Do:** count unique tracks and clips per class per split, and add this to the EDA.
- **Do:** don't choose between models on rare-class AP alone. Report mAP with and without bus and truck.
- **Do:** for the final model, use 3-fold GroupKFold by clip so every clip, and every bus, is validated once.

### 9. Tune temporal post-processing on test-length clips
- **Problem:** test clips are short (3,811 frames / 48 clips ≈ 80 frames each), while train clips average about 430. A tracker (SORT/ByteTrack) needs a few frames to confirm a track, so on short clips the warm-up is a large share of each clip.
- **Note:** the per-frame detector has no memory, so cutting training clips doesn't change what it learns. This matters only for the temporal bonus.
- **Do:** cut val into ~80-frame chunks when tuning tracker or rescoring parameters (min hits, max age, score smoothing).

### 10. Low-resolution test clips (down to 680×382)
- **Problem:** CNN detectors accept any input size, but in a 680 px frame a car is only about 5–10 px.
- **Do (inference):** resize every test frame to the same long side (for example 1280). Low-res frames are then upscaled about 1.9×, bringing their objects closer to the training sizes.
- **Do (training):** add a "downscale then upscale" augmentation (for example `A.Downscale`, scale 0.35–0.7) to reproduce the blur of low-res clips.
- **Check:** evaluate on val downscaled to 680 px to measure the drop and whether the augmentation recovers it.

### Decided against
- **Capping detections at 200–300:** use `max_det = 500` for CNN detectors. mAP only ranks predictions, so extra low-confidence boxes never hurt, while a cap can push true positives out in dense frames. 300 queries is fine for DETR (the busiest frames have about 150–180 objects).
- **Training a detector for ignored regions:** detections there are neither rewarded nor penalised, so it earns no points. The grey fill already stops the model learning that crowds are background.

## Skip

- 3D-conv or GAN-based data synthesis: too expensive for the remaining time.
- Re-labelling missing annotations: too slow.

## Sources

- [VisDrone-VID2019 challenge results (Zhu et al., ICCVW 2019)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Zhu_VisDrone-VID2019_The_Vision_Meets_Drone_Object_Detection_in_Video_Challenge_ICCVW_2019_paper.pdf)
- [Tackling the Background Bias in Sparse Object Detection via Cropped Windows (Varga & Zell, ICCVW 2021)](https://ar5iv.labs.arxiv.org/html/2106.02288)
- [The Power of Tiling for Small Object Detection (CVPRW 2019)](https://mlanthology.org/cvprw/2019/unel2019cvprw-power)
- [Dynamic Tiling for small object detection](https://arxiv.org/pdf/2309.11069)
- [GBR Starfish Kaggle solution: tiling, GroupKFold by video, tracking, WBF](https://www.github.com/lucaordronneau/gbr-starfish-kaggle)
