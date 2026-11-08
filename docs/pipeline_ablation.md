# Final-pipeline ablation (CPU, val subset)

122 of the 359 val frames (every third frame of each clip), shared evaluator, models from the 4-epoch runs:
`yolo_low_scale_960` (YOLO26s, 960), `yolo_baseline_1280` (YOLO26s, 1280), `rtdetr_960` (RT-DETR-L, 960).
Tiles: 640 px, 20% overlap, model input 640. Passes merged with class-wise NMS (IoU 0.6); models fused with WBF.
Numbers are only comparable within this table (the subset scores higher than full val: e.g. yolo_low_scale_960 full = 0.2380 here vs 0.2202 on all 359).
Regenerate on all val frames with `09_final_pipeline.ipynb`.

## Single models and WBF (main run)

| Variant | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR |
|---|---|---|---|---|---|---|---|
| yolo_low_scale_960 full | 0.2380 | 0.4191 | 0.2303 | 0.0987 | 0.1897 | 0.4442 | 0.3538 |
| yolo_low_scale_960 +flip | 0.2400 | 0.4316 | 0.2327 | 0.1092 | 0.1960 | 0.4390 | 0.3665 |
| yolo_low_scale_960 +tiles | 0.2219 | 0.4029 | 0.2132 | 0.1537 | 0.1836 | 0.3935 | 0.3782 |
| yolo_low_scale_960 +tiles+flip | 0.2240 | 0.4098 | 0.2153 | 0.1561 | 0.1870 | 0.3994 | 0.3871 |
| yolo_low_scale_960 tiles only | 0.0949 | 0.1886 | 0.0873 | 0.1325 | 0.0977 | 0.1588 | 0.2908 |
| yolo_baseline_1280 full | 0.2216 | 0.4222 | 0.2064 | 0.1585 | 0.2193 | 0.3529 | 0.3546 |
| yolo_baseline_1280 +flip | 0.2225 | 0.4333 | 0.2053 | 0.1376 | 0.2186 | 0.3519 | 0.3596 |
| yolo_baseline_1280 +tiles | 0.2141 | 0.4149 | 0.1953 | 0.1550 | 0.2198 | 0.3241 | 0.3735 |
| yolo_baseline_1280 +tiles+flip | 0.2171 | 0.4212 | 0.2008 | 0.1559 | 0.2214 | 0.3307 | 0.3806 |
| yolo_baseline_1280 tiles only | 0.1327 | 0.2470 | 0.1301 | 0.1243 | 0.1380 | 0.2165 | 0.3301 |
| rtdetr_960 full | 0.2060 | 0.3726 | 0.2074 | 0.0669 | 0.1725 | 0.3781 | 0.3863 |
| rtdetr_960 +flip | 0.2388 | 0.4512 | 0.2283 | 0.0884 | 0.1990 | 0.4279 | 0.3904 |
| rtdetr_960 +tiles | 0.1846 | 0.3798 | 0.1656 | 0.0831 | 0.1722 | 0.3724 | 0.4206 |
| rtdetr_960 +tiles+flip | 0.1984 | 0.3995 | 0.1819 | 0.0869 | 0.1813 | 0.3920 | 0.4326 |
| rtdetr_960 tiles only | 0.0774 | 0.1634 | 0.0697 | 0.0591 | 0.0860 | 0.1486 | 0.3059 |
| WBF yolo960+yolo1280 full | 0.2467 | 0.4480 | 0.2402 | 0.1242 | 0.2105 | 0.4293 | 0.4006 |
| WBF yolo960+rtdetr full | 0.2437 | 0.4507 | 0.2326 | 0.0850 | 0.1940 | 0.4489 | 0.4076 |
| WBF all3 full | 0.2577 | 0.4737 | 0.2494 | 0.0998 | 0.2195 | 0.4508 | 0.4314 |
| WBF all3 +tiles | 0.2349 | 0.4441 | 0.2215 | 0.1423 | 0.2092 | 0.4232 | 0.4717 |
| WBF all3 +tiles+flip | 0.2349 | 0.4458 | 0.2220 | 0.1416 | 0.2095 | 0.4186 | 0.4762 |

## Extra variants (from the same cached passes)

| Variant | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR |
|---|---|---|---|---|---|---|---|
| yolo_low_scale_960 +tiles small<64 | 0.2319 | 0.4214 | 0.2176 | 0.1539 | 0.1866 | 0.4375 | 0.3699 |
| yolo_low_scale_960 +flip+tiles small<64 | 0.2357 | 0.4316 | 0.2238 | 0.1563 | 0.1944 | 0.4362 | 0.3811 |
| yolo_low_scale_960 +tiles small<96 | 0.2236 | 0.4092 | 0.2102 | 0.1536 | 0.1803 | 0.4345 | 0.3754 |
| yolo_low_scale_960 +flip+tiles small<96 | 0.2256 | 0.4160 | 0.2127 | 0.1561 | 0.1841 | 0.4324 | 0.3857 |
| yolo_baseline_1280 +tiles small<64 | 0.2186 | 0.4249 | 0.1986 | 0.1551 | 0.2202 | 0.3493 | 0.3556 |
| yolo_baseline_1280 +flip+tiles small<64 | 0.2216 | 0.4317 | 0.2043 | 0.1559 | 0.2214 | 0.3495 | 0.3651 |
| yolo_baseline_1280 +tiles small<96 | 0.2172 | 0.4218 | 0.1982 | 0.1550 | 0.2176 | 0.3515 | 0.3600 |
| yolo_baseline_1280 +flip+tiles small<96 | 0.2212 | 0.4291 | 0.2057 | 0.1559 | 0.2200 | 0.3503 | 0.3680 |
| rtdetr_960 +tiles small<64 | 0.2018 | 0.3968 | 0.1889 | 0.0834 | 0.1779 | 0.4163 | 0.3761 |
| rtdetr_960 +flip+tiles small<64 | 0.2126 | 0.4176 | 0.1997 | 0.0871 | 0.1876 | 0.4277 | 0.4002 |
| rtdetr_960 +tiles small<96 | 0.1942 | 0.3925 | 0.1765 | 0.0831 | 0.1689 | 0.4171 | 0.3791 |
| rtdetr_960 +flip+tiles small<96 | 0.2069 | 0.4133 | 0.1908 | 0.0869 | 0.1797 | 0.4277 | 0.4012 |
| WBF all3 +flip iou0.55 | 0.2680 | 0.4963 | 0.2618 | 0.1347 | 0.2318 | 0.4605 | 0.4422 |
| WBF all3 +flip iou0.6 | 0.2744 | 0.5041 | 0.2656 | 0.1393 | 0.2377 | 0.4753 | 0.4450 |
| WBF all3 +flip+tiles small<64 iou0.55 | 0.2573 | 0.4802 | 0.2493 | 0.1424 | 0.2242 | 0.4645 | 0.4396 |
| WBF all3 +flip+tiles small<64 iou0.6 | 0.2638 | 0.4884 | 0.2550 | 0.1519 | 0.2304 | 0.4736 | 0.4396 |
| WBF all3 +flip+tiles small<96 iou0.55 | 0.2456 | 0.4610 | 0.2369 | 0.1415 | 0.2101 | 0.4609 | 0.4398 |
| WBF all3 +flip+tiles small<96 iou0.6 | 0.2561 | 0.4738 | 0.2496 | 0.1516 | 0.2211 | 0.4706 | 0.4412 |
| WBF all3 +flip, YOLO960 tiles small<96 iou0.55 | 0.2656 | 0.4939 | 0.2579 | 0.1668 | 0.2314 | 0.4613 | 0.4448 |
| WBF all3 +flip, YOLO960 tiles small<96 iou0.6 | 0.2722 | 0.5017 | 0.2636 | 0.1716 | 0.2369 | 0.4745 | 0.4491 |
| WBF all3 +flip w=2,1,2 | 0.2749 | 0.5012 | 0.2704 | 0.1228 | 0.2267 | 0.4737 | 0.4385 |
| WBF all3 +flip iou0.65 | 0.2770 | 0.5059 | | 0.1412 | | | |
| WBF all3 +flip iou0.7 | 0.2767 | 0.5046 | | 0.1425 | | | |
