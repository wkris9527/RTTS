# Dataset preparation

Only validation splits are used. Download datasets from their original providers and follow [MMSegmentation's preparation instructions](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md). Datasets and annotations are not redistributed here.

| Preset | Dataset root passed to `--data-dir` | Required files and directories |
| :--- | :--- | :--- |
| `v20`, `v21` | `VOC2012/` | `JPEGImages/`, `SegmentationClass/`, `ImageSets/Segmentation/val.txt` |
| `p59`, `p60` | Prepared PASCAL Context root | `JPEGImages/`, `SegmentationClassContext/`, `ImageSets/SegmentationContext/val.txt` |
| `cityscapes` | Cityscapes root | `leftImg8bit/val/`, `gtFine/val/` with `_gtFine_labelTrainIds.png` |
| `coco_obj`, `coco_stuff` | Prepared COCO root | `images/val2017/`, `annotations/val2017/` with `_labelTrainIds.png` |

VOC21/Context60 include background. VOC20 excludes the VOC background label by remapping it to ignore; Context59 uses zero-label reduction. COCOObjectDataset maps the first 80 categories to objects and ignores remaining labels. COCOStuffDataset contains 171 categories in the released metadata. Check prepared label IDs against `utils_local/segmentation_datasets.py` before evaluating; directory existence alone does not validate annotations.

Class synonyms are provided under `utils_local/class_extensions/`. The launcher enables them and converts synonym-level predictions back to benchmark categories.

## Preprocessing

The standard presets resize non-Cityscapes images to 224 x 224. The Cityscapes preset retains the archived resize of 1120 x 560 (orientation-adjusted), then extracts 224 x 224 windows with stride 112. Ground truth is resized with nearest-neighbor interpolation and evaluated at the resulting resolution. This archived resize detail should be considered when comparing to protocols that score original-resolution images.

Corruptions are applied before resizing, at severity 5, with a deterministic seed based on dataset sample index. `original` applies no corruption. The 15 evaluation corruptions are Gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelation, and JPEG compression.
