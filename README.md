# Face Segmentation with U-Net and ResNet-34

A PyTorch-based multi-class face segmentation project built on the CelebAMask-HQ dataset, using a U-Net decoder with a ResNet-34 encoder pretrained on ImageNet. The pipeline covers data preparation, reproducible train/validation/test splitting, augmentation, training, evaluation, and qualitative visualization of predicted masks.

## Overview

This project performs semantic segmentation of facial regions at the pixel level. The model predicts six classes: background, skin, eyebrow, eye, nose, and mouth/lips.

## Highlights

- U-Net with **ResNet-34** encoder.
- Multi-class segmentation with 6 output classes.
- Albumentations-based preprocessing and augmentation.
- Reproducible data splits with fixed seed `42`.
- Training with combined CrossEntropyLoss and multiclass Dice loss.
- Validation and test evaluation with Dice and IoU metrics.
- Qualitative visualizations for masks and model predictions.

## Dataset

The notebook uses **CelebAMask-HQ** and reports **30,000 usable images**. A full split is created as:

- Train: 24,000 images
- Validation: 3,000 images
- Test: 3,000 images

The notebook also creates a smaller **12,000-image subset** for faster experiments:

- Train: 9,600 images
- Validation: 1,200 images
- Test: 1,200 images

## Classes

## CelebAMask-HQ Original Classes

The CelebAMask-HQ dataset originally contains 19 semantic classes, including facial parts and accessories. In this project, we only use the **face-related classes** and exclude accessory classes such as hair, hat, neck, and cloth.

| ID | Original CelebAMask-HQ Class |
|---|---|
| 0 | background |
| 1 | skin |
| 2 | nose |
| 3 | eye_g |
| 4 | l_eye |
| 5 | r_eye |
| 6 | l_brow |
| 7 | r_brow |
| 8 | l_ear |
| 9 | r_ear |
| 10 | mouth |
| 11 | u_lip |
| 12 | l_lip |
| 13 | hair |
| 14 | hat |
| 15 | ear_r |
| 16 | neck_l |
| 17 | neck |
| 18 | cloth |

and our custom classes:

| ID | Class |
|---|---|
| 0 | Background |
| 1 | Skin |
| 2 | Eyebrow |
| 3 | Eye |
| 4 | Nose |
| 5 | Mouth / Lips |

## Model

The segmentation model is defined with:

- Architecture: `U-Net`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input size: `512 × 512`
- Number of classes: `6`

## Training Setup

Main training configuration from the notebook:

- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: `8`
- Epochs: `20`
- LR scheduler: `ReduceLROnPlateau`
- Model selection metric: validation mean IoU over face classes `1..5`

## Best Checkpoint

The best trained checkpoint can be downloaded from [Google Drive](https://drive.google.com/file/d/1nKL4BsMTXplF0kJfZts0XvTlrM6MdCv8/view?usp=drive_link).

## Results

Best metrics reported in the notebook:

## Evaluation Results

| Split | Images | Pixel Accuracy | Mean IoU (Face Classes) | Mean Dice (Face Classes) | Background IoU / Dice | Skin IoU / Dice | Eyebrow IoU / Dice | Eye IoU / Dice | Nose IoU / Dice | Mouth / Lips IoU / Dice |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| Validation | 1200 | 0.9763 | 0.8606 | 0.9239 | 0.9753 / 0.9875 | 0.9165 / 0.9564 | 0.7582 / 0.8625 | 0.8268 / 0.9052 | 0.8879 / 0.9406 | 0.9135 / 0.9548 |
| Test | 1200 | 0.9769 | 0.8629 | 0.9254 | 0.9762 / 0.9880 | 0.9188 / 0.9577 | 0.7648 / 0.8667 | 0.8322 / 0.9084 | 0.8897 / 0.9417 | 0.9090 / 0.9523 |

## Final Report

| Metric | Validation | Test |
|---|---:|---:|
| Mean IoU (Face Classes) | 0.8606 | 0.8629 |
| Mean Dice (Face Classes) | 0.9239 | 0.9254 |
| Pixel Accuracy | 0.9763 | 0.9769 |

These results indicate strong segmentation quality across the selected face-part classes.

## Visual Results

### Sample training data
This is one example of the training dataset with its annotation: 

<p align="center">
  <img src="original_image and mask.png" alt="Sample training visualization" width="700">
</p>


### Prediction examples

The notebook also includes qualitative prediction visualizations for segmentation outputs and overlays:


<p align="center">
  <img src="face_segmentation_results.jpg" alt="Prediction results" width="700">
</p>




## Project Structure

```bash
FaceSegmentationProject/
├── checkpoints/
│   ├── bestunetresnet34.pth
│   └── lastunetresnet34.pth
├── logs/
│   └── traininghistory.csv
├── config.json
├── trainids.txt
├── valids.txt
└── testids.txt
```

## Installation

```bash
pip install torch torchvision
pip install segmentation-models-pytorch timm
pip install albumentations opencv-python matplotlib pillow
pip install scikit-learn numpy pandas tqdm
```

## Usage

1. Open `Face_Segmentation_final.ipynb`.
2. Set dataset paths correctly.
3. Run preprocessing and splitting cells.
4. Train the model or load the saved checkpoint.
5. Evaluate on validation and test sets.
6. Run inference cells to visualize predictions.

## Notes

- The notebook is configured for GPU training when CUDA is available.
- Best and last checkpoints are saved automatically during training.
- The code supports both full-data and reduced-data experiments.

## Acknowledgments

- Computer Vision
- CelebAMask-HQ dataset
- PyTorch
- segmentation-models-pytorch
- Albumentations

## Any Question

If you have any question, please don't hesitate to ask me.
heliazakeriyan@gmail.com


```

