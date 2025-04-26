# AlphaPolyp – Technical Documentation

This document provides a comprehensive technical overview of AlphaPolyp, serving as the authoritative reference for engineers, researchers, clinicians, and stakeholders by detailing the project’s objectives, data pipeline, model architecture, training strategy, evaluation outcomes, deployment procedures, and future business roadmap.

## Overview

AlphaPolyp is an AI assistant that automatically segments colorectal polyps in colonoscopy frames _and_ estimates their 3‑D morphology (overall volume + orthogonal diameters x, y, z).  
In clinical workflow this eliminates the extra CT‑colonography step currently required to size a lesion and helps the endoscopist select an appropriately sized polypectomy snare on the spot.


---

## System Architecture & Data Flow

### Front‑End

- **Next.js** – Renders UI components, handles routing, and manages user interactions.
- **Tailwind CSS** – Utility‑first CSS framework for styling and responsive design.
- **Custom Components** – Built with React and **shadcn/ui** for modularity and reusability.
- **TypeScript** – Ensures type safety and an improved developer experience.

### Back‑End

- **Flask** – Serves as the main REST API server, handling requests and responses.
- **TensorFlow / Keras** – Runs the custom CNN model for polyp detection.
- **Model Weights** – Stored in `alphapolyp_optimized_model_3500cases.h5`.
- **Utilities** – Pre‑processing and post‑processing functions for image analysis.
- **REST API** – Exposes endpoints for image upload, analysis, and model information.

### Data Flow

1. **Image Upload** – User uploads an image via the frontend UI.
2. **API Request** – Frontend sends the image to the backend `/api/analyze` endpoint.
3. **Processing** – Backend preprocesses the image, runs inference with the ML model, and post‑processes the results.
4. **Response** – Backend returns analysis results (detected polyps, volume estimation and dimensions) to the frontend.
5. **Display** – Frontend visualizes results, highlights polyps, and provides user feedback.

---

## Dataset

- **Source:** Synthetic Colon Polyp CT & Colonoscopy Images (Mendeley Data, v1)  
  <https://data.mendeley.com/datasets/p2g5sk8brb/1>

- **Raw corpus:** ≈20 000 colonoscopy frames (CycleGAN-enhanced + fully-synthetic).

- **Cleaned corpus:** **18 378** images with:
  - 1 × RGB frame (500 × 500 px → resized to 352 × 352 px)
  - 1 × binary mask (polyp segmentation)
  - 4 × regression labels – volume, x, y, z (mm³ / mm).

- **Numeric distribution**

  | statistic | volume | x | y | z |
  |-----------|-------:|--:|--:|--:|
  | mean | 54.96 | 6.79 | 4.93 | 4.89 |
  | std dev | 51.76 | 6.25 | 2.12 | 2.15 |
  | min | 0.0004 | 0.005 | 0.002 | 0.001 |
  | max | 582.21 | 35.57 | 16.31 | 15.46 |

---

## Model architecture

![RAPUNet diagram](model_architecture.png)

* **Backbone:** RAPUNet encoder–decoder (MetaFormer CAFormer‑S18)  
* **Skip‑connected aggregation** → segmentation head (sigmoid, Dice)  
* Concatenate ↓sampled mask with encoder output → **regression MLP** (2 × Dense 64→32, ReLU)  
* Bias initialiser set to normalised global mean `[0.094, 0.191, 0.302, 0.317]`

(See `model_architecture/model.py` for full code.)

---

## Training pipeline

| phase | frozen layers | epochs | images | LR |
|-------|---------------|-------:|-------:|----|
| 1 – Regression Head | encoder+decoder | 15 | 2 000 | 1 e‑4 |
| 2 – Full fine‑tune | none | 30 | 2000 | 1 e‑5 |
| 3 – Incremental | none | 1 | +5 000 | 1 e‑5 |

* Optimiser **AdamW** (weight‑decay 1 e‑6)  
* Loss = Dice + MSE (equal weights)  
* A100 GPU   
* Best checkpoint ⇒ **`alphapolyp_optimized_model.h5`**

(See `train.py` for full code.)

---

## Evaluation

### 1. Synthetic vs. CycleGAN hold‑out (1 000 frames)

| task | metric | CycleGAN | Synthetic |
|------|--------|---------:|----------:|
| **Segmentation** | Dice ↑ | 0.976 | 0.981 |
| | Precision ↑ | 0.965 | 0.970 |
| | Recall ↑ | 0.963 | 0.963 |
| **Regression** | RMSE (mm³) ↓ | 39.10 | 37.25 |
| | Accuracy (<10 % error) ↑ | 65.7 % | 72.9 % |
| | µ Precision (±5 mm) ↑ | 0.782 | 0.812 |

(See `test.py` and `predict.py` for full code.)
### 2. Real‑world quality‑assurance 
**Dr Alaa Sharara, MD, FACG**  
Professor of Medicine, Division of Gastroenterology & Hepatology, American University of Beirut Medical Center   
- >25 years’ experience in diagnostic and therapeutic endoscopy  
- >200 peer-reviewed publications in inflammatory bowel disease & advanced polypectomy  
- >Former president, Lebanese Society of Gastroenterology

To obtain an independent clinical sanity-check, we exported 11 real colonoscopy images (not used for training/validation) along with the model’s predicted volume, x, y, z dimensions.  
Dr Sharara evaluated each case and assigned a qualitative score:

| Rating criterion | Definition (internal guideline) |
|------------------|---------------------------------|
| **Excellent**    | Predicted values ≈ clinical estimate ± ≤10 % |
| **Fair**         | Within 10–25 % of clinical estimate |
| **Poor**         | >25 % deviation or anatomically implausible |

| rating | count |
|--------|------:|
| Excellent | 2 |
| Fair | 6 |
| Poor | 3 |


---

## Edge cases & failure modes
* **Poor illumination / motion blur** – mask may fragment → under‑estimate volume  
* **Multiple adjacent polyps** – model merges instances  
* **Sub‑mucosal flat lesions** – low contrast, segmentation drop  
* **Camera distance < 2 cm** – perspective distortion beyond training distribution

Mitigation in future work: domain‑randomised augmentation, real intra‑op fine‑tuning, multi‑view fusion.

---

## Limitations & future work
* Lack of real labelled dataset  
* Compute constraints limited full 18 k image training  
* Road‑map: real‑time inference, 3‑D mesh reconstruction, snare‑size recommender.

---

## Files in repository
```
model_architecture/
 ├─ model.py
 ├─ RAPU_blocks.py
 ├─ ImageLoader2D.py
 └─ DiceLoss.py
train.py,  test.py,  predict.py
requirements.txt
```

---

## References
Sharara A. et al., AUBMC, Department of Gastroenterology – _Expert QA_, 2025.  
Yuan H. _“RAPUNet: Residual Aware Polyp U‑Net”_ (2024).  
Loshchilov I. & Hutter F., _“Decoupled Weight Decay Regularization”_ ICLR 2019.

