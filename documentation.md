# AlphaPolyp – Technical Documentation

This document provides a comprehensive technical overview of AlphaPolyp, serving as the authoritative reference for engineers, researchers, clinicians, and stakeholders by detailing the project’s objectives, data pipeline, model architecture, training strategy, evaluation outcomes, deployment procedures, and future business roadmap.

## Overview

AlphaPolyp is an AI assistant that automatically segments colorectal polyps in colonoscopy frames _and_ estimates their 3‑D morphology (overall volume + orthogonal diameters x, y, z).  
In clinical workflow this eliminates the extra CT‑colonography step currently required to size a lesion and helps the endoscopist select an appropriately sized polypectomy snare on the spot.

---

## System Architecture & Data Flow

The system architecture comprises the following components:

*   **Flask Web Application:** Handles user interface, image upload, result visualization, and communication with the Model API.
*   **Model API:** Performs the core image processing tasks, including polyp segmentation and feature extraction.
### Code Structure
*   `app.py`: Contains the Flask application logic, including routing, image processing, and API communication.
*   `templates/`: Contains HTML templates for the web interface.
    *   `index.html`: Main page for image upload.
    *   `result.html`: Displays the results of the polyp detection.
    *   `about.html`: About page.
*   `static/`: Contains static files such as images, CSS, and JavaScript.
    *   `uploads/`: Temporarily stores uploaded images.
    *   `results/`: Stores the visualization results.
### Data Flow

1.  **Image Upload:**
    *   The user uploads an image through the web interface.
2.  **Flask Application (Backend):**
    *   Receives the uploaded image.
    *   Generates a unique filename.
    *   Saves the image temporarily to the `static/uploads` directory.
    *   Sends the image to the Model API.
3.  **Model API:**
    *   Receives the image.
    *   Processes the image for polyp segmentation.
    *   Returns a JSON response containing:
        *   `volume`: Volume of the detected polyp.
        *   `dimensions`: Dimensions (x, y, z) of the detected polyp.
        *   `segmentation`: Segmentation map of the polyp.
        *   `processing_time`: Time taken for processing.
4.  **Flask Application (Post-processing & Visualization):**
    *   Receives the JSON response.
    *   Extracts `volume`, `dimensions`, and `segmentation` data.
    *   Calls the `visualize_results` function to:
        *   Read the original image.
        *   Overlay the segmentation map.
        *   Combine the original image and overlay.
    *   Saves the visualization to the `static/results` directory.
5.  **Results Display:**
    *   Renders the `result.html` template.
    *   Displays:
        *   Original image.
        *   Segmentation visualization.
        *   Polyp `volume` and `dimensions`.
        *   `processing_time`.

## Configuration

The following environment variables can be configured:

*   `MODEL_API_URL`: The URL of the Model API. Defaults to `http://model:5001`.
*   `PORT`: The port on which the Flask application runs. Defaults to `5000`.

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

![RAPUNet diagram](https://github.com/LineIntegralx/AlphaPolyp/blob/main/images/model_architecture.png)

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

