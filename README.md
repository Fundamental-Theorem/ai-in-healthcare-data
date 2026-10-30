
# Artificial Intelligence for Healthcare Data

### Machine Learning, Deep Learning & Signal Processing Applications in Healthcare

This project explores the application of Artificial Intelligence and Machine Learning techniques to healthcare data across three different medical domains:

- **Prostate Cancer:** Statistical analysis and machine learning for predicting metastatic prostate cancer using clinical and laboratory data.
- **Colon Capsule Endoscopy:** Deep learning approaches for polyp classification and semantic segmentation in medical images.
- **Epileptic Seizure Detection:** EEG signal processing and machine learning for identifying epileptic seizure activity.

Each application addresses a different type of healthcare data:

| Application | Data Type | Main Objective |
|---|---|---|
| Prostate Cancer | Clinical and laboratory data | Predict metastatic disease |
| Colon Polyp Analysis | Medical images | Classify and segment polyps |
| Seizure Detection | EEG time-series signals | Detect epileptic seizures |

The project was developed as part of the *Artificial Intelligence for Healthcare Data* course at the University of Southern Denmark (SDU). A report in featured as `AI_for_Healthcare_Data.pdf`

---

<table>
  <tr><img src="Part II/polyp_characterization.jpg" width="600"/></tr>
</table>
*polyp characterization via MobileNetV2*


## 1. Prostate Cancer Metastasis Prediction

### Objective

Investigate whether clinical characteristics, biochemical measurements, and patient medical histories can be used to predict metastatic prostate cancer.

The analysis is based on a cohort of **1,159 patients** diagnosed with prostate cancer.

### Data Processing and Exploratory Analysis

The analysis includes three datasets:

- General patient information, including age, civil status, family cancer history, and metastasis status.
- Laboratory measurements, including PSA, monocytes, lymphocytes, thrombocytes, erythrocytes, albumin, creatinine, globulin, urea, and coagulation factor.
- Patient journal information containing ICD-10 diagnostic codes.

The preprocessing pipeline includes:

- Missing-value analysis and visualization.
- Date conversion and age calculation.
- ICD-10 diagnostic code mapping.
- Transformation of patient diagnoses into binary features.
- Median imputation for missing laboratory values.
- Correlation analysis and visualization.

### Statistical Analysis and Feature Reduction

Several statistical approaches were investigated:

- PSA threshold analysis using confusion matrices.
- Receiver Operating Characteristic (ROC) curves.
- Optimal PSA threshold selection using:
  - Youden Index.
  - Maximum Accuracy.
  - Minimum Distance to the Upper-Left Corner.
- Principal Component Analysis (PCA).
- Variance Inflation Factor (VIF) analysis.

PCA identified three principal components explaining approximately **69% of the total variance**.

### Machine Learning Models

The following classification algorithms were evaluated:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest
- XGBoost

Feature selection and model evaluation were also investigated using Recursive Feature Elimination (RFE) and 10-fold cross-validation.

### Results

The final model comparison reported the following results:

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7886 | 0.6722 | 0.8550 | 0.6979 | 0.8393 |
| SVM (RBF) | 0.7645 | 0.6580 | 0.8252 | 0.6699 | 0.8153 |
| Random Forest | 0.7826 | 0.6627 | 0.8509 | 0.6889 | 0.8372 |
| XGBoost | 0.7731 | 0.6461 | 0.8455 | 0.6741 | 0.8334 |

Logistic Regression achieved the strongest overall performance in the reported comparison.

The analysis also highlighted the importance of addressing multicollinearity, particularly among biochemical measurements.

---

## 2. Colon Polyp Analysis

### Objective

Investigate deep learning methods for analyzing Colon Capsule Endoscopy (CCE) images, focusing on two tasks:

1. **Polyp Classification:** Distinguishing neoplastic from non-neoplastic polyps.
2. **Semantic Segmentation:** Identifying polyp regions at pixel level.

### A. Polyp Classification

A pretrained **MobileNetV2** architecture was used for binary polyp classification.

MobileNetV2 was selected because of its computational efficiency and relatively lightweight architecture.

The model was evaluated using:

- Accuracy
- Sensitivity
- Specificity
- F1-score
- ROC-AUC

### Results

| Metric | Score |
|---|---:|
| Accuracy | 94.48% |
| Sensitivity | 89.68% |
| Specificity | 98.39% |
| F1-score | 93.58% |
| AUC | 98.49% |

The model demonstrated strong classification performance, although the training and validation curves indicated mild overfitting.

### B. Semantic Segmentation

For pixel-level polyp segmentation, a pretrained **TransUNet** architecture was investigated.

TransUNet combines convolutional neural networks with Transformer-based self-attention mechanisms to capture both local image features and global contextual information.

The evaluation focused on:

- Accuracy
- Precision
- Recall
- F1-score
- Intersection over Union (IoU)

### Results

| Metric | Score |
|---|---:|
| Accuracy | 91.62% |
| Precision | 95.41% |
| Recall | 93.73% |
| F1-score | 94.56% |
| IoU | 89.69% |

The segmentation model achieved strong results, with high precision and recall, indicating effective identification of polyp regions.

---

## 3. Epileptic Seizure Detection

### Objective

Develop a machine learning pipeline capable of identifying epileptic seizure activity from EEG recordings.

The analysis uses approximately two-hour EEG recordings containing clinically annotated absence seizures.

### Signal Processing

The EEG preprocessing pipeline includes:

- FIR band-pass filtering (1–70 Hz).
- Optional channel normalization.
- Extraction of seizure annotations from EDF files.
- Segmentation into overlapping windows:
  - Window duration: 3 seconds.
  - Overlap: 1 second.
- Binary labeling based on overlap with annotated seizure intervals.

### Feature Extraction

A 14-dimensional feature vector is extracted for each EEG channel and time window.

The features include:

**Statistical and time-domain features**
- Mean
- Median
- Variance
- Kurtosis
- Linear Energy
- Teager-Kaiser Energy
- Shannon Entropy

**Frequency-domain features**
- Delta band power
- Theta band power
- Alpha band power
- Beta band power
- Gamma band power
- Peak Frequency
- Spectral Edge Frequency (SEF 90%)

Welch's method is used for Power Spectral Density (PSD) estimation.

### Dimensionality Reduction

Before classification:

1. Invalid feature values are replaced with zeros.
2. Features are standardized using StandardScaler.
3. PCA is applied, retaining 90% of the total variance.
4. Whitening is enabled to decorrelate the principal components.

The dataset is divided chronologically into:
- 80% training data.
- 20% validation data.

### Machine Learning Models

Two classification algorithms were investigated:

- Logistic Regression
- Support Vector Machine (SVM) with an RBF kernel

GridSearchCV was used to optimize the SVM, prioritizing the reduction of false negatives.

### Results

| Metric | Logistic Regression | SVM (RBF) |
|---|---:|---:|
| True Negatives | 661 | 671 |
| False Positives | 30 | 20 |
| False Negatives | 6 | 7 |
| True Positives | 24 | 23 |
| ROC-AUC | 0.941 | 0.932 |

Logistic Regression achieved a slightly higher ROC-AUC, while the optimized SVM produced fewer false-positive detections.

The comparison highlights the trade-off between seizure detection sensitivity and false alarms.

---

## Technologies and Methods

The project combines several techniques from data science, machine learning, deep learning, and signal processing.

### Statistical Analysis
- Exploratory Data Analysis (EDA)
- Missing-value analysis and imputation
- Correlation analysis
- Principal Component Analysis (PCA)
- Variance Inflation Factor (VIF)
- ROC and AUC analysis

### Machine Learning
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- XGBoost
- Recursive Feature Elimination (RFE)
- GridSearchCV
- Cross-validation

### Deep Learning
- MobileNetV2
- TransUNet
- CNN architectures
- Transformer-based architectures
- Image classification
- Semantic segmentation

### Signal Processing
- EEG preprocessing
- FIR filtering
- Autocorrelation analysis
- Welch's Power Spectral Density estimation
- Time-domain and frequency-domain feature extraction

---

## Results Summary

The project demonstrates how different AI methodologies can be adapted to different types of medical data.

| Domain | Approach | Main Result |
|---|---|---|
| Prostate Cancer | Classical ML | Logistic Regression AUC: 0.8393 |
| Polyp Classification | MobileNetV2 | AUC: 0.9849 |
| Polyp Segmentation | TransUNet | IoU: 0.8969 |
| Seizure Detection | Logistic Regression | AUC: 0.941 |

## Limitations

Despite the promising results, several limitations should be considered:

- Medical datasets may contain missing values and population-specific biases.
- Prostate cancer prediction is affected by multicollinearity among laboratory variables.
- The colon polyp classification model exhibits mild overfitting.
- External validation is necessary to assess the generalizability of the medical imaging models.
- EEG seizure detection remains challenging due to the complexity and non-stationarity of physiological signals.
- False-negative predictions are particularly important in medical applications and require careful evaluation.

**These models are developed for academic and research purposes and should not be considered clinically validated diagnostic systems.**

---

## Disclaimer

This repository contains  experimental implementations.

The reported metrics reflect the experiments described in the accompanying project report. They do not establish clinical effectiveness or readiness for deployment.

Patient data and medical images should only be shared in accordance with applicable privacy, licensing, and institutional requirements.

