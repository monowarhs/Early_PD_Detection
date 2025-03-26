# Early_PD_Detection
This repository demonstrates a machine learning pipeline for classifying Parkinson's Disease using LightGBM and explaining model decisions with SHAP. 

## Overview
This project identifies early-stage Parkinson’s disease (PD) using voice biomarkers and explains predictions with SHAP (SHapley Additive exPlanations). 

**Dataset**: [Oxford PD Dataset (Kaggle)](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)  

**Model**: LightGBM classifier optimized for accuracy and interpretability.

## Key Features
- **Feature Selection**: Identifies top 3 voice biomarkers (Jitter, Shimmer, HNR) for PD detection.
- **Explainable AI**: SHAP plots reveal model decision logic.
- **Minimalist Model**: Achieves >90% accuracy with only 3 features.
- **Clinical Relevance**: Prioritizes metrics used in real-world PD diagnosis.

## Prerequisites
- Python 3.8+
- Libraries: `lightgbm`, `shap`, `pandas`, `scikit-learn`
