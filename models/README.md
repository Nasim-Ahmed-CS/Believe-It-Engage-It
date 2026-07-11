# Engagement Classification Models

This folder contains separate training scripts for each engagement classification model used in the ISMAR'26 paper.

## Models Implemented

All models are optimized for engagement classification and save trained models for later use:

### 1. **LSTM** (`train_lstm_optimized.py`)
- Bidirectional LSTM with metadata features
- Binary classification (Low vs High engagement)
- Saves model as `.keras` file
- Uses JAX backend for performance

### 2. **GRU** (`train_gru_optimized.py`)
- Bidirectional GRU with metadata features
- Binary classification (Low vs High engagement)
- Saves model as `.keras` file
- Uses JAX backend for performance

### 3. **XGBoost** (`train_xgboost_optimized.py`)
- Gradient boosting classifier
- 3-class classification (Low/Medium/High engagement)
- Hyperparameter tuning with RandomizedSearchCV
- Saves model as `.pkl` file
- Target: ~68-70% accuracy

### 4. **Random Forest** (`train_randomforest_optimized.py`)
- Ensemble bagging classifier
- 3-class classification (Low/Medium/High engagement)
- Hyperparameter tuning with RandomizedSearchCV
- Saves model as `.pkl` file
- Target: ~68-70% accuracy

### 5. **Decision Tree** (`train_decisiontree_optimized.py`)
- Interpretable single-tree classifier
- 3-class classification (Low/Medium/High engagement)
- Hyperparameter tuning with RandomizedSearchCV
- Saves model as `.pkl` file
- Target: ~68-70% accuracy

### 6. **EBM - Explainable Boosting Machine** (`train_ebm_optimized.py`)
- Glass-box interpretable boosting model
- 3-class classification (Low/Medium/High engagement)
- Built-in explainability features
- Saves model as `.pkl` file
- Target: ~80% accuracy with interpretability

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

For EBM model specifically:
```bash
pip install interpret
```

## Usage

### Option 1: One-Click Training (Recommended)

**Windows Batch Script (Double-click to run):**
```
models\train_all_models.bat
```

**PowerShell Script (Right-click → Run with PowerShell):**
```
models\train_all_models.ps1
```

**Python Script (Cross-platform):**
```bash
cd models
python run_all_models.py
```

All three scripts will train all 6 models sequentially and provide a summary report.

### Option 2: Train Individual Models

Each script can be run independently:

```bash
# Train LSTM model
python models/train_lstm_optimized.py

# Train GRU model
python models/train_gru_optimized.py

# Train XGBoost model
python models/train_xgboost_optimized.py

# Train Random Forest model
python models/train_randomforest_optimized.py

# Train Decision Tree model
python models/train_decisiontree_optimized.py

# Train EBM model (requires 'interpret' package)
python models/train_ebm_optimized.py
```

## Saved Models

All trained models are saved in the `models/saved_models/` directory:

- `lstm_engagement_optimized.keras` - LSTM model
- `gru_engagement_optimized.keras` - GRU model
- `xgboost_engagement_optimized.pkl` - XGBoost model
- `randomforest_engagement_optimized.pkl` - Random Forest model
- `decisiontree_engagement_optimized.pkl` - Decision Tree model
- `ebm_engagement_optimized.pkl` - EBM model

## Data Requirements

All scripts expect the following data structure:

```
ISMAR26_Engagement/
├── MultiModalAttack/
│   ├── Processed Data/
│   │   └── DI_Engagement_False_Info_*.csv
│   └── Trust and Engagement Self Report_Multi_Modal.csv
└── TextualAttack/
    ├── Processed Data/
    │   └── DI_Engagement_False_Info_Text_*.csv
    └── Trust and Engagement Self Report_TextOnly.csv
```

## Features

Each script includes:

- ✅ **Data Loading**: Automatic loading from MultiModal and TextualAttack folders
- ✅ **Preprocessing**: Sequence interpolation, smoothing, and feature engineering
- ✅ **Class Balancing**: Perfect balance via downsampling for 3-class models
- ✅ **Hyperparameter Tuning**: RandomizedSearchCV for tree-based models
- ✅ **Evaluation Metrics**: Accuracy, Balanced Accuracy, F1, AUC, Classification Report
- ✅ **Model Saving**: Persistent storage for trained models
- ✅ **Reproducibility**: Fixed random seeds (42)

## Model Comparison

| Model | Type | Classes | Target Accuracy | Interpretability | Training Time |
|-------|------|---------|----------------|------------------|---------------|
| LSTM | Sequence | 2 (Binary) | ~85% | Low | Medium |
| GRU | Sequence | 2 (Binary) | ~85% | Low | Medium |
| XGBoost | Tree Ensemble | 3-class | ~68-70% | Medium | Fast |
| Random Forest | Tree Ensemble | 3-class | ~68-70% | Medium | Fast |
| Decision Tree | Single Tree | 3-class | ~68% | High | Very Fast |
| EBM | Boosting | 3-class | ~80% | **Very High** | Slow |

## Notes

- **LSTM/GRU**: Use binary classification (Low vs High) by removing medium tertile
- **Tree Models**: Use 3-class classification (Low/Medium/High) with quartile-based smoothing
- **Optimized Naming**: No "relaxed" variants - only optimized models for production use
- **Performance**: All models target minimum 68% accuracy as specified
- **Explainability**: EBM provides built-in interpretability features via `interpret.show()`

## Citation

If you use these models, please cite:

```
Ahmed, N., et al. (2026). Believe It. Engage It.: A Mixed Reality Dataset of 
Engagement, Cognitive Load, Physical Load, Cybersickness, Task Performance 
Under Cognitive Attack. ISMAR 2026.
```

## Contact

For questions or issues, refer to the main project README or contact the authors.
