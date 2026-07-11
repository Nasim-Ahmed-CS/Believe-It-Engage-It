@echo off
REM Batch script to train all engagement classification models
REM Double-click this file to run all models sequentially

echo ================================================================================
echo ENGAGEMENT CLASSIFICATION MODEL TRAINING SUITE
echo ================================================================================
echo.
echo This will train all 6 engagement classification models sequentially.
echo Expected total time: 15-30 minutes depending on hardware.
echo.
echo Models to be trained:
echo   1. LSTM
echo   2. GRU
echo   3. XGBoost
echo   4. Random Forest
echo   5. Decision Tree
echo   6. EBM (Explainable Boosting Machine)
echo.
echo Trained models will be saved in: models\saved_models\
echo.
pause

cd /d "%~dp0"
echo.
echo Current directory: %CD%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Starting training...
echo.

REM Create saved_models directory if it doesn't exist
if not exist "saved_models" mkdir saved_models

REM Run all training scripts
echo ================================================================================
echo [1/6] Training LSTM Model...
echo ================================================================================
python train_lstm_optimized.py
if errorlevel 1 (
    echo WARNING: LSTM training failed
) else (
    echo SUCCESS: LSTM model trained
)
echo.

echo ================================================================================
echo [2/6] Training GRU Model...
echo ================================================================================
python train_gru_optimized.py
if errorlevel 1 (
    echo WARNING: GRU training failed
) else (
    echo SUCCESS: GRU model trained
)
echo.

echo ================================================================================
echo [3/6] Training XGBoost Model...
echo ================================================================================
python train_xgboost_optimized.py
if errorlevel 1 (
    echo WARNING: XGBoost training failed
) else (
    echo SUCCESS: XGBoost model trained
)
echo.

echo ================================================================================
echo [4/6] Training Random Forest Model...
echo ================================================================================
python train_randomforest_optimized.py
if errorlevel 1 (
    echo WARNING: Random Forest training failed
) else (
    echo SUCCESS: Random Forest model trained
)
echo.

echo ================================================================================
echo [5/6] Training Decision Tree Model...
echo ================================================================================
python train_decisiontree_optimized.py
if errorlevel 1 (
    echo WARNING: Decision Tree training failed
) else (
    echo SUCCESS: Decision Tree model trained
)
echo.

echo ================================================================================
echo [6/6] Training EBM Model...
echo ================================================================================
python train_ebm_optimized.py
if errorlevel 1 (
    echo WARNING: EBM training failed or interpret package not installed
) else (
    echo SUCCESS: EBM model trained
)
echo.

echo ================================================================================
echo ALL TRAINING COMPLETE!
echo ================================================================================
echo.
echo Trained models saved in: %CD%\saved_models\
echo.

REM List saved models
if exist "saved_models" (
    echo Saved model files:
    dir /b saved_models
    echo.
)

echo Press any key to exit...
pause >nul
