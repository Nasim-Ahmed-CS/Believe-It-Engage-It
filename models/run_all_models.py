"""
Run All Engagement Classification Models
Trains all models sequentially and compares their performance.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add models folder to path
MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(MODELS_DIR))


def run_model(model_name, script_name):
    """Run a single model training script."""
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Import and run the model's main function
        module = __import__(script_name.replace('.py', ''))
        module.main()
        
        elapsed = time.time() - start_time
        status = "SUCCESS"
        print(f"\n✓ {model_name} completed in {elapsed:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        status = "FAILED"
        print(f"\n✗ {model_name} failed after {elapsed:.1f}s")
        print(f"   Error: {str(e)}")
    
    return {
        'Model': model_name,
        'Status': status,
        'Time (s)': round(elapsed, 1)
    }


def main():
    print("="*80)
    print("ENGAGEMENT CLASSIFICATION MODEL TRAINING SUITE")
    print("="*80)
    print("\nThis will train all engagement classification models sequentially.")
    print("Expected total time: 15-30 minutes depending on hardware.\n")
    
    input("Press Enter to begin training all models, or Ctrl+C to cancel...")
    
    # Model configurations
    models = [
        ("LSTM", "train_lstm_optimized"),
        ("GRU", "train_gru_optimized"),
        ("XGBoost", "train_xgboost_optimized"),
        ("Random Forest", "train_randomforest_optimized"),
        ("Decision Tree", "train_decisiontree_optimized"),
        ("EBM", "train_ebm_optimized"),
    ]
    
    results = []
    overall_start = time.time()
    
    # Train each model
    for model_name, script_name in models:
        result = run_model(model_name, script_name)
        results.append(result)
    
    overall_time = time.time() - overall_start
    
    # Summary report
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    print(f"\nTotal training time: {overall_time/60:.1f} minutes")
    
    successful = sum(1 for r in results if r['Status'] == 'SUCCESS')
    failed = len(results) - successful
    
    print(f"\nSuccessful: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(results)}")
        failed_models = [r['Model'] for r in results if r['Status'] == 'FAILED']
        print(f"Failed models: {', '.join(failed_models)}")
    
    # Check saved models
    saved_models_dir = MODELS_DIR / 'saved_models'
    if saved_models_dir.exists():
        saved_files = list(saved_models_dir.glob('*'))
        print(f"\nSaved models directory: {saved_models_dir}")
        print(f"Total saved model files: {len(saved_files)}")
        for f in sorted(saved_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
