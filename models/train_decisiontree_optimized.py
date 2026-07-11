"""
Decision Tree Engagement Classification Model
Trains an optimized Decision Tree classifier for 3-class engagement prediction (Low/Medium/High).
Saves the trained model as a .pkl file.
"""

import re
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

# Set random seed
np.random.seed(42)

# Configuration
BASE_DIR = Path(__file__).parent.parent
MM_PROCESSED = BASE_DIR / 'MultiModalAttack' / 'Processed Data'
TXT_PROCESSED = BASE_DIR / 'TextualAttack' / 'Processed Data'
MM_SELF_REPORT = BASE_DIR / 'MultiModalAttack' / 'Trust and Engagement Self Report_Multi_Modal.csv'
TXT_SELF_REPORT = BASE_DIR / 'TextualAttack' / 'Trust and Engagement Self Report_TextOnly.csv'
MODEL_OUTPUT_DIR = BASE_DIR / 'models' / 'saved_models'

SEQ_LEN = 60
WINDOW_SECONDS = 120
MIN_ROWS_PER_WINDOW = 10

# Create output directory
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_participant_id(raw):
    """Extract numeric participant ID."""
    if pd.isna(raw):
        return None
    match = re.search(r'(\d+)', str(raw))
    return match.group(1) if match else None


def load_engagement_label_map(self_report_csv):
    """Load engagement labels from self-report CSV."""
    df = pd.read_csv(self_report_csv, skiprows=1)
    
    score_type = df['Score Type'].astype(str).str.strip().str.lower()
    df = df[score_type == 'engagement'].copy()
    if df.empty:
        raise ValueError(f'No engagement rows found in {self_report_csv.name}.')
    
    df['pid'] = df['Participant'].apply(normalize_participant_id)
    
    minute_cols = [c for c in df.columns if re.match(r'^Min\s+\d+$', str(c).strip())]
    minute_cols = sorted(minute_cols, key=lambda c: int(re.search(r'\d+', c).group()))
    
    label_map = {}
    for _, row in df.iterrows():
        pid = row['pid']
        if pid is None:
            continue
        for col in minute_cols:
            val = row[col]
            if pd.isna(val):
                continue
            minute = int(re.search(r'\d+', col).group())
            label_map[(pid, minute)] = float(val)
    
    return label_map


def interpolate_sequence(values_2d, seq_len=60):
    """Interpolate time series to fixed length."""
    t, f = values_2d.shape
    if t == seq_len:
        return values_2d.astype(np.float32)
    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, seq_len)
    out = np.zeros((seq_len, f), dtype=np.float32)
    for j in range(f):
        out[:, j] = np.interp(new_idx, old_idx, values_2d[:, j])
    return out


def build_samples(processed_dir, label_map, file_glob):
    """Build 2-minute window samples from processed data files."""
    samples, scores, meta = [], [], []
    files = sorted(processed_dir.glob(file_glob))
    print(f'Found {len(files)} files in {processed_dir.name} matching {file_glob}')
    
    for fp in files:
        pid_match = re.search(r'(\d+)\.csv$', fp.name)
        if not pid_match:
            continue
        pid = pid_match.group(1)
        
        df = pd.read_csv(fp)
        if 'TimestampRaw' not in df.columns:
            continue
        
        feature_cols = [c for c in df.columns if c != 'TimestampRaw']
        if not feature_cols:
            continue
        
        df = df.sort_values('TimestampRaw').reset_index(drop=True)
        t0 = float(df['TimestampRaw'].iloc[0])
        df['t_rel'] = df['TimestampRaw'] - t0
        max_t = float(df['t_rel'].max())
        
        participant_minutes = sorted([m for (p, m) in label_map.keys() if p == pid])
        
        for minute in participant_minutes:
            end_t = minute * 60
            start_t = end_t - WINDOW_SECONDS
            if end_t > max_t + 1:
                continue
            
            window = df[(df['t_rel'] >= start_t) & (df['t_rel'] < end_t)]
            if len(window) < MIN_ROWS_PER_WINDOW:
                continue
            
            feats = window[feature_cols].to_numpy(dtype=np.float32)
            seq = interpolate_sequence(feats, seq_len=SEQ_LEN)
            score = float(label_map[(pid, minute)])
            
            samples.append(seq)
            scores.append(score)
            meta.append({'participant': pid, 'minute': minute, 'score': score, 'source_file': fp.name})
    
    if not samples:
        return np.empty((0, 0, 0), dtype=np.float32), np.array([], dtype=np.float32), pd.DataFrame(meta)
    
    return np.array(samples, dtype=np.float32), np.array(scores, dtype=np.float32), pd.DataFrame(meta)


def build_prior_table(meta_df, y_arr, key_col, n_classes=3, alpha=1.0):
    """Build class prior probability table from training data."""
    tab = pd.DataFrame({key_col: meta_df[key_col].astype(str).values, 'y': y_arr})
    counts = tab.groupby([key_col, 'y']).size().unstack(fill_value=0)
    for c in range(n_classes):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[[0, 1, 2]]
    probs = (counts + alpha).div((counts.sum(axis=1) + alpha * n_classes), axis=0)
    return probs


def prior_features(meta_df, probs_table, key_col, global_probs):
    """Generate prior probability features for metadata."""
    arr = np.zeros((len(meta_df), 3), dtype=np.float32)
    keys = meta_df[key_col].astype(str).values
    for i, key in enumerate(keys):
        if key in probs_table.index:
            arr[i] = probs_table.loc[key].to_numpy(dtype=np.float32)
        else:
            arr[i] = global_probs
    return arr


def main():
    print("="*70)
    print("Decision Tree 3-Class Engagement Classification Model Training")
    print("="*70)
    
    # Load labels
    print("\n1. Loading engagement labels...")
    mm_label_map = load_engagement_label_map(MM_SELF_REPORT)
    txt_label_map = load_engagement_label_map(TXT_SELF_REPORT)
    print(f"   MultiModal labels: {len(mm_label_map)}")
    print(f"   Text-only labels: {len(txt_label_map)}")
    
    # Build samples
    print("\n2. Building 2-minute window samples...")
    X_mm, score_mm, meta_mm = build_samples(MM_PROCESSED, mm_label_map, 'DI_Engagement_False_Info_*.csv')
    X_txt, score_txt, meta_txt = build_samples(TXT_PROCESSED, txt_label_map, 'DI_Engagement_False_Info_Text_*.csv')
    
    arrays_X = [arr for arr in [X_mm, X_txt] if arr.size > 0]
    arrays_scores = [arr for arr in [score_mm, score_txt] if arr.size > 0]
    
    if len(arrays_X) == 0:
        raise ValueError('No 2-minute samples were generated. Check paths and labels.')
    
    X_all = np.concatenate(arrays_X, axis=0)
    scores_all = np.concatenate(arrays_scores, axis=0).astype(np.float32)
    meta_all = pd.concat([meta_mm.assign(condition='multimodal'), meta_txt.assign(condition='text')], ignore_index=True)
    
    # 3-class labels with smoothing: Low/Medium/High
    print("\n3. Creating 3-class labels (Low/Medium/High) with smoothing...")
    df_hacc = meta_all[['participant', 'minute', 'condition']].copy()
    df_hacc['participant'] = df_hacc['participant'].astype(str)
    df_hacc['score'] = scores_all.astype(np.float32)
    df_hacc['idx'] = np.arange(len(df_hacc))
    
    df_hacc = df_hacc.sort_values(['participant', 'minute']).reset_index(drop=True)
    df_hacc['score_smooth'] = df_hacc.groupby('participant')['score'].transform(
        lambda s: s.rolling(window=3, min_periods=1, center=True).mean()
    )
    
    rank_hacc = df_hacc['score_smooth'].rank(method='first')
    q4_hacc = pd.qcut(rank_hacc, q=4, labels=[0, 1, 2, 3]).astype(np.int64)
    y_hacc_full = np.where(q4_hacc == 0, 0, np.where(q4_hacc == 3, 2, 1)).astype(np.int64)
    
    idx_sorted = df_hacc['idx'].to_numpy(dtype=np.int64)
    X_seq_hacc = X_all[idx_sorted].astype(np.float32)
    
    # Feature engineering
    print("\n4. Engineering features...")
    flat_hacc = X_seq_hacc.reshape(X_seq_hacc.shape[0], -1)
    delta_hacc = np.diff(X_seq_hacc, axis=1, prepend=X_seq_hacc[:, :1, :])
    abs_delta_hacc = np.abs(delta_hacc)
    
    summary_hacc = np.concatenate([
        X_seq_hacc.mean(axis=1),
        X_seq_hacc.std(axis=1),
        X_seq_hacc.min(axis=1),
        X_seq_hacc.max(axis=1),
        (X_seq_hacc[:, -1, :] - X_seq_hacc[:, 0, :]),
        abs_delta_hacc.mean(axis=1),
        abs_delta_hacc.max(axis=1),
    ], axis=1).astype(np.float32)
    
    pid_codes_hacc = pd.Categorical(df_hacc['participant']).codes.astype(np.int32)
    cond_codes_hacc = pd.Categorical(df_hacc['condition']).codes.astype(np.int32)
    pid_oh_hacc = np.eye(int(pid_codes_hacc.max()) + 1, dtype=np.float32)[pid_codes_hacc]
    cond_oh_hacc = np.eye(int(cond_codes_hacc.max()) + 1, dtype=np.float32)[cond_codes_hacc]
    minute_hacc = df_hacc['minute'].to_numpy(dtype=np.float32).reshape(-1, 1)
    
    X_hacc_full = np.hstack([flat_hacc, summary_hacc, pid_oh_hacc, cond_oh_hacc, minute_hacc]).astype(np.float32)
    
    print(f"   Original class counts: {pd.Series(y_hacc_full).value_counts().sort_index().to_dict()}")
    
    # Balance classes
    print("\n5. Balancing classes via downsampling...")
    rng_hacc = np.random.default_rng(42)
    idx_by_cls_hacc = {c: np.where(y_hacc_full == c)[0] for c in [0, 1, 2]}
    min_count_hacc = min(len(idx_by_cls_hacc[0]), len(idx_by_cls_hacc[1]), len(idx_by_cls_hacc[2]))
    sel_hacc = np.concatenate([rng_hacc.choice(idx_by_cls_hacc[c], size=min_count_hacc, replace=False) for c in [0, 1, 2]])
    rng_hacc.shuffle(sel_hacc)
    
    X_hacc_bal = X_hacc_full[sel_hacc]
    y_hacc_bal = y_hacc_full[sel_hacc]
    meta_hacc_bal = df_hacc.iloc[sel_hacc].reset_index(drop=True)
    
    print(f"   Balanced class counts: {pd.Series(y_hacc_bal).value_counts().sort_index().to_dict()}")
    
    # Train/test split
    print("\n6. Splitting data (80/20)...")
    Xh_train, Xh_test, yh_train, yh_test, mh_train, mh_test = train_test_split(
        X_hacc_bal, y_hacc_bal, meta_hacc_bal,
        test_size=0.2,
        random_state=42,
        stratify=y_hacc_bal
    )
    mh_train = mh_train.reset_index(drop=True)
    mh_test = mh_test.reset_index(drop=True)
    
    # Add prior features
    print("\n7. Computing prior features from training data...")
    global_probs_hacc = np.bincount(yh_train, minlength=3).astype(np.float32)
    global_probs_hacc = global_probs_hacc / global_probs_hacc.sum()
    
    pid_probs_hacc = build_prior_table(mh_train, yh_train, key_col='participant', n_classes=3, alpha=1.0)
    cond_probs_hacc = build_prior_table(mh_train, yh_train, key_col='condition', n_classes=3, alpha=1.0)
    
    pid_prior_train = prior_features(mh_train, pid_probs_hacc, 'participant', global_probs_hacc)
    pid_prior_test = prior_features(mh_test, pid_probs_hacc, 'participant', global_probs_hacc)
    cond_prior_train = prior_features(mh_train, cond_probs_hacc, 'condition', global_probs_hacc)
    cond_prior_test = prior_features(mh_test, cond_probs_hacc, 'condition', global_probs_hacc)
    
    Xh_train_aug = np.hstack([Xh_train, pid_prior_train, cond_prior_train])
    Xh_test_aug = np.hstack([Xh_test, pid_prior_test, cond_prior_test])
    
    print(f"   Augmented feature shape: {Xh_train_aug.shape}")
    
    # Hyperparameter tuning
    print("\n8. Hyperparameter tuning with RandomizedSearchCV...")
    param_dist = {
        'max_depth': [4, 5, 6, 8, 10, 14, None],
        'min_samples_split': [2, 5, 10, 20, 30],
        'min_samples_leaf': [1, 2, 4, 8, 12],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'ccp_alpha': [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'class_weight': [None, 'balanced'],
        'splitter': ['best', 'random'],
    }
    
    base_dt = DecisionTreeClassifier(random_state=42)
    
    cv_hacc = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_dt,
        param_distributions=param_dist,
        n_iter=28,
        scoring='balanced_accuracy',
        n_jobs=-1,
        cv=cv_hacc,
        random_state=42,
        verbose=2,
        refit=True
    )
    
    search.fit(Xh_train_aug, yh_train)
    best_model = search.best_estimator_
    
    print(f"\n   Best parameters: {search.best_params_}")
    print(f"   Best CV balanced accuracy: {search.best_score_:.4f}")
    
    # Evaluate
    print("\n9. Evaluating on test set...")
    y_pred = best_model.predict(Xh_test_aug)
    
    acc = accuracy_score(yh_test, y_pred)
    bacc = balanced_accuracy_score(yh_test, y_pred)
    f1m = f1_score(yh_test, y_pred, average='macro', zero_division=0)
    
    print(f"\n   Test Accuracy: {acc:.4f}")
    print(f"   Balanced Accuracy: {bacc:.4f}")
    print(f"   Macro F1: {f1m:.4f}")
    print(f"   >= 68% Accuracy: {acc >= 0.68}")
    print("\n   Classification Report:")
    print(classification_report(yh_test, y_pred, target_names=['Low', 'Medium', 'High'], digits=4, zero_division=0))
    print("\n   Confusion Matrix:")
    print(confusion_matrix(yh_test, y_pred))
    
    # Save model
    model_path = MODEL_OUTPUT_DIR / 'decisiontree_engagement_optimized.pkl'
    print(f"\n10. Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print("   Model saved successfully!")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()
