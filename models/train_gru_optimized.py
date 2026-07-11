"""
GRU-based Engagement Classification Model
Trains a bidirectional GRU with metadata features for engagement prediction.
Saves the trained model as a .keras file.
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Use Keras with JAX backend
os.environ['KERAS_BACKEND'] = 'jax'
import keras
from keras import layers, models, callbacks

# Set random seeds
np.random.seed(42)
keras.utils.set_random_seed(42)

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


def build_gru_with_meta(input_shape, n_participants):
    """Build bidirectional GRU model with metadata inputs."""
    seq_in = keras.Input(shape=input_shape, name='seq_input')
    pid_in = keras.Input(shape=(1,), dtype='int32', name='pid_input')
    minute_in = keras.Input(shape=(1,), dtype='float32', name='minute_input')
    
    seq_x = layers.Bidirectional(layers.GRU(96, return_sequences=True))(seq_in)
    seq_x = layers.Dropout(0.25)(seq_x)
    seq_x = layers.Bidirectional(layers.GRU(48))(seq_x)
    seq_x = layers.Dropout(0.20)(seq_x)
    
    pid_emb = layers.Embedding(input_dim=n_participants + 1, output_dim=8)(pid_in)
    pid_emb = layers.Flatten()(pid_emb)
    
    meta_x = layers.Concatenate()([pid_emb, minute_in])
    x = layers.Concatenate()([seq_x, meta_x])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=[seq_in, pid_in, minute_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


def main():
    print("="*70)
    print("GRU Engagement Classification Model Training")
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
    
    # Binary classification: Low vs High (remove medium tertile)
    print("\n3. Creating binary labels (Low vs High)...")
    score_rank = pd.Series(scores_all).rank(method='first')
    tertile = pd.qcut(score_rank, q=3, labels=[0, 1, 2]).astype(np.int64).to_numpy()
    keep_mask = tertile != 1
    
    X = X_all[keep_mask]
    y = (tertile[keep_mask] == 2).astype(np.int64)
    meta = meta_all.loc[keep_mask].reset_index(drop=True)
    
    meta['participant'] = meta['participant'].astype(str)
    meta['participant_code'] = pd.Categorical(meta['participant']).codes.astype(np.int32)
    participant_ids = meta['participant_code'].to_numpy(dtype=np.int32)
    minutes = meta['minute'].to_numpy(dtype=np.float32).reshape(-1, 1)
    n_participants = int(meta['participant_code'].nunique())
    
    print(f"   Sample shape: {X.shape}")
    print(f"   Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")
    print(f"   Participants: {n_participants}")
    
    # Train/test split
    print("\n4. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test, pid_train, pid_test, minute_train, minute_test = train_test_split(
        X, y, participant_ids, minutes,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Scaling
    print("\n5. Scaling features...")
    scaler = StandardScaler()
    n_features = X.shape[-1]
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    minute_scaler = StandardScaler()
    minute_train_scaled = minute_scaler.fit_transform(minute_train).astype(np.float32)
    minute_test_scaled = minute_scaler.transform(minute_test).astype(np.float32)
    
    pid_train_in = pid_train.reshape(-1, 1).astype(np.int32)
    pid_test_in = pid_test.reshape(-1, 1).astype(np.int32)
    
    # Compute class weights
    classes = np.array([0, 1], dtype=np.int64)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"   Class weights: {class_weight}")
    
    # Build model
    print("\n6. Building GRU model...")
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = build_gru_with_meta(input_shape, n_participants)
    print(f"   Model parameters: {model.count_params():,}")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    )
    
    # Train
    print("\n7. Training model...")
    train_inputs = [X_train_scaled, pid_train_in, minute_train_scaled]
    test_inputs = [X_test_scaled, pid_test_in, minute_test_scaled]
    
    history = model.fit(
        train_inputs, y_train,
        validation_split=0.2,
        epochs=80,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight,
        verbose=1
    )
    
    # Evaluate
    print("\n8. Evaluating on test set...")
    probs = model.predict(test_inputs, verbose=0).reshape(-1)
    y_pred = (probs >= 0.5).astype(np.int64)
    
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)
    
    print(f"\n   Test Accuracy: {acc:.4f}")
    print(f"   Balanced Accuracy: {bacc:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'High'], digits=4, zero_division=0))
    
    # Save model
    model_path = MODEL_OUTPUT_DIR / 'gru_engagement_optimized.keras'
    print(f"\n9. Saving model to {model_path}...")
    model.save(model_path)
    print("   Model saved successfully!")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()
