import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- KONFIGURACJA ---
FILES = {
    "speed": "curr_lct_dl.csv",
    "latency": "curr_udplatency.csv",
    "jitter": "curr_udpjitter.csv",
    "loss": "curr_udpcloss.csv"
}
TARGET = "bytes_sec"
FEATURES = [
    'rtt_avg', 'rtt_std',           # z udplatency
    'jitter_down', 'jitter_up',     # z udpjitter
    'loss_packets',                 # z udpcloss
    'hour', 'day_of_week'           # cechy czasowe
]

def log_progress(step, duration=None):
    ts = time.strftime("%H:%M:%S")
    msg = f"[{ts}] {step}"
    if duration: msg += f" (zajęło {duration:.2f}s)"
    print(msg)
    sys.stdout.flush()

def load_and_sync_data():
    start_time = time.time()
    log_progress("Rozpoczynanie procesu wczytywania i synchronizacji danych...")

    # Sprawdzenie plików
    for name, path in FILES.items():
        if not Path(path).exists():
            print(f"BŁĄD: Brak pliku {path}!")
            sys.exit(1)

    dfs = {k: pd.read_csv(v, low_memory=False) for k, v in FILES.items()}

    def prepare_df(df):
        df['dtime'] = pd.to_datetime(df['dtime'], errors='coerce')
        df = df.dropna(subset=['dtime']).copy()
        df = df.sort_values(by='dtime')
        df = df.drop_duplicates(subset=['dtime', 'unit_id'])
        return df.reset_index(drop=True)

    df_speed = prepare_df(dfs["speed"])
    df_lat = prepare_df(dfs["latency"])
    df_jit = prepare_df(dfs["jitter"])
    df_loss = prepare_df(dfs["loss"])

    log_progress("Dane przygotowane i odśmiecone. Łączenie kaskadowe...")

    try:
        merged = pd.merge_asof(
            df_speed, 
            df_lat[['unit_id', 'dtime', 'rtt_avg', 'rtt_std']], 
            on='dtime', by='unit_id', direction='nearest'
        )
        merged = merged.sort_values('dtime')
        merged = pd.merge_asof(
            merged, 
            df_jit[['unit_id', 'dtime', 'jitter_down', 'jitter_up']], 
            on='dtime', by='unit_id', direction='nearest'
        )
        merged = merged.sort_values('dtime')
        df_loss_clean = df_loss.rename(columns={'packets': 'loss_packets'})
        merged = pd.merge_asof(
            merged, 
            df_loss_clean[['unit_id', 'dtime', 'loss_packets']], 
            on='dtime', by='unit_id', direction='nearest'
        )
    except Exception as e:
        print(f"Błąd łączenia: {e}")
        sys.exit(1)

    merged['hour'] = merged['dtime'].dt.hour
    merged['day_of_week'] = merged['dtime'].dt.dayofweek
    
    final_df = merged.dropna(subset=[TARGET, 'rtt_avg', 'jitter_down']).copy()
    final_df['loss_packets'] = final_df['loss_packets'].fillna(0)

    log_progress(f"Synchronizacja zakończona. Rekordy: {len(final_df)}", time.time() - start_time)
    return final_df[FEATURES], final_df[TARGET]

def main():
    X, y = load_and_sync_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definicja preprocesora
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), FEATURES)
    ])

    # Słownik modeli do porównania
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    all_results = []

    print("\n" + "="*50)
    print("   ROZPOCZYNANIE ANALIZY PORÓWNAWCZEJ MODELI")
    print("="*50)

    for name, model_obj in models.items():
        log_progress(f"Trenowanie modelu: {name}...")
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_obj)
        ])
        
        start_m = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_m
        
        y_pred = pipeline.predict(X_test)
        
        # Metryki
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        all_results.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Train Time [s]": round(train_time, 2)
        })

    # Prezentacja wyników
    results_df = pd.DataFrame(all_results).sort_values(by="R2 Score", ascending=False)
    
    print("\n--- PORÓWNANIE WYDAJNOŚCI ALGORYTMÓW ---")
    print(results_df.to_string(index=False))
    
    print("\n" + "="*50)
    best_m = results_df.iloc[0]['Model']
    print(f" NAJLEPSZY MODEL: {best_m}")
    print("="*50)

    # Istotność cech dla najlepszego modelu (jeśli to drzewo)
    # Dla uproszczenia wyświetlamy dla Random Forest, bo zawsze jest w zestawie
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models["Random Forest"])
    ])
    rf_pipeline.fit(X_train, y_train) # Upewniamy się, że wytrenowany
    
    importances = rf_pipeline.named_steps['model'].feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
    print("\nISTOTNOŚĆ CECH (wg Random Forest):")
    print(feat_imp)

if __name__ == "__main__":
    main()