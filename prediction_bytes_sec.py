"""
Predykcja bytes_sec (przepustowości) dla danych FCC Measuring Broadband America
Modele: KNN, Random Forest, Gradient Boosting, MLP
Dane: curr_lct_ul (upload) - pooling wszystkich unit_id
Dataset: data-raw-2023-feb
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ── Konfiguracja ────────────────────────────────────────────────────────────

DATA_DIR = ".\\Dane\\dane_2023"   # ścieżka do rozpakowanego archiwum 2023

TARGET = "bytes_sec"
RANDOM_STATE = 67
TEST_SIZE = 0.2

# ── Wczytywanie danych ───────────────────────────────────────────────────────

def load_files(pattern: str) -> pd.DataFrame:
    """Wczytuje wszystkie pliki CSV pasujące do wzorca i łączy w jeden DataFrame."""
    files = glob.glob(os.path.join(DATA_DIR, pattern))
    if not files:
        raise FileNotFoundError(f"Brak plików pasujących do: {os.path.join(DATA_DIR, pattern)}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  Pominięto {f}: {e}")
    return pd.concat(dfs, ignore_index=True)


def load_dataset() -> pd.DataFrame:
    """Zwraca df_ul (dane z testów uploadu)."""
    print(f"Wczytywanie danych z: {DATA_DIR}")
    df_ul = load_files("curr_lct_ul*")
    print(f"  Upload: {df_ul.shape}")
    return df_ul


# ── Przygotowanie cech ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, le_target=None, le_error=None, fit=True) -> tuple:
    """
    Inżynieria cech dla modeli predykcji bytes_sec.
    
    Cechy numeryczne:
        packets_received, packets_sent, packet_size, bytes_total,
        duration, successes, failures
    
    Cechy czasowe:
        hour, day_of_week (z dtime)
    
    Cechy kategoryczne (label encoded):
        unit_id, target (serwer), error_code
    
    Zwraca: X, y, le_target, le_error
    """
    df = df.copy()

    # Usunięcie wierszy bez celu
    df = df.dropna(subset=[TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET] > 0].copy()

    # Cechy czasowe
    df["dtime"] = pd.to_datetime(df["dtime"], errors="coerce")
    df["hour"] = df["dtime"].dt.hour.fillna(-1).astype(int)
    df["day_of_week"] = df["dtime"].dt.dayofweek.fillna(-1).astype(int)

    # Kodowanie kategorii
    if fit:
        le_target = LabelEncoder()
        le_error = LabelEncoder()
        df["target_enc"] = le_target.fit_transform(df["target"].astype(str))
        df["error_enc"] = le_error.fit_transform(df["error_code"].astype(str))
    else:
        df["target_enc"] = df["target"].astype(str).map(
            lambda x: le_target.transform([x])[0] if x in le_target.classes_ else -1
        )
        df["error_enc"] = df["error_code"].astype(str).map(
            lambda x: le_error.transform([x])[0] if x in le_error.classes_ else -1
        )

    # unit_id jako cecha numeryczna (pooling)
    df["unit_id"] = pd.to_numeric(df["unit_id"], errors="coerce").fillna(-1).astype(int)

    feature_cols = [
        "unit_id",
        "packets_received", "packets_sent", "packet_size",
        "bytes_total", "duration",
        "successes", "failures",
        "hour", "day_of_week",
        "target_enc", "error_enc",
    ]

    # Konwersja na numeryczne
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[feature_cols].values
    y = df[TARGET].values

    return X, y, le_target, le_error, feature_cols


# ── Definicje modeli ─────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Zwraca słownik modeli opakowanych w Pipeline ze StandardScaler.
    
    KNN            - baseline nieparametryczny; wymaga skalowania; wolny przy dużych danych
    Random Forest  - ensemble drzew; odporny na outliery; brak potrzeby skalowania (ale nie zaszkodzi)
    Gradient Boost - sekwencyjny boosting; często najlepszy dla danych tabelarycznych
    MLP            - sieć neuronowa (gęsta); wymaga skalowania; może uchwycić nieliniowości


    """

    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=0.005,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ── Ewaluacja ────────────────────────────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    """Trenuje model i zwraca słownik metryk."""
    print(f"  Trenuję: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}


def run_experiment(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Przeprowadza pełny eksperyment (przygotowanie danych + trening + ewaluacja)."""
    print(f"\n{'='*60}")
    print(f"Eksperyment: {label}")
    print(f"{'='*60}")

    X, y, le_target, le_error, feature_names = prepare_features(df)
    print(f"  Próbki: {X.shape[0]:,}  |  Cechy: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    models = build_models()
    for name, model in models.items():
        res = evaluate(model, X_train, X_test, y_train, y_test, name)
        results.append(res)
        
        # --- Szukanie istotności cech ---
        inner_model = model.named_steps["model"]
        
        if hasattr(inner_model, "feature_importances_"):
            print(f"  Wyliczanie istotności cech (feature_importances_) dla {name}...")
            importances = inner_model.feature_importances_
        else:
            print(f"  Wyliczanie istotności cech (Permutation Importance) dla {name}...")
            imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
            importances = imp.importances_mean
            
        imp_df = pd.DataFrame({
            "Cecha": feature_names,
            "Istotność": importances
        }).sort_values(by="Istotność", ascending=False)
        
        print("\n  TOP 10 Najważniejszych cech:")
        print("  " + imp_df.head(10).to_string(index=False).replace("\n", "\n  "))
        print("  " + "-" * 40)

    results_df = pd.DataFrame(results).set_index("Model")
    print(f"\nWyniki [{label}]:")
    print(results_df.round(4).to_string())
    return results_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ul = load_dataset()

    results_ul = run_experiment(ul, "Upload 2023")

    print("\n\n" + "="*60)
    print("ZESTAWIENIE")
    print("="*60)
    print(results_ul.round(4).to_string())

    results_ul.to_csv("wyniki_predykcji.csv")
    print("\nWyniki zapisano do: wyniki_predykcji.csv")


if __name__ == "__main__":
    main()
