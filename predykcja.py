import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

# --- KONFIGURACJA ---
FILE_NAME = "./Dane/dane_2023/curr_httpgetmt.csv"
TARGET = "bytes_sec"
# Skupiamy się na czasie i identyfikacji urządzenia
FEATURES = ['unit_id', 'hour_sin', 'hour_cos', 'day_of_week']

def prepare_data(file_path):
    print(f" Wczytywanie danych z {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. Konwersja czasu
    df['dtime'] = pd.to_datetime(df['dtime'], errors='coerce')
    df = df.dropna(subset=['dtime', TARGET])
    
    # 2. Inżynieria cech czasowych
    df['hour'] = df['dtime'].dt.hour
    df['day_of_week'] = df['dtime'].dt.dayofweek
    
    # Transformacja cykliczna godziny (żeby 23:00 i 00:00 były blisko siebie)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Opcjonalnie: Filtrowanie błędów (tylko udane testy)
    if 'error_code' in df.columns:
        df = df[df['error_code'] == 'NO_ERROR']
        
    return df[FEATURES], df[TARGET]

def main():
    # Pobranie danych
    X, y = prepare_data(FILE_NAME)
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=67)
    
    # Skalowanie cech (niezbędne dla modelu KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definicja 3 różnych modeli (zgodnie z pkt 5 instrukcji)
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=130, max_depth=45, n_jobs=-1, random_state=67),
        # Dla HistGradientBoosting zamiast n_estimators używamy max_iter.

        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=600,
        ),
        "K-Neighbors (KNN)": KNeighborsRegressor(n_neighbors=30, weights='distance')
    }
    
    print("\n" + "="*50)
    print(f"PORÓWNANIE MODELI (Rekordów: {len(X)})")
    print("="*50)
    
    results = []
    for name, model in models.items():
        start = time.time()
        # Trenowanie
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start
        
        # Predykcja
        y_pred = model.predict(X_test_scaled)
        
        # Metryki
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        
        results.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "MAE [Mbps]": round(mae / 125000, 2), # Konwersja z bytes_sec na Megabity dla czytelności
            "RMSE [Mbps]": round(rmse / 125000, 2),
            "Czas [s]": round(train_time, 2)
        })
        print("Zakończono model:", name)
    #Zapis wyników do txt
    with open("model_results.txt", "w") as f:
        f.write("POROWNANIE MODELI (Rekordow: {})\n".format(len(X)))
        f.write("="*50 + "\n")
        for res in results:
            f.write(f"{res['Model']}: R2={res['R2 Score']}, MAE={res['MAE [Mbps]']} Mbps, RMSE={res['RMSE [Mbps]']} Mbps, Czas={res['Czas [s]']} s\n")

        rf_model = models["Random Forest"]
        importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        f.write("\nISTOTNOSC CECH (Random Forest):\n")
        for feature, importance in importances.items():
            f.write(f"{feature}: {importance:.6f}\n")
    # Wyświetlenie tabeli wyników (spełnia pkt 6 instrukcji)
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print(results_df.to_string(index=False))
    
    # Analiza istotności cech dla Random Forest
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nISTOTNOŚĆ CECH (Random Forest):")
    print(importances)

if __name__ == "__main__":
    main()