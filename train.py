from pathlib import Path
import pandas as pd

from src.features import engineer_features
from src.model import train

CSV_PATH=Path("C:/Users/adith/Documents/PLANET PROJECT!!!!/PSCompPars_2025.11.16_08.45.44.csv")
OUT_CSV = Path("data/exoplanets_engineered.csv")

def main():
    print("Loading NASA Exoplanet data...")
    df=pd.read_csv(CSV_PATH, comment="#", low_memory=False)
    print(f"  {len(df)} planets loaded, {len(df.columns)} columns")
    
    print("Engineering features...")
    df = engineer_features(df)
    print(f"  ESI range: {df['esi'].min():.3f} - {df['esi'].max():.3f}")
    print(f"  Planets with ESI > 0.5: {(df['esi'] > 0.5).sum()}")
    print(f"  Planets in habitable zone: {int(df['in_hz'].sum())}")
    
    print("Training ensemble model (this takes 1-2 minutes)...")
    model = train(df)
    print("  Saved to models/ensemble.joblib")
    
    print("Saving engineered dataset...")
    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved to {OUT_CSV}")
    
    print("\nTop 10 most Earth-like planets:")
    cols = ["pl_name", "hostname", "esi", "pl_eqt", "pl_rade", "pl_bmasse", "in_hz"]
    top = df.nlargest(10, "esi")[cols].round(3)
    print(top.to_string(index=False))

    print("\nDone! Run: streamlit run app.py")
    
if __name__ == "__main__":
    main()
