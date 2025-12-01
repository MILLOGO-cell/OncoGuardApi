"""
Script de vérification rapide pour voir la structure des CSV
"""
import pandas as pd
from pathlib import Path

# Chemins corrigés
BASE_DIR = Path(r"D:\PERSO\Master-2025\OncoGuardApi")
MIAS_DIR = BASE_DIR / "app" / "data" / "mias"
CBIS_CSV_DIR = BASE_DIR / "app" / "data" / "cbis-ddsm" / "csv"

print("=" * 70)
print("🔍 VÉRIFICATION DES CSV")
print("=" * 70)

# Vérifier MIAS
print("\n📋 MIAS Dataset:")
print(f"Dossier: {MIAS_DIR}")
print(f"Existe: {MIAS_DIR.exists()}")

if MIAS_DIR.exists():
    csv_files = list(MIAS_DIR.glob("*.csv"))
    print(f"Fichiers CSV trouvés: {len(csv_files)}")
    
    if csv_files:
        for csv_file in csv_files:
            print(f"\n  📄 Fichier: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                print(f"     Nombre de lignes: {len(df)}")
                print(f"     Colonnes: {list(df.columns)}")
                print(f"\n     Aperçu des premières lignes:")
                print(df.head(3).to_string())
            except Exception as e:
                print(f"     ❌ Erreur: {e}")
    else:
        print("  ⚠️ Aucun fichier CSV trouvé!")

# Vérifier CBIS-DDSM
print("\n" + "=" * 70)
print("📋 CBIS-DDSM Dataset:")
print(f"Dossier: {CBIS_CSV_DIR}")
print(f"Existe: {CBIS_CSV_DIR.exists()}")

if CBIS_CSV_DIR.exists():
    patterns = [
        "*mass_case_description*set*.csv",
        "*calc_case_description*set*.csv",
    ]
    
    all_csv = []
    for pattern in patterns:
        all_csv.extend(list(CBIS_CSV_DIR.glob(pattern)))
    
    print(f"Fichiers CSV trouvés: {len(all_csv)}")
    
    if all_csv:
        for csv_file in all_csv:
            print(f"\n  📄 Fichier: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                print(f"     Nombre de lignes: {len(df)}")
                print(f"     Colonnes: {list(df.columns)[:10]}...")  # Premiers 10
                
                # Vérifier les colonnes importantes
                if 'pathology' in df.columns:
                    print(f"     ✓ Colonne 'pathology' trouvée")
                    print(f"       Valeurs: {df['pathology'].unique()}")
                
                img_cols = [c for c in df.columns if 'image' in c.lower() and 'path' in c.lower()]
                if img_cols:
                    print(f"     ✓ Colonne(s) d'image: {img_cols}")
                
            except Exception as e:
                print(f"     ❌ Erreur: {e}")
    else:
        print("  ⚠️ Aucun fichier CSV trouvé!")
        print(f"\n  Fichiers disponibles dans le dossier:")
        for f in CBIS_CSV_DIR.glob("*"):
            print(f"    - {f.name}")

print("\n" + "=" * 70)
print("✅ Vérification terminée")
print("=" * 70)