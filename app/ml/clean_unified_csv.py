"""
Script pour nettoyer le CSV unifié en ne gardant que les images qui existent réellement
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Configuration
BASE_DIR = Path(r"D:\PERSO\Master-2025\OncoGuardApi")
UNIFIED_CSV = BASE_DIR / "app" / "data" / "unified" / "unified_annotations.csv"
CLEANED_CSV = BASE_DIR / "app" / "data" / "unified" / "unified_annotations_clean.csv"

def main():
    logging.info("=" * 70)
    logging.info("NETTOYAGE DU CSV UNIFIÉ")
    logging.info("=" * 70)
    
    # Charger le CSV
    logging.info(f"Chargement: {UNIFIED_CSV}")
    df = pd.read_csv(UNIFIED_CSV)
    logging.info(f"Nombre total de lignes: {len(df)}")
    
    # Vérifier chaque fichier
    existing = []
    missing = []
    
    logging.info("\nVérification de l'existence des fichiers...")
    for idx, row in df.iterrows():
        filepath = BASE_DIR / row['filepath']
        
        if filepath.exists():
            existing.append(idx)
        else:
            missing.append(idx)
        
        # Afficher la progression tous les 500 fichiers
        if (idx + 1) % 500 == 0:
            logging.info(f"  Vérifié: {idx + 1}/{len(df)} fichiers")
    
    logging.info(f"\nRésultats:")
    logging.info(f"  ✓ Fichiers existants: {len(existing)}")
    logging.info(f"  ✗ Fichiers manquants: {len(missing)}")
    
    # Statistiques par dataset
    df_existing = df.iloc[existing]
    df_missing = df.iloc[missing]
    
    logging.info(f"\nRépartition des fichiers existants:")
    for dataset in df_existing['dataset'].unique():
        count = len(df_existing[df_existing['dataset'] == dataset])
        logging.info(f"  - {dataset}: {count}")
    
    logging.info(f"\nRépartition des fichiers manquants:")
    for dataset in df_missing['dataset'].unique():
        count = len(df_missing[df_missing['dataset'] == dataset])
        logging.info(f"  - {dataset}: {count}")
    
    # Sauvegarder le CSV nettoyé
    df_clean = df.iloc[existing].reset_index(drop=True)
    df_clean.to_csv(CLEANED_CSV, index=False)
    
    logging.info(f"\n✅ CSV nettoyé sauvegardé: {CLEANED_CSV}")
    logging.info(f"Lignes dans le CSV nettoyé: {len(df_clean)}")
    
    logging.info(f"\nRépartition par label dans le CSV nettoyé:")
    for label in df_clean['label'].unique():
        count = len(df_clean[df_clean['label'] == label])
        logging.info(f"  - {label}: {count}")
    
    # Sauvegarder aussi la liste des fichiers manquants
    if missing:
        missing_csv = BASE_DIR / "app" / "data" / "unified" / "missing_files.csv"
        df_missing.to_csv(missing_csv, index=False)
        logging.info(f"\nListe des fichiers manquants: {missing_csv}")
    
    logging.info("\n" + "=" * 70)
    logging.info("RECOMMANDATIONS")
    logging.info("=" * 70)
    
    if len(existing) < len(df) * 0.5:
        logging.warning("⚠️ Plus de 50% des fichiers sont manquants!")
        logging.info("Options:")
        logging.info("  1. Entraîner uniquement sur MIAS (plus rapide)")
        logging.info("     python -m app.ml.train_eval_mias --csv app/data/mias/mias_annotations.csv")
        logging.info("  2. Vérifier que les images CBIS-DDSM sont bien au format JPEG")
    else:
        logging.info("✓ Le CSV nettoyé peut être utilisé pour l'entraînement:")
        logging.info(f"  python -m app.ml.train_eval_mias --csv {CLEANED_CSV.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()