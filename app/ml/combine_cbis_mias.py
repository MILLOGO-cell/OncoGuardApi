"""
Script pour combiner CBIS-DDSM (2 classes) + MIAS (3 classes)
Résultat : Dataset unifié avec normal, benign, malignant

CBIS-DDSM : 3568 images (benign, malignant)
MIAS      : 330 images (normal, benign, malignant)
TOTAL     : 3898 images (3 classes)
"""
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(r"D:\PERSO\Master-2025\OncoGuardApi")
CBIS_CSV = BASE_DIR / "app" / "data" / "cbis-ddsm" / "cbis_final.csv"
MIAS_CSV = BASE_DIR / "app" / "data" / "mias" / "mias_annotations.csv"
OUTPUT_CSV = BASE_DIR / "app" / "data" / "unified_cbis_mias.csv"

# Dossiers des images
CBIS_IMG_DIR = BASE_DIR / "app" / "data" / "cbis-ddsm" / "jpeg"
MIAS_IMG_DIR = BASE_DIR / "app" / "data" / "mias"


def normalize_label(label: str) -> str:
    """Normalise les labels en minuscule."""
    lab = str(label).strip().lower()
    
    # Mapper les variantes
    if lab in {"normal", "norm"}:
        return "normal"
    elif lab in {"benign", "ben", "b"}:
        return "benign"
    elif lab in {"malignant", "malin", "mal", "m"}:
        return "malignant"
    else:
        return lab


def check_file_exists(filepath: Path, base_dir: Path) -> bool:
    """Vérifie si un fichier existe."""
    if filepath.is_absolute() and filepath.exists():
        return True
    
    full_path = base_dir / filepath
    return full_path.exists()


def main():
    print("=" * 70)
    print("🔗 COMBINAISON CBIS-DDSM + MIAS")
    print("=" * 70)
    print()
    
    # 1. Charger CBIS-DDSM
    print("📂 Chargement CBIS-DDSM...")
    if not CBIS_CSV.exists():
        print(f"❌ CSV CBIS introuvable: {CBIS_CSV}")
        print("   Exécutez d'abord: python -m app.ml.build_cbis_final")
        return
    
    df_cbis = pd.read_csv(CBIS_CSV)
    print(f"   ✓ CBIS-DDSM: {len(df_cbis)} images")
    
    # Vérifier les colonnes
    if 'filepath' not in df_cbis.columns or 'label' not in df_cbis.columns:
        print(f"❌ Colonnes manquantes dans CBIS CSV")
        print(f"   Colonnes présentes: {list(df_cbis.columns)}")
        return
    
    # Normaliser les labels CBIS
    df_cbis['label'] = df_cbis['label'].apply(normalize_label)
    df_cbis['dataset'] = 'cbis'
    
    print(f"   Distribution CBIS:")
    for label, count in df_cbis['label'].value_counts().items():
        print(f"      - {label}: {count}")
    
    # 2. Charger MIAS
    print("\n📂 Chargement MIAS...")
    if not MIAS_CSV.exists():
        print(f"❌ CSV MIAS introuvable: {MIAS_CSV}")
        print(f"   Créez-le d'abord ou placez mias_annotations.csv dans:")
        print(f"   {MIAS_CSV.parent}")
        return
    
    df_mias = pd.read_csv(MIAS_CSV)
    print(f"   ✓ MIAS: {len(df_mias)} images")
    
    # Vérifier les colonnes
    if 'filename' not in df_mias.columns or 'label' not in df_mias.columns:
        print(f"❌ Colonnes manquantes dans MIAS CSV")
        print(f"   Colonnes présentes: {list(df_mias.columns)}")
        return
    
    # Normaliser les labels MIAS
    df_mias['label'] = df_mias['label'].apply(normalize_label)
    
    # Construire les chemins relatifs pour MIAS
    df_mias['filepath'] = df_mias['filename'].apply(
        lambda x: f"app/data/mias/{x}"
    )
    df_mias['dataset'] = 'mias'
    
    # Garder les bonnes colonnes
    df_mias = df_mias[['filepath', 'label', 'dataset']]
    
    print(f"   Distribution MIAS:")
    for label, count in df_mias['label'].value_counts().items():
        print(f"      - {label}: {count}")
    
    # 3. Vérifier l'existence des fichiers
    print("\n🔍 Vérification de l'existence des fichiers...")
    
    # CBIS
    cbis_missing = 0
    for idx, row in df_cbis.iterrows():
        fp = BASE_DIR / row['filepath']
        if not fp.exists():
            cbis_missing += 1
            if cbis_missing <= 3:
                print(f"   ⚠️  CBIS manquant: {row['filepath']}")
    
    if cbis_missing > 0:
        print(f"   ⚠️  CBIS: {cbis_missing}/{len(df_cbis)} fichiers manquants")
    else:
        print(f"   ✓ CBIS: Tous les fichiers présents")
    
    # MIAS
    mias_missing = 0
    for idx, row in df_mias.iterrows():
        fp = BASE_DIR / row['filepath']
        if not fp.exists():
            mias_missing += 1
            if mias_missing <= 3:
                print(f"   ⚠️  MIAS manquant: {row['filepath']}")
    
    if mias_missing > 0:
        print(f"   ⚠️  MIAS: {mias_missing}/{len(df_mias)} fichiers manquants")
    else:
        print(f"   ✓ MIAS: Tous les fichiers présents")
    
    # 4. Combiner les datasets
    print("\n🔗 Combinaison des datasets...")
    df_combined = pd.concat([df_cbis, df_mias], ignore_index=True)
    
    # 5. Statistiques finales
    print("\n" + "=" * 70)
    print("📊 DATASET UNIFIÉ - STATISTIQUES")
    print("=" * 70)
    print(f"\nTotal images: {len(df_combined)}")
    print(f"  - CBIS-DDSM: {len(df_cbis)} images")
    print(f"  - MIAS:      {len(df_mias)} images")
    print()
    
    print("Distribution par label:")
    for label in sorted(df_combined['label'].unique()):
        count = len(df_combined[df_combined['label'] == label])
        percentage = count / len(df_combined) * 100
        
        # Détail par dataset
        cbis_count = len(df_cbis[df_cbis['label'] == label])
        mias_count = len(df_mias[df_mias['label'] == label])
        
        print(f"  {label.capitalize():12} : {count:4} ({percentage:5.1f}%) "
              f"[CBIS: {cbis_count:4}, MIAS: {mias_count:3}]")
    print()
    
    print("Distribution par dataset:")
    for dataset in df_combined['dataset'].unique():
        count = len(df_combined[df_combined['dataset'] == dataset])
        percentage = count / len(df_combined) * 100
        print(f"  {dataset.upper():8} : {count:4} ({percentage:5.1f}%)")
    print()
    
    # 6. Sauvegarder
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV unifié sauvegardé:")
    print(f"   {OUTPUT_CSV.relative_to(BASE_DIR)}")
    print()
    
    # 7. Commandes d'entraînement
    print("=" * 70)
    print("🚀 ENTRAÎNEMENT SUR LE DATASET UNIFIÉ (3 CLASSES)")
    print("=" * 70)
    print()
    
    print("Commande de base (train/test 80/20):")
    print(f"python -m app.ml.train_eval_mias \\")
    print(f"    --csv {OUTPUT_CSV.relative_to(BASE_DIR)} \\")
    print(f"    --resize 160")
    print()
    
    print("Avec validation croisée (RECOMMANDÉ):")
    print(f"python -m app.ml.train_eval_mias \\")
    print(f"    --csv {OUTPUT_CSV.relative_to(BASE_DIR)} \\")
    print(f"    --cv --n-folds 5 \\")
    print(f"    --resize 160")
    print()
    
    # Estimation du temps
    total_images = len(df_combined)
    estimated_time_min = total_images * 2 // 60  # ~2 sec par image
    estimated_time_max = total_images * 3 // 60
    
    print(f"⏱️  Temps estimé (validation croisée 5-fold):")
    print(f"   {estimated_time_min} - {estimated_time_max} minutes")
    print()
    
    # Résolution recommandée
    print("💡 Résolutions recommandées:")
    print("   --resize 160  : Rapide, bonnes performances (par défaut)")
    print("   --resize 224  : Plus de détails, temps x2")
    print("   --resize 256  : Maximum de détails, temps x3")
    print()
    
    print("=" * 70)
    print("✅ Prêt pour l'entraînement sur 3 CLASSES !")
    print("=" * 70)


if __name__ == "__main__":
    main()