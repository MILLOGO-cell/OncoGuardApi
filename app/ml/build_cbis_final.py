"""
SOLUTION FINALE - Créer un CSV CBIS-DDSM fonctionnel
Match les CSV avec les JPEG en utilisant les identifiants DICOM
"""
from pathlib import Path
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(r"D:\PERSO\Master-2025\OncoGuardApi")
CBIS_CSV_DIR = BASE_DIR / "app" / "data" / "cbis-ddsm" / "csv"
CBIS_JPEG_DIR = BASE_DIR / "app" / "data" / "cbis-ddsm" / "jpeg"
OUTPUT_CSV = BASE_DIR / "app" / "data" / "cbis-ddsm" / "cbis_final.csv"

def normalize_label(value: str) -> str:
    """Normalise les labels"""
    v = str(value).strip().upper()
    if v == "MALIGNANT":
        return "malignant"
    if v in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return "benign"
    return ""

def extract_dicom_ids(path_str):
    """Extrait TOUS les identifiants DICOM du chemin"""
    pattern = r'1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+'
    return re.findall(pattern, str(path_str))

def main():
    print("=" * 70)
    print("🚀 CRÉATION CSV CBIS-DDSM FINAL")
    print("=" * 70)
    print()
    
    # ÉTAPE 1: Indexer TOUS les JPEG par leur identifiant DICOM
    print("📇 Étape 1/3 : Indexation des JPEG...")
    jpeg_by_dicom = {}
    all_jpegs = list(CBIS_JPEG_DIR.rglob("*.jpg"))
    
    for jpeg_path in all_jpegs:
        # Le nom du dossier parent = identifiant DICOM
        dicom_id = jpeg_path.parent.name
        if dicom_id not in jpeg_by_dicom:
            jpeg_by_dicom[dicom_id] = []
        jpeg_by_dicom[dicom_id].append(jpeg_path)
    
    print(f"   ✓ {len(all_jpegs)} fichiers JPEG")
    print(f"   ✓ {len(jpeg_by_dicom)} identifiants DICOM uniques")
    print()
    
    # ÉTAPE 2: Charger et combiner les CSV
    print("📄 Étape 2/3 : Chargement des CSV...")
    csv_files = []
    csv_files.extend(sorted(CBIS_CSV_DIR.glob("*mass*.csv")))
    csv_files.extend(sorted(CBIS_CSV_DIR.glob("*calc*.csv")))
    
    frames = []
    for csv_file in csv_files:
        if 'meta' in csv_file.name.lower():
            continue
        try:
            df = pd.read_csv(csv_file)
            frames.append(df)
            print(f"   ✓ {csv_file.name}: {len(df)} lignes")
        except Exception as e:
            print(f"   ✗ {csv_file.name}: {e}")
    
    df_all = pd.concat(frames, ignore_index=True)
    print(f"   ✓ Total: {len(df_all)} annotations")
    print()
    
    # ÉTAPE 3: Matching intelligent
    print("🔗 Étape 3/3 : Matching CSV ↔ JPEG...")
    
    valid_rows = []
    matched = 0
    
    for idx, row in df_all.iterrows():
        # Vérifier le label
        if pd.isna(row.get('pathology')):
            continue
        
        label = normalize_label(row['pathology'])
        if not label:
            continue
        
        # Essayer de matcher avec TOUTES les colonnes de chemins
        path_columns = ['image file path', 'cropped image file path', 'ROI mask file path']
        jpeg_found = None
        
        for col in path_columns:
            if col not in row or pd.isna(row[col]):
                continue
            
            # Extraire les identifiants DICOM du chemin CSV
            dicom_ids = extract_dicom_ids(row[col])
            
            # Chercher une correspondance
            for dicom_id in dicom_ids:
                if dicom_id in jpeg_by_dicom:
                    # Prendre le premier JPEG de ce dossier
                    jpeg_found = jpeg_by_dicom[dicom_id][0]
                    break
            
            if jpeg_found:
                break
        
        if jpeg_found:
            matched += 1
            rel_path = jpeg_found.relative_to(BASE_DIR)
            valid_rows.append({
                'filepath': str(rel_path).replace("\\", "/"),
                'label': label,
                'dataset': 'cbis'
            })
        
        # Progression
        if (idx + 1) % 500 == 0:
            percentage = (matched / (idx + 1)) * 100
            print(f"   Progression: {idx + 1}/{len(df_all)} | Matchés: {matched} ({percentage:.1f}%)")
    
    print()
    print(f"✓ Matching terminé: {matched}/{len(df_all)} correspondances")
    print()
    
    if matched == 0:
        print("❌ Aucune correspondance trouvée!")
        print()
        print("🔍 Debug:")
        print(f"   Exemple chemin CSV: {df_all['image file path'].iloc[0][:100]}...")
        print(f"   Exemple dossier JPEG: {list(jpeg_by_dicom.keys())[0]}")
        print()
        print("Les structures ne correspondent pas du tout.")
        print("Vous devez télécharger le dataset CBIS-DDSM compatible depuis:")
        print("https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        return
    
    # Créer le CSV final
    df_final = pd.DataFrame(valid_rows)
    
    print("=" * 70)
    print("📊 RÉSULTATS")
    print("=" * 70)
    print(f"Annotations dans CSV: {len(df_all)}")
    print(f"Correspondances trouvées: {len(df_final)}")
    print(f"Taux de matching: {len(df_final)/len(df_all)*100:.1f}%")
    print()
    
    print("Répartition par label:")
    for label in df_final['label'].unique():
        count = len(df_final[df_final['label'] == label])
        pct = count / len(df_final) * 100
        print(f"   - {label}: {count} ({pct:.1f}%)")
    print()
    
    # Sauvegarder
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV sauvegardé: {OUTPUT_CSV.relative_to(BASE_DIR)}")
    print()
    
    # Commandes d'entraînement
    print("=" * 70)
    print("🎯 ENTRAÎNEMENT SUR CBIS-DDSM")
    print("=" * 70)
    print()
    print("Commande de base:")
    print(f"python -m app.ml.train_eval_mias --csv {OUTPUT_CSV.relative_to(BASE_DIR)}")
    print()
    print("Avec validation croisée (RECOMMANDÉ):")
    print(f"python -m app.ml.train_eval_mias --csv {OUTPUT_CSV.relative_to(BASE_DIR)} --cv --n-folds 5 --resize 160")
    print()
    print(f"⏱️  Temps estimé: {len(df_final) * 2 // 60} - {len(df_final) * 3 // 60} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()