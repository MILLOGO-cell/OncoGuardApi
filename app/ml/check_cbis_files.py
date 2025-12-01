"""
Script pour diagnostiquer où sont les fichiers CBIS-DDSM
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(r"D:\PERSO\Master-2025\OncoGuardApi")
CBIS_DIR = BASE_DIR / "app" / "data" / "cbis-ddsm"

print("=" * 70)
print("🔍 DIAGNOSTIC CBIS-DDSM")
print("=" * 70)
print()

# Vérifier la structure
print(f"📁 Dossier CBIS-DDSM: {CBIS_DIR}")
print(f"   Existe: {CBIS_DIR.exists()}")
print()

if not CBIS_DIR.exists():
    print("❌ Le dossier CBIS-DDSM n'existe pas!")
    print("   Vous devez télécharger le dataset depuis:")
    print("   https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    exit(1)

# Compter les fichiers par type
print("📊 Analyse des fichiers présents...")
print()

dcm_files = list(CBIS_DIR.rglob("*.dcm"))
jpg_files = list(CBIS_DIR.rglob("*.jpg"))
jpeg_files = list(CBIS_DIR.rglob("*.jpeg"))
png_files = list(CBIS_DIR.rglob("*.png"))

total_images = len(dcm_files) + len(jpg_files) + len(jpeg_files) + len(png_files)

print(f"🖼️  Types de fichiers trouvés:")
print(f"   - Fichiers DICOM (.dcm): {len(dcm_files)}")
print(f"   - Fichiers JPEG (.jpg): {len(jpg_files)}")
print(f"   - Fichiers JPEG (.jpeg): {len(jpeg_files)}")
print(f"   - Fichiers PNG (.png): {len(png_files)}")
print(f"   - TOTAL: {total_images}")
print()

# Vérifier les sous-dossiers
subdirs = [d for d in CBIS_DIR.iterdir() if d.is_dir()]
print(f"📂 Sous-dossiers ({len(subdirs)}):")
for subdir in subdirs[:10]:
    file_count = len(list(subdir.rglob("*.*")))
    print(f"   - {subdir.name}: {file_count} fichiers")
if len(subdirs) > 10:
    print(f"   ... et {len(subdirs) - 10} autres dossiers")
print()

# Recommandations
print("=" * 70)
print("💡 RECOMMANDATIONS")
print("=" * 70)
print()

if len(dcm_files) > 0:
    print(f"✓ Vous avez {len(dcm_files)} fichiers DICOM")
    print("  → Il faut les convertir en JPEG")
    print("  → Utilisez le script: python convert_cbis_dicom_to_jpeg.py")
    print()
elif len(jpg_files) + len(jpeg_files) > 0:
    print(f"✓ Vous avez {len(jpg_files) + len(jpeg_files)} fichiers JPEG")
    print("  → Les images sont prêtes!")
    print("  → Problème possible: les chemins dans le CSV ne correspondent pas")
    print("  → Solution: Reconstruire le CSV avec: python rebuild_cbis_csv.py")
    print()
else:
    print("❌ Aucune image trouvée dans le dossier CBIS-DDSM!")
    print()
    print("Actions nécessaires:")
    print("1. Téléchargez le dataset CBIS-DDSM complet depuis:")
    print("   https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print()
    print("2. Extraire dans: app/data/cbis-ddsm/")
    print()
    print("3. Structure attendue:")
    print("   app/data/cbis-ddsm/")
    print("   ├── csv/")
    print("   │   ├── mass_case_description_train_set.csv")
    print("   │   └── calc_case_description_train_set.csv")
    print("   └── jpeg/ (ou dossier avec images)")
    print()

print("=" * 70)