import os
import csv
from app.core.config import MIAS_DATA_DIR

def parse_mias_annotations():
    input_txt_path = os.path.join(MIAS_DATA_DIR, "Info.txt")  # ou "info.txt"
    output_csv_path = os.path.join(MIAS_DATA_DIR, "mias_annotations.csv")

    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"Le fichier Info.txt est introuvable dans : {MIAS_DATA_DIR}")

    with open(input_txt_path, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['filename', 'label'])

        for line in infile:
            line = line.strip()
            if not line or not line.lower().startswith("mdb"):
                continue

            parts = line.split()
            filename = parts[0] + ".pgm"

            # Cas normal
            if len(parts) > 2 and parts[2].upper() == "NORM":
                label = "Normal"
            else:
                severity = (parts[3] if len(parts) > 3 else "").upper()
                if severity == "B":
                    label = "Benign"
                elif severity == "M":
                    label = "Malignant"
                else:
                    label = "Unknown"

            writer.writerow([filename, label])

    print(f"[✅] Parsing terminé. Fichier CSV créé : {output_csv_path}")


if __name__ == "__main__":
    parse_mias_annotations()
