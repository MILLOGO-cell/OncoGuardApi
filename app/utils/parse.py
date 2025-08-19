import csv

def parse_mias_annotations(input_txt_path: str, output_csv_path: str):
    with open(input_txt_path, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['filename', 'label'])
        
        for line in infile:
            line = line.strip()
            # Filtrer les lignes utiles qui commencent par "mdb" (nom image)
            if line.startswith("mdb"):
                parts = line.split()
                filename = parts[0] + ".pgm"
                
                # Gérer le cas Normal : colonne 3 = 'NORM'
                if len(parts) > 2 and parts[2] == "NORM":
                    label = "Normal"
                else:
                    # Gravité de l’anomalie (colonne 4) B=Benign, M=Malignant
                    severity = parts[3] if len(parts) > 3 else ""
                    if severity == "B":
                        label = "Benign"
                    elif severity == "M":
                        label = "Malignant"
                    else:
                        label = "Unknown"
                
                writer.writerow([filename, label])

    print(f"Parsing terminé. Fichier CSV sauvegardé : {output_csv_path}")

# Exemple d'utilisation (à adapter selon tes chemins)
input_txt = r"C:\Users\XPS\Documents\Perso\Mémoire\data\mias\info.txt"
output_csv = r"C:\Users\XPS\Documents\Perso\Mémoire\data\mias\mias_annotations.csv"

parse_mias_annotations(input_txt, output_csv)
