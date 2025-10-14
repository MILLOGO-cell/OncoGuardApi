# 🩺 OncoGuardAPI

Système d’analyse et de détection d’anomalies médicales (cancer du sein) basé sur **FastAPI** et des modèles de **Machine Learning** (HOG/LBP/Wavelets + SVM/XGBoost).  
L’API permet de **téléverser, anonymiser et classer** des images mammographiques selon la classification **BI-RADS** (1, 2, 5).

---

## 🚀 Fonctionnalités principales

- API **FastAPI** documentée via **Swagger** et **ReDoc**
- Pipeline complet d’analyse d’images médicales (upload → prétraitement → inférence)
- Extraction de **features** : HOG, LBP, Wavelets
- Modèles **SVM** et **XGBoost** avec sélection automatique du meilleur score
- Enregistrement automatique des prédictions en base de données
- Système d’**anonymisation DICOM/PNG**
- Endpoints pour **exportation ZIP** et **statistiques BI-RADS**
- Configuration **CORS prête pour Next.js**

---

## 🧱 Structure du projet

OncoGuardAPI/
├─ app/
│ ├─ api/v1/
│ │ ├─ routes/
│ │ │ ├─ auth.py # Authentification / utilisateurs
│ │ │ ├─ image_inference.py # Prédiction d’image
│ │ │ ├─ ingest.py # Anonymisation + prédiction
│ │ │ ├─ ingest_files.py # Listing / téléchargement / export
│ │ │ └─ stats.py # Statistiques et rapports
│ │ ├─ models/ # SQLAlchemy
│ │ └─ schemas/ # Pydantic
│ ├─ db/ # Connexion / Base SQLAlchemy
│ ├─ ml/ # Entraînement et pipeline ML
│ ├─ ingest/ # Modules d’anonymisation DICOM
│ ├─ core/config.py # Variables d’environnement
│ └─ main.py # Entrée principale FastAPI
├─ .env.development
├─ requirements.txt
├─ README.md
└─ venv/

---

## 🛠️ Installation

```bash
git clone <URL_DU_REPO> OncoGuardAPI
cd OncoGuardAPI

python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

requirements.txt minimal
fastapi
uvicorn[standard]
SQLAlchemy
pydantic
pydantic-settings
python-multipart
python-jose[cryptography]
passlib[bcrypt]
email-validator
numpy
pandas
scikit-learn
scikit-image
opencv-python-headless
PyWavelets
xgboost
joblib
pydicom
Pillow

⚙️ Configuration .env.development
DEBUG=True
DATABASE_URL=sqlite:///./dev.db
SECRET_KEY=dev_key

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USE_TLS=True
SMTP_USER=nicolasmillogo3@gmail.com
SMTP_PASSWORD=fnbt tyzk nipm etuc
FROM_EMAIL=nicolasmillogo3@gmail.com

FRONTEND_URL=http://localhost:3000
API_KEY=devapikey

🌐 CORS et connexion Next.js

La configuration CORS dans app/main.py autorise :

origins = {"http://localhost:3000", "http://127.0.0.1:3000", settings.FRONTEND_URL}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Et côté Next.js, assure-toi d’envoyer les credentials :

await fetch("http://127.0.0.1:8000/api/v1/image-inference/predict", {
  method: "POST",
  credentials: "include",
  body: formData,
});

🧠 Entraînement du modèle
python -m app.ml.train


Cela :

Charge le dataset MIAS

Extrait les features (HOG/LBP/Wavelets)

Évalue SVM et XGBoost

Sauvegarde le meilleur pipeline dans app/ml/models/*.joblib

🔬 Endpoints principaux
Catégorie Endpoint Description
🔐 Authentification /api/v1/auth/* Gestion des utilisateurs / mot de passe
🩻 Analyse d'images /api/v1/image-inference/predict Téléverse une image et retourne la prédiction BI-RADS
📊 Statistiques /api/v1/stats/summary Résumé statistique des analyses
🧩 Anonymisation /api/v1/ingest/anonymize Anonymise des images DICOM/PNG et (optionnel) lance la prédiction
🗂️ Fichiers /api/v1/ingest/files Liste les images anonymisées disponibles
💾 Exportation /api/v1/ingest/export/zip Exporte tout ou partie des images en ZIP
📈 Exemple de prédiction
curl -X POST "http://127.0.0.1:8000/api/v1/image-inference/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@app/data/mias/mdb001.pgm;type=image/pgm"


Réponse :

{
  "label": "Normal",
  "birads": "BI-RADS 1",
  "confidence": 0.9575,
  "filename": "9bc177f2-ac36-4cbf-93bc-2279ca1881af.png"
}

📊 Statistiques

Endpoint :
GET /api/v1/stats/summary

Retourne :

total d’images analysées

moyenne de confiance

répartition par BI-RADS

série temporelle (30 derniers jours)

histogramme de confiance (10 bins)

🧮 Anonymisation

Endpoint :
POST /api/v1/ingest/anonymize

Prend plusieurs fichiers (PNG ou DICOM), exécute le pipeline app/ingest/pipeline.py, puis :

Crée des copies anonymisées

Stocke les métadonnées (âge, ID, etc.)

Optionnel : effectue la prédiction automatique

Permet la persistance en base (ImageAnalysis)

💾 Export et téléchargement

Endpoints :

/api/v1/ingest/files → liste des images disponibles

/api/v1/ingest/download/{kind}/{filename} → téléchargement individuel

/api/v1/ingest/export/zip → export groupé (avec filenames= ou all_files=true)

📚 Documentation interactive

Swagger : http://127.0.0.1:8000/docs

ReDoc : http://127.0.0.1:8000/redoc

🧯 Dépannage rapide
Erreur Solution
ImportError: No module named 'pydicom' pip install pydicom
FileNotFoundError: mias_annotations.csv python -m app.utils.parse
cv2 error (GUI) pip install opencv-python-headless
xgboost not found pip install xgboost
📜 Licence

MIT License © 2025 Nicolas Millogo
Utilisation libre, avec mention de l’auteur.

📧 Contact

Mainteneur : nicolasmillogo3@gmail.com

## 📝 Licence

MIT License

Copyright (c) 2025 Nicolas Millogo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

