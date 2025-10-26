# 🩺 OncoGuardAPI

Système d'analyse et de détection d'anomalies médicales (cancer du sein) basé sur **FastAPI** et des modèles de **Machine Learning** (HOG/LBP/Wavelets + SVM/XGBoost).  

L'API permet de **téléverser, anonymiser, analyser et classer** des images mammographiques selon la classification **BI-RADS** (1, 2, 5) avec **preview DICOM intégré**.

---

## ✨ Fonctionnalités principales

### 🔬 Analyse d'images

- ✅ Upload d'images individuelles ou par lots (PNG, JPEG, PGM, DICOM)
- ✅ Preview automatique des images DICOM (conversion PNG à la volée)
- ✅ Extraction de features : HOG, LBP, Wavelets
- ✅ Classification BI-RADS avec modèles SVM et XGBoost
- ✅ Annotations visuelles avec confiance et classification
- ✅ Enregistrement automatique en base de données

### 🔒 Anonymisation

- ✅ Pipeline d'anonymisation DICOM/PNG
- ✅ Extraction et anonymisation des métadonnées patient
- ✅ Génération d'identifiants uniques
- ✅ Conservation sécurisée des données originales

### 📦 Gestion de fichiers

- ✅ Upload fichiers individuels ou multiples
- ✅ Liste paginée avec filtres et recherche
- ✅ Preview images (DICOM → PNG automatique)
- ✅ Téléchargement individuel ou par lots (ZIP)
- ✅ Suppression de fichiers
- ✅ Export ZIP optimisé avec streaming par chunks

### 📊 Statistiques et rapports

- ✅ Dashboard statistique complet
- ✅ Répartition par classification BI-RADS
- ✅ Analyse temporelle (30 derniers jours)
- ✅ Histogramme de confiance
- ✅ Export CSV des résultats

### 🔐 Sécurité

- ✅ Authentification JWT
- ✅ Gestion des utilisateurs
- ✅ Validation stricte des chemins de fichiers
- ✅ CORS configuré pour Next.js

---

## 🧱 Structure du projet

OncoGuardAPI/
├── app/
│   ├── api/v1/
│   │   ├── routes/
│   │   │   ├── auth.py                 # Authentification / utilisateurs
│   │   │   ├── image_inference.py      # Prédiction + annotations
│   │   │   ├── ingest.py               # Upload, preview, anonymisation
│   │   │   └── stats.py                # Statistiques et rapports
│   │   ├── models/                     # SQLAlchemy ORM
│   │   └── schemas/                    # Pydantic validation
│   ├── db/                             # Connexion base de données
│   ├── ml/                             # Pipeline ML + entraînement
│   ├── ingest/                         # Modules anonymisation
│   ├── core/
│   │   └── config.py                   # Variables d'environnement
│   └── main.py                         # Application FastAPI
├── data/
│   ├── derived/                        # Images PGM/PNG
│   ├── normalized_dicom/               # Fichiers DICOM
│   └── mias/                           # Dataset MIAS
├── uploads/                            # Uploads temporaires
├── tagged/                             # Images annotées
├── .env.development                    # Configuration
├── requirements.txt                    # Dépendances Python
├── start_server.py                     # Script démarrage optimisé
└── README.md
---

## 🛠️ Installation

### 1. Cloner le repository

```bash
git clone <URL_DU_REPO> OncoGuardAPI
cd OncoGuardAPI
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration requise

Créer un fichier `.env.development` :

```env
# Base de données
DATABASE_URL=sqlite:///./dev.db

# Sécurité
DEBUG=True
SECRET_KEY=dev_key_change_in_production
API_KEY=devapikey

# Email (optionnel)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USE_TLS=True
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com

# Frontend
FRONTEND_URL=http://localhost:3000
```

---

## 📦 Dépendances principales

```txt
# API Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Base de données
SQLAlchemy>=2.0.0
alembic>=1.12.0

# Validation
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6

# Authentification
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
email-validator>=2.0.0

# Machine Learning
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
xgboost>=2.0.0
joblib>=1.3.0

# Traitement d'images
opencv-python>=4.8.0
PyWavelets>=1.4.1
Pillow>=10.0.0

# Médical
pydicom>=2.4.0
```

---

## 🚀 Démarrage

### Option 1 : Script optimisé (recommandé)

```bash
python start_server.py
```

Ce script :

- ✅ Exclut le `venv` du hot-reload (évite WinError 1450)
- ✅ Optimise les performances sous Windows
- ✅ Active le hot-reload uniquement sur `app/`

### Option 2 : Commande directe

```bash
uvicorn app.main:app --reload
```

### Option 3 : Sans reload

```bash
uvicorn app.main:app
```

L'API sera disponible sur : **<http://127.0.0.1:8000>**

---

## 🧠 Entraînement du modèle

### 1. Préparer le dataset MIAS

```bash
# Télécharger le dataset MIAS dans app/data/mias/
# Structure attendue :
# app/data/mias/
#   ├── mdb001.pgm
#   ├── mdb002.pgm
#   └── mias_annotations.csv
```

### 2. Entraîner les modèles

```bash
python -m app.ml.train
```

Ce script :

- Charge les images du dataset MIAS
- Extrait les features (HOG + LBP + Wavelets)
- Entraîne SVM et XGBoost
- Sauvegarde le meilleur modèle dans `app/ml/models/`

### 3. Vérifier les modèles

```bash
ls app/ml/models/
# Devrait contenir :
# - best_pipeline.joblib
# - feature_names.joblib
```

---

## 🔬 Endpoints API

### 🔐 Authentification

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/auth/register` | Créer un compte utilisateur |
| POST | `/api/v1/auth/login` | Se connecter (JWT) |
| POST | `/api/v1/auth/forgot-password` | Réinitialiser mot de passe |

### 🩻 Analyse d'images

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/image-inference/predict` | Analyse une image (retourne BI-RADS) |
| POST | `/api/v1/image-inference/predict-batch` | Analyse multiple d'images |
| GET | `/api/v1/image-inference/results` | Liste des résultats d'analyses |
| GET | `/api/v1/image-inference/tagged/{filename}` | Télécharger image annotée |
| GET | `/api/v1/image-inference/report.csv` | Export CSV des résultats |
| GET | `/api/v1/image-inference/export-tagged.zip` | Export ZIP des annotations |

### 📦 Gestion des fichiers

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/v1/ingest/files` | Liste des fichiers (PGM/DICOM) |
| GET | `/api/v1/ingest/preview/{kind}/{filename}` | Preview image (DICOM→PNG) |
| GET | `/api/v1/ingest/download/{kind}/{filename}` | Télécharger un fichier |
| POST | `/api/v1/ingest/upload` | Upload un fichier |
| POST | `/api/v1/ingest/upload-batch` | Upload multiple |
| POST | `/api/v1/ingest/export/zip` | Export ZIP (sélection/tout) |
| DELETE | `/api/v1/ingest/delete/{kind}/{filename}` | Supprimer un fichier |

### 🧩 Anonymisation

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/ingest/anonymize` | Anonymise DICOM/PNG + prédiction optionnelle |

### 📊 Statistiques

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/v1/stats/summary` | Dashboard statistique complet |

---

## 📖 Exemples d'utilisation

### 1. Prédiction d'une image

#### cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/image-inference/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.pgm"
```

#### Python

```python
import requests

with open("image.pgm", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/image-inference/predict",
        files={"file": f}
    )
    print(response.json())
```

#### Réponse

```json
{
  "label": "Normal",
  "birads": "BI-RADS 1",
  "confidence": 0.9575,
  "filename": "9bc177f2.png",
  "tagged_filename": "9bc177f2__tag.png"
}
```

### 2. Upload de fichiers

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ingest/upload?kind=dicom" \
  -F "file=@scan.dcm"
```

### 3. Preview DICOM

```bash
# L'endpoint convertit automatiquement DICOM → PNG
curl "http://127.0.0.1:8000/api/v1/ingest/preview/dicom/scan.dcm" \
  --output preview.png
```

### 4. Export ZIP

```bash
# Tout exporter
curl -X POST "http://127.0.0.1:8000/api/v1/ingest/export/zip?kind=pgm&all_files=true" \
  --output export.zip

# Sélection spécifique
curl -X POST "http://127.0.0.1:8000/api/v1/ingest/export/zip?kind=dicom&filenames=scan1.dcm&filenames=scan2.dcm" \
  --output selected.zip
```

### 5. Statistiques

```bash
curl "http://127.0.0.1:8000/api/v1/stats/summary"
```

Réponse :

```json
{
  "total": 150,
  "distinct_patients": null,
  "avg_confidence": 0.8923,
  "by_birads": [
    {"label": "BI-RADS 1", "count": 80},
    {"label": "BI-RADS 2", "count": 50},
    {"label": "BI-RADS 5", "count": 20}
  ],
  "by_label": [
    {"label": "Normal", "count": 80},
    {"label": "Benign", "count": 50},
    {"label": "Malignant", "count": 20}
  ],
  "last_30d_series": [...],
  "confidence_histogram": {...}
}
```

---

## 🌐 Intégration Next.js

### Configuration CORS

Le backend autorise automatiquement :

```python
origins = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    settings.FRONTEND_URL
}
```

### Exemple fetch client

```typescript
// Upload et prédiction
const formData = new FormData();
formData.append('file', file);

const response = await fetch('http://127.0.0.1:8000/api/v1/image-inference/predict', {
  method: 'POST',
  credentials: 'include',
  body: formData,
});

const result = await response.json();
console.log(result);
```

### Affichage preview DICOM

```tsx
// components/ImagePreview.tsx
<img 
  src={`http://127.0.0.1:8000/api/v1/ingest/preview/dicom/${filename}`}
  alt="DICOM Preview"
/>
```

---

## 📚 Documentation interactive

Une fois le serveur démarré :

- **Swagger UI** : <http://127.0.0.1:8000/docs>
- **ReDoc** : <http://127.0.0.1:8000/redoc>
- **OpenAPI JSON** : <http://127.0.0.1:8000/openapi.json>

---

## 🧪 Tests

### Tests unitaires

```bash
pytest tests/ -v
```

### Tests d'intégration

```bash
pytest tests/test_files_integration.py -v
```

### Tests de performance

```bash
pytest tests/test_performance.py -v --benchmark
```

---

## 🐛 Dépannage

### Problème : Export ZIP bloque à 25%

**Solution :** Utiliser `start_server.py` au lieu de `uvicorn --reload`

```bash
python start_server.py
```

### Problème : WinError 1450 (Windows)

**Cause :** Uvicorn surveille trop de fichiers
**Solution :** Le script `start_server.py` exclut le venv automatiquement

### Problème : Images DICOM ne s'affichent pas

**Solution :** Installer les dépendances manquantes

```bash
pip install pydicom opencv-python numpy
```

### Problème : ImportError pydicom

```bash
pip install --upgrade pydicom
```

### Problème : ModuleNotFoundError

```bash
# Réinstaller toutes les dépendances
pip install -r requirements.txt --force-reinstall
```

### Problème : Base de données locked

```bash
# Supprimer et recréer la base
rm dev.db
python -m app.db.init_db
```

---

## 🔧 Configuration avancée

### Variables d'environnement complètes

```env
# Application
DEBUG=True
API_VERSION=v1
APP_NAME=OncoGuardAPI

# Base de données
DATABASE_URL=sqlite:///./dev.db
# ou PostgreSQL :
# DATABASE_URL=postgresql://user:pass@localhost/onceguard

# Sécurité
SECRET_KEY=your-secret-key-here-min-32-chars
API_KEY=your-api-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
FRONTEND_URL=http://localhost:3000
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USE_TLS=True
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com

# Storage
UPLOAD_DIR=./uploads
TAGGED_DIR=./tagged
DERIVED_IMG_DIR=./data/derived
NORMALIZED_DICOM_DIR=./data/normalized_dicom

# ML
MODEL_PATH=./app/ml/models/best_pipeline.joblib
CONFIDENCE_THRESHOLD=0.5
```

### Personnaliser les ports

```bash
# Port 8080
python start_server.py --port 8080

# Ou
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Mode production

```bash
# Désactiver DEBUG
DEBUG=False

# Utiliser gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 📊 Performance

### Optimisations implémentées

- ✅ Streaming ZIP par chunks (256KB)
- ✅ Hot-reload optimisé (exclusion venv)
- ✅ Conversion DICOM à la volée (pas de cache)
- ✅ Pagination des résultats (limite 5000)
- ✅ Index base de données sur colonnes fréquentes

### Benchmarks

| Operation | Temps moyen | Mémoire |
|-----------|-------------|---------|
| Prédiction 1 image | ~500ms | ~150MB |
| Upload DICOM | ~200ms | ~50MB |
| Preview DICOM→PNG | ~300ms | ~100MB |
| Export ZIP 100 images | ~5s | ~200MB |

---

## 🚀 Déploiement

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t onceguard-api .
docker run -p 8000:8000 onceguard-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - DATABASE_URL=postgresql://user:pass@db/onceguard
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=onceguard
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
```

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📝 Changelog

### v2.0.0 (2025-10-26)

- ✨ Ajout preview DICOM automatique
- ✨ Upload fichiers individuels/multiples
- ✨ Suppression de fichiers
- 🐛 Fix streaming ZIP (chunks 256KB)
- 🐛 Fix WinError 1450 Windows
- ⚡ Optimisation hot-reload
- 📚 Documentation complète

### v1.0.0 (2025-10-19)

- 🎉 Release initiale
- ✅ Pipeline ML complet
- ✅ Anonymisation DICOM
- ✅ API REST FastAPI

---

## 📄 Licence

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

---

## 📧 Contact

**Mainteneur :** Nicolas Millogo  
**Email :** <nicolasmillogo3@gmail.com>  
**GitHub :** [OncoGuardAPI](https://github.com/your-username/OncoGuardAPI)

---

## 🙏 Remerciements

- Dataset MIAS pour les images médicales
- FastAPI pour le framework web
- scikit-learn et XGBoost pour le ML
- pydicom pour le traitement DICOM
- La communauté open-source

---

## 🔗 Liens utiles

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Dataset MIAS](http://peipa.essex.ac.uk/info/mias.html)
- [BI-RADS Classification](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Bi-Rads)
- [pydicom Documentation](https://pydicom.github.io/)

---

**⭐ Si ce projet vous a aidé, n'hésitez pas à laisser une étoile !**
