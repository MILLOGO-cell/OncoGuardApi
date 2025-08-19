text

# OncoGuardAPI

## Description

OncoGuardAPI est un système avancé d’analyse et de détection d’anomalies médicales, spécialisé dans le traitement d’images pour le diagnostic du cancer du sein.  
Cette API RESTful, développée en Python avec FastAPI, intègre des modules performants de traitement d’images et d’analyse scientifique, utilisant notamment :

- Clustering K-means pour la segmentation d’images  
- Classification floue pour la détection précise d’anomalies  
- Réseaux de neurones artificiels pour des modèles profonds d’apprentissage  
- Traitement d’images avec OpenCV et Pillow  
- Validation et gestion des données avec Pydantic  
- Sécurité des endpoints grâce à JWT, clés API et support optionnel de Mutual TLS (mTLS)

Ce projet est conçu pour être performant, modulaire et sécurisé, afin de s’intégrer dans des environnements cliniques exigeants.

---

## Outils et bibliothèques utilisées

- **FastAPI** : Framework web moderne, rapide et asynchrone pour construire l’API  
- **Uvicorn** : Serveur ASGI ultra-rapide pour exécuter FastAPI  
- **Pillow & OpenCV** : Librairies de traitement d’image  
- **NumPy & SciPy** : Pour les calculs numériques et le traitement scientifique  
- **Scikit-learn** : Implémentation d’algorithmes ML dont K-means  
- **Scikit-fuzzy** : Logiciel pour la classification floue  
- **TensorFlow / PyTorch** : Frameworks pour réseaux de neurones artificiels (optionnel)  
- **Python-JOSE & PassLib** : Gestion et sécurisation des tokens JWT et des mots de passe  
- **Matplotlib & Seaborn** : Visualisation graphique pour débogage et analyses exploratoires  

---

## Sécurité et certificats TLS/mTLS

Pour garantir la confidentialité et l’intégrité des échanges, OncoGuardAPI supporte la sécurisation via **TLS (HTTPS)**, avec la possibilité d’ajouter une couche supplémentaire de sécurité via **Mutual TLS (mTLS)**.  

- **TLS standard** sécurise la communication entre le client et le serveur en chiffrant les données.  
- **Mutual TLS (mTLS)** ajoute une authentification mutuelle, où le client doit présenter un certificat valide en plus du certificat serveur.  

### Gestion des certificats

- Les certificats privés (`server.key`) et publics (`server.crt`), ainsi que les certificats d’autorité (CA) pour la validation des clients, sont placés dans un dossier sécurisé dédié `certs/` à la racine du projet, par exemple :  

OncoGuardAPI/
├── certs/
│ ├── server.key
│ ├── server.crt
│ └── ca_client.crt # Pour mTLS

text

- Ces fichiers **ne sont pas versionnés** dans Git (ajouter `certs/` dans `.gitignore`) pour des raisons de sécurité.  
- Les chemins d’accès aux certificats sont référencés dans un fichier `.env` via des variables d’environnement, par exemple :  

TLS_CERT_PATH=./certs/server.crt
TLS_KEY_PATH=./certs/server.key
TLS_CA_CLIENT_CERT=./certs/ca_client.crt # Pour mTLS

text

### Mise en œuvre dans le déploiement

- FastAPI, via Uvicorn, permet d’activer TLS classique en lançant le serveur avec des options SSL :  

uvicorn app.main:app --host 0.0.0.0 --port 443
--ssl-keyfile ./certs/server.key
--ssl-certfile ./certs/server.crt

text

- Pour une gestion complète du **mTLS** (authentification client), il est recommandé d’utiliser un **proxy inverse** comme **Nginx** ou **Traefik** configuré pour valider les certificats clients avant de transmettre les requêtes à FastAPI.  
- Cette configuration garantit que seuls les clients disposant d’un certificat valide peuvent accéder à l’API, renforçant ainsi la sécurité en milieux sensibles.

---

## Installation

1. Clonez ce dépôt git :

git clone <https://votre-repository-url/OncoGuardAPI.git>
cd OncoGuardAPI

2. Créez et activez un environnement virtuel Python (recommandé) :

python -m venv venv
Linux/macOS

source venv/bin/activate
Windows PowerShell

venv\Scripts\activate

1. Installez les dépendances avec pip :

pip install -r requirements.txt

---

## Structure du projet

OncoGuardAPI/
│
├── app/
│ ├── main.py # Point d'entrée FastAPI
│ ├── api/ # Endpoints et routeurs API
│ ├── core/ # Configuration, sécurité
│ ├── db/ # Configuration base de données (SQLite par exemple)
│ ├── ml/ # Modules de machine learning (K-means, réseaux, classification BIRADS)
│ │ ├── birads_classifier.py # Logique classification BIRADS
│ │ ├── preprocess.py # Prétraitement d'image
│ │ └── utils.py # Fonctions auxiliaires
│ └── utils/ # Fonctions utilitaires diverses
│
├── certs/ # Certificats TLS/mTLS (non versionné)
│ ├── server.key
│ ├── server.crt
│ └── ca_client.crt
│
├── tests/ # Tests unitaires et d'intégration
├── .env # Variables d'environnement (non versionné)
├── requirements.txt # Dépendances Python
├── README.md # Ce fichier
└── .gitignore # Ignorer fichiers sensibles, envs, certs, etc.

text

---

## Lancement de l’API en mode développement

Démarrez le serveur FastAPI en mode développement avec rechargement automatique :

uvicorn app.main:app --reload

Migration
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

text

L’API sera accessible à l’adresse : `http://127.0.0.1:8000`

---

## Documentation interactive

FastAPI génère automatiquement la documentation Swagger accessible via :

<http://127.0.0.1:8000/docs>

text

ou la documentation Redoc :

<http://127.0.0.1:8000/redoc>

text

---

## Exemple minimal et complet d’endpoint FastAPI pour import et classification d’images BIRADS

Dans `app/api/v1/endpoints/analyze.py` :

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
from app.ml.preprocess import preprocess_image
from app.ml.birads_classifier import classify_birads

router = APIRouter()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
try:

# Ouvrir l'image uploadée

image = Image.open(file.file).convert("RGB")
except Exception:
raise HTTPException(status_code=400, detail="Fichier image invalide")

text

# Prétraiter l'image (redimension, normalisation, etc.)

processed_img = preprocess_image(image)

# Classifier avec le modèle BIRADS

birads_class = classify_birads(processed_img)

return {"birads_category": birads_class}

text

---

Dans `app/ml/preprocess.py` :

from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:

# Exemple de prétraitement : redimensionnement, conversion en numpy array normalisé

image = image.resize((224, 224))
img_array = np.array(image).astype("float32") / 255.0

# Ajouter batch dimension si nécessaire

img_array = np.expand_dims(img_array, axis=0)
return img_array

text

---

Dans `app/ml/birads_classifier.py` :

import numpy as np
Exemple simplifié : fonction simulant une classification BIRADS

def classify_birads(image_array: np.ndarray) -> str:

# Ici, on chargerait et exécuterait un modèle ML réel (ex: TensorFlow/PyTorch)

# Pour simplifier, on retourne une catégorie factice

# Exemples BIRADS : "BIRADS 1", "BIRADS 2", ..., "BIRADS 6"

# Ici, classification aléatoire exemple

import random
categories = ["BIRADS 1", "BIRADS 2", "BIRADS 3", "BIRADS 4", "BIRADS 5", "BIRADS 6"]
return random.choice(categories)

text

---

## Contact

Pour plus d’informations, contactez l’équipe de développement via [nicolasmillogo3@gmail.com](mailto:nicolasmillogo3@gmail.com).

---

*Ce projet est réalisé dans le cadre d’un mémoire de fin d’études pour améliorer la qualité et la rapidité du diagnostic médical dans des contextes à ressources limitées.*

---

*Licence : [Indiquez votre licence ici]*
