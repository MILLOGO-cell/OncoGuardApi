import base64
import hashlib
import hmac
import json
import os
from fastapi import Request, HTTPException

# Chargement dynamique des secrets depuis fichier JSON
CERTS_PATH = os.path.join(os.path.dirname(__file__), "..", "certs", "api_secrets.json")

with open(CERTS_PATH, "r") as f:
    API_SECRETS = json.load(f)

async def verify_signature(request: Request):
    """
    Vérifie la signature HMAC d'une requête signée.

    Les clés sont chargées depuis `certs/api_secrets.json`.

    En-têtes obligatoires :
    - X-Cert-ID : identifiant client
    - X-Signature : signature base64 du corps
    """
    cert_id = request.headers.get("X-Cert-ID")
    signature = request.headers.get("X-Signature")

    if not cert_id or not signature:
        raise HTTPException(status_code=401, detail="Signature requise")

    secret = API_SECRETS.get(cert_id)
    if not secret:
        raise HTTPException(status_code=403, detail="Certificat non reconnu")

    raw_body = await request.body()
    computed_signature = base64.b64encode(
        hmac.new(secret.encode(), raw_body, hashlib.sha256).digest()
    ).decode()

    if not hmac.compare_digest(computed_signature, signature):
        raise HTTPException(status_code=401, detail="Signature invalide")
