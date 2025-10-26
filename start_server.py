# start_server.py - Démarrage uvicorn optimisé pour Windows
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=[
            # Exclure venv pour éviter WinError 1450
            "venv/*",
            ".venv/*",
            "env/*",
            "*.pyc",
            "__pycache__/*",
            
            # Exclure dossiers de données
            "data/*",
            "uploads/*",
            "tagged/*",
            
            # Exclure dossiers de test
            "tests/*",
            ".pytest_cache/*",
            
            # Exclure fichiers système
            ".git/*",
            ".vscode/*",
            "*.log",
        ],
        # Limite de logs pour éviter les buffers trop gros
        log_level="info",
        access_log=True,
    )