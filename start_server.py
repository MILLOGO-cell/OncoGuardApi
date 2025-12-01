import os
import uvicorn

if __name__ == "__main__":
    is_production = os.getenv("DEBUG", "True").lower() == "false"
    
    if is_production:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True,
        )
    else:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_excludes=[
                "venv/*",
                ".venv/*",
                "env/*",
                "*.pyc",
                "__pycache__/*",
                "data/*",
                "uploads/*",
                "tagged/*",
                "tests/*",
                ".pytest_cache/*",
                ".git/*",
                ".vscode/*",
                "*.log",
            ],
            log_level="info",
            access_log=True,
        )