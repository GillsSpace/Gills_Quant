import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from interface.mobile_app.backend.token_service import (
    get_token_status,
    get_auth_url,
    perform_token_update,
    upload_tokens_db
)

app = FastAPI(
    title="Gills Schwab Token Update Mobile API",
    description="Mobile backend API for updating Charles Schwab OAuth tokens and syncing with Tailscale server.",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UpdateTokenRequest(BaseModel):
    redirect_url: Optional[str] = None

@app.get("/api/status")
def status_endpoint():
    """Returns current token status and expiry information."""
    return get_token_status()

@app.get("/api/auth-url")
def auth_url_endpoint(callback_url: str = "https://127.0.0.1"):
    """Returns the Schwab OAuth Authorization URL for logging in."""
    try:
        return get_auth_url(callback_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-tokens")
def update_tokens_endpoint(req: UpdateTokenRequest = UpdateTokenRequest()):
    """
    Triggers token refresh/update.
    Optionally accepts a redirect URL or authorization code from Schwab login.
    Uploads updated tokens.db to the Tailscale remote server.
    """
    try:
        res = perform_token_update(req.redirect_url)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-db")
def upload_db_endpoint():
    """Directly uploads tokens.db to Tailscale remote server."""
    try:
        return upload_tokens_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static file serving for Frontend PWA build
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
