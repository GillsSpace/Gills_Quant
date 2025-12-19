from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()

# This must match the internal path in your docker-compose
TOKEN_PATH = "/app/secrets/tokens.json"

@app.post("/update-token")
async def update_token(file: UploadFile = File(...)):
    with open(TOKEN_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "Success", "path": TOKEN_PATH}