FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn python-multipart

COPY token-server.py .

# Change 'main' to 'token-server'
CMD ["uvicorn", "token-server:app", "--host", "0.0.0.0", "--port", "8000"]