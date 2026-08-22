from fastapi import FastAPI

app = FastAPI(title="Fraud Detection System")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud detection API is running"}