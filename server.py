from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Treasure Runner is running",
        "status": "success"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}