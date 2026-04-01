# backend/main.py

from fastapi import FastAPI
from model import search
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "IR system is running 🚀"}


@app.get("/search")
def search_api(query: str, alpha: float = 0.2):
    results = search(query, alpha)
    return {
        "query": query,
        "results": results
    }




