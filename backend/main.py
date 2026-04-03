from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .model import search as search_func

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "IR system is running 🚀"}


@app.get("/search")
def search_api(query: str, alpha: float = 0.2):
    try:
        if not query.strip():
            return {"error": "Empty query"}

        results = search_func(query, alpha)

        return {
            "query": query,
            "bm25": results["bm25"],
            "dense": results["dense"],
            "hybrid": results["hybrid"]
        }

    except Exception as e:
        return {"error": str(e)}