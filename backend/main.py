# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import search as search_func

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search = None


# 🔥 Preload model on startup (VERY IMPORTANT for Render)
@app.on_event("startup")
def startup_event():
    global search
    print("🚀 Starting server and loading model...")
    search = search_func

    # Warmup (loads model + embeddings)
    try:
        search("test")
        print("✅ Warmup complete")
    except Exception as e:
        print("⚠️ Warmup failed:", e)


@app.get("/")
def root():
    return {"message": "IR system is running 🚀"}


@app.get("/search")
def search_api(query: str, alpha: float = 0.2):
    try:
        if not query.strip():
            return {"error": "Empty query"}

        results = search(query, alpha)

        return {
            "query": query,
            "bm25": results["bm25"],
            "dense": results["dense"],
            "hybrid": results["hybrid"]
        }

    except Exception as e:
        return {"error": str(e)}