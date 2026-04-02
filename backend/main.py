from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

search = None

@app.get("/")
def root():
    return {"message": "IR system is running 🚀"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
def search_api(query: str, alpha: float = 0.2):
    global search

    if search is None:
        print("Importing model lazily...")
        from backend.model import search as search_func
        search = search_func

    return {
        "query": query,
        "results": search(query, alpha)
    }