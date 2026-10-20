from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
import httpx
from pydantic import BaseModel
from sample_products import SAMPLE_PRODUCTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1.  الربط والتوكن
# -------------------------------------------------------------
BACKEND_API_URL = "http://shoppix.runasp.net/api/products"

BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhYmYyYjczZS0yNGVmLTQ5NTMtMWI0ZC0wOGRmMjExZjZmOWMiLCJqdGkiOiJhMTUwYWY4ZS1lZGU0LTQ0ODItYjUyNi0yMGZkNWEwMGMzYmQiLCJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6ImFiZjJiNzNlLTI0ZWYtNDk1My0xYjRkLTA4ZGYyMTFmNmY5YyIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL25hbWUiOiJZb3Vzc2VmIFNoYWFiYW4iLCJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9lbWFpbGFkZHJlc3MiOiJ5b3Vzc2VmQGdtYWlsLmNvbSIsImV4cCI6MTc5MjE4MDgwMCwiaXNzIjoiU2hvcHBpeEFwcCIsImF1ZCI6IlNob3BwaXggdXNlcnMifQ.N0e4qsV_P79wTBqLBNsoKNmvtr_ClqzBcAEmBe-Y8WM"
# 2.-Memory State
# -------------------------------------------------------------
products_catalog: List[dict] = []
tfidf_matrix = None
vectorizer = None
id_to_index: dict = {}


def train_engine(products: List[dict]):
   
    global products_catalog, tfidf_matrix, vectorizer, id_to_index

    if not products:
        return False

    products_catalog = products
   
    corpus = [
        f"{p.get('name', '')} {p.get('description', '')}".strip() for p in products
    ]


    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    
    id_to_index = {product["id"]: idx for idx, product in enumerate(products)}
    return True


async def sync_products_from_backend():
    headers = {}
    if BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {BEARER_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(BACKEND_API_URL, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    train_engine(data)
                    print(
                        f"[AI Service] Successfully fetched and indexed {len(data)} products from backend."
                    )
                    return
            else:
                print(
                    f"[AI Service] Backend returned status code {response.status_code}. Falling back to local data."
                )
    except Exception as error:
        print(
            f"[AI Service] Connection error ({error}). Falling back to local data."
        )

    # هشغل البيانات المحلية كبديل 
    train_engine(SAMPLE_PRODUCTS)
    print(
        f"[AI Service] Fallback active: Loaded {len(SAMPLE_PRODUCTS)} products from sample_products.py"
    )



# 3.Lifespan Contex
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await sync_products_from_backend()
    yield


app = FastAPI(
    title="Shoppix AI Recommendation & Search Service",
    description="Content-Based Recommendation and Semantic Search Engine using TF-IDF and Cosine Similarity",
    version="1.2.0",
    lifespan=lifespan,
)



# 4. Pydantic Data Contracts
# -------------------------------------------------------------
class ProductResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = ""
    price: float
    stockQuantity: int
    imageUrl: Optional[str] = ""
    score: float



# 5. Endpoints
# -------------------------------------------------------------


@app.get(
    "/related/{product_id}",
    response_model=List[ProductResponse],
    summary="Get Related Products",
)
def get_related_products(product_id: int, limit: int = Query(default=4, ge=1)):
    """ترشيح المنتجات المشابهة لمنتج معين بناءً على الاسم والوصف."""
    if product_id not in id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Product with id {product_id} was not found in catalog.",
        )

    target_idx = id_to_index[product_id]
    scores = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()

    results = []
    for idx, score in enumerate(scores):
        candidate = products_catalog[idx]
        # نستبعد المنتج نفسه من القائمة 
        if candidate["id"] != product_id:
            item = candidate.copy()
            item["score"] = round(float(score), 4)
            results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


@app.get(
    "/search",
    response_model=List[ProductResponse],
    summary="Semantic Product Search",
)
def search_products(
    q: str = Query(..., min_length=1, description="Search query keyword"),
    limit: int = Query(default=5, ge=1),
):
    """السمانتك هستخدم keywords التشابه الدلالي"""
    if not vectorizer or tfidf_matrix is None:
        raise HTTPException(
            status_code=503, detail="Search engine is not ready."
        )

    
    query_vector = vectorizer.transform([q])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = []
    for idx, score in enumerate(scores):

        if score > 0.0:
            item = products_catalog[idx].copy()
            item["score"] = round(float(score), 4)
            results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


@app.post("/sync", summary="Re-sync Products and Retrain Engine")
async def trigger_sync():
    
    await sync_products_from_backend()
    return {
        "status": "success",
        "total_products_indexed": len(products_catalog),
        "message": "AI Engine synchronized and retrained successfully.",
    } 
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)