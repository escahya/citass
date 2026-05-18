import os
from typing import List, Optional

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing")

app = FastAPI(title="Citation Recommendation API")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


class RecommendationRequest(BaseModel):
    text: str
    limit: int = 5


class PaperRecommendation(BaseModel):
    id: int
    title: str
    authors: Optional[str]
    publication_year: Optional[int]
    abstract: str
    doi: Optional[str]
    url: Optional[str]
    similarity: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/embed-papers")
def embed_papers():
    """
    Generate embeddings for papers that do not have embeddings yet.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, abstract
                FROM papers
                WHERE embedding IS NULL
            """)

            rows = cur.fetchall()

            if not rows:
                return {"message": "No papers need embedding", "updated": 0}

            updated = 0

            for paper_id, title, abstract in rows:
                text = f"{title}. {abstract}"
                embedding = model.encode(text, normalize_embeddings=True).tolist()

                cur.execute(
                    """
                    UPDATE papers
                    SET embedding = %s
                    WHERE id = %s
                    """,
                    (embedding, paper_id)
                )

                updated += 1

            conn.commit()

    return {"message": "Embeddings generated", "updated": updated}


@app.post("/recommendations", response_model=List[PaperRecommendation])
def recommend_papers(request: RecommendationRequest):
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short")

    query_embedding = model.encode(
        request.text,
        normalize_embeddings=True
    ).tolist()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    title,
                    authors,
                    publication_year,
                    abstract,
                    doi,
                    url,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM papers
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, request.limit)
            )

            rows = cur.fetchall()

    return [
        PaperRecommendation(
            id=row[0],
            title=row[1],
            authors=row[2],
            publication_year=row[3],
            abstract=row[4],
            doi=row[5],
            url=row[6],
            similarity=float(row[7]),
        )
        for row in rows
    ]
