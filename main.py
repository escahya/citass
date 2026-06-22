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
    user_session_id: int
    limit: int = 5


class PaperRecommendation(BaseModel):
    recommendation_result_id: int
    paper_id: int
    title: str
    authors: Optional[str]
    publication_year: Optional[int]
    abstract: str
    doi: Optional[str]
    url: Optional[str]
    similarity: float


class StartSessionRequest(BaseModel):
    google_user_id: Optional[str] = None


class StartSessionResponse(BaseModel):
    user_session_id: int


class RecommendationsResponse(BaseModel):
    recommendation_session_id: int
    papers: List[PaperRecommendation]


class FeedbackRequest(BaseModel):
    recommendation_result_id: int
    relevance_score: Optional[int] = None
    opened_paper: Optional[bool] = False
    citation_inserted: Optional[bool] = False


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

@app.post("/session/start", response_model=StartSessionResponse)
def start_session(request: StartSessionRequest):

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            user_id = None

            # 1. Create/find user
            if request.google_user_id:

                cur.execute(
                    """
                    INSERT INTO users
                    (
                        google_user_id
                    )

                    VALUES (%s)

                    ON CONFLICT
                    (
                        google_user_id
                    )

                    DO UPDATE SET
                        google_user_id =
                            EXCLUDED.google_user_id

                    RETURNING id
                    """,
                    (
                        request.google_user_id,
                    )
                )

                user_id = cur.fetchone()[0]


            else:

                # anonymous user
                cur.execute(
                    """
                    INSERT INTO users
                    DEFAULT VALUES
                    RETURNING id
                    """
                )

                user_id = cur.fetchone()[0]


            # 2. Create user session
            cur.execute(
                """
                INSERT INTO user_sessions
                (
                    user_id
                )

                VALUES (%s)

                RETURNING id
                """,
                (
                    user_id,
                )
            )

            user_session_id = cur.fetchone()[0]

            conn.commit()


    return StartSessionResponse(
        user_session_id=user_session_id
    )

@app.post("/recommendations", response_model=RecommendationsResponse)
def recommend_papers(request: RecommendationRequest):

    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text is too short"
        )


    query_embedding = model.encode(
        request.text,
        normalize_embeddings=True
    ).tolist()

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            # 1. Create recommendation session
            cur.execute(
                """
                INSERT INTO recommendation_sessions
                (
                    user_session_id,
                    selected_text
                )
                VALUES (%s,%s)
                RETURNING id
                """,
                (
                    request.user_session_id,
                    request.text
                )
            )

            recommendation_session_id = cur.fetchone()[0]

            # 2. Search papers
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
                (
                    query_embedding,
                    query_embedding,
                    request.limit
                )
            )

            rows = cur.fetchall()

            papers = []

            # 3. Store recommendation results
            position = 1

            for row in rows:

                cur.execute(
                    """
                    INSERT INTO recommendation_results
                    (
                        recommendation_session_id,
                        paper_id,
                        similarity_score,
                        position
                    )

                    VALUES (%s,%s,%s,%s)

                    RETURNING id

                    """,
                    (
                        recommendation_session_id,
                        row[0],
                        float(row[7]),
                        position
                    )
                )

                recommendation_result_id = cur.fetchone()[0]

                papers.append(
                    PaperRecommendation(
                        recommendation_result_id=
                            recommendation_result_id,

                        paper_id=row[0],
                        title=row[1],
                        authors=row[2],
                        publication_year=row[3],
                        abstract=row[4],
                        doi=row[5],
                        url=row[6],
                        similarity=float(row[7])
                    )
                )

                position += 1
            
            # 4. Update Session time
            cur.execute(
                """
                UPDATE user_sessions
                SET last_activity_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    request.user_session_id,
                )
            )
            if cur.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="User session not found"
                )

            conn.commit()


    return RecommendationsResponse(
        recommendation_session_id=
            recommendation_session_id,

        papers=papers
    )


@app.post("/feedback")
def save_feedback(request: FeedbackRequest):

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:


            cur.execute(
                """
                INSERT INTO recommendation_feedback
                (
                    recommendation_result_id,
                    relevance_score,
                    opened_paper,
                    citation_inserted
                )

                VALUES (%s,%s,%s,%s)


                ON CONFLICT
                (
                    recommendation_result_id
                )

                DO UPDATE SET

                    relevance_score =
                    COALESCE(
                        EXCLUDED.relevance_score,
                        recommendation_feedback.relevance_score
                    ),


                    opened_paper =
                    recommendation_feedback.opened_paper
                    OR EXCLUDED.opened_paper,


                    citation_inserted =
                    recommendation_feedback.citation_inserted
                    OR EXCLUDED.citation_inserted

                """,
                (
                    request.recommendationResultId,
                    request.relevanceScore,
                    request.openedPaper,
                    request.citationInserted
                )
            )


            conn.commit()


    return {
        "success": True
    }
