from fastapi import FastAPI
from pydantic import BaseModel
import sys, os, time, re
from fastapi.middleware.cors import CORSMiddleware

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 내부 모듈 import
from getCleanQuery import getCleanQuery
from hybrid_searcher import hybrid_search
from llm_answer_generator import generate_llm_answer

# FastAPI 앱
app = FastAPI(title="PMI RAG System API with LLM")

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 바디 스키마
class QueryItem(BaseModel):
    query: str
    count: int = 100  # 기본값 100개
    target_table: str = "wellcome1st"

# 메인 엔드포인트
@app.post("/ask")
def ask_rag(item: QueryItem):
    """
    3단계 RAG 파이프라인:
    1. Claude로 쿼리 전처리 (오타 제거, 깔끔문장 생성)
    2. RRF 하이브리드 검색 (벡터 + 키워드)
    3. Claude로 최종 답변 생성
    """
    
    raw_query = item.query
    
    # -------------------------
    # 1단계: 쿼리 전처리 (Claude Sonnet 4.5)
    # -------------------------
    t0 = time.time()

    # --- 새 검색 파이프라인 ---
    from fastapi import FastAPI
    from pydantic import BaseModel
    import sys, os, time, re
    from fastapi.middleware.cors import CORSMiddleware
    from .db_search import bm25_search, vector_search
    from .sonnet_api import preprocess_query, filter_panels
    from .rrf_logic import rrf_rank
    import numpy as np

    app = FastAPI(title="PMI RAG System API with Sonnet & RRF")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class QueryItem(BaseModel):
        query: str
        count: int = 20

    @app.post("/search")
    def search_pipeline(item: QueryItem):
        """
        1. Sonnet으로 쿼리 전처리(깔끔문장)
        2. BM25, 벡터 검색 각각 top_k 추출
        3. RRF 융합
        4. Sonnet으로 패널 필터링
        5. JSON 반환
        """
        t0 = time.time()
        # 1. Sonnet 전처리
        clean_query = preprocess_query(item.query)

        # 2. BM25, 벡터 검색
        bm25_results = bm25_search(clean_query, top_k=item.count)
        # 벡터 임베딩 생성(여기서는 임시: 실제론 query_vectorizer 등 활용)
        # 예시: query_vec = get_query_vector(clean_query)
        query_vec = np.random.rand(1024)  # 실제론 적절한 벡터 생성 함수 사용
        vector_results = vector_search(query_vec, top_k=item.count)

        # 3. RRF 융합
        # id 기준으로 결과 통합
        bm25_ids = [r['id'] for r in bm25_results]
        vector_ids = [r['fk_id'] for r in vector_results]
        all_ids = list(set(bm25_ids + vector_ids))
        # RRF 점수 계산
        rrf_input = []
        for idx, id_ in enumerate(all_ids):
            bm25_rank = bm25_ids.index(id_) + 1 if id_ in bm25_ids else len(bm25_ids) + 1
            vector_rank = vector_ids.index(id_) + 1 if id_ in vector_ids else len(vector_ids) + 1
            rrf_score = rrf_rank([bm25_rank, vector_rank])
            rrf_input.append({"id": id_, "bm25_rank": bm25_rank, "vector_rank": vector_rank, "rrf_score": rrf_score})
        # RRF 점수 기준 정렬
        rrf_sorted = sorted(rrf_input, key=lambda x: x["rrf_score"], reverse=True)
        top_panels = rrf_sorted[:item.count * 2]

        # 4. Sonnet으로 패널 필터링
        panel_ids = [p["id"] for p in top_panels]
        filtered_ids = filter_panels(clean_query, [{"id": pid} for pid in panel_ids])

        # 5. JSON 반환
        t1 = time.time()
        return {
            "original_query": item.query,
            "clean_query": clean_query,
            "filtered_panel_ids": filtered_ids,
            "metrics": {
                "total_time": f"{t1-t0:.2f}s",
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
                "rrf_candidates": len(top_panels),
                "final_count": len(filtered_ids)
            }
        }

    @app.get("/")
    def root():
        return {"message": "RRF Search API with Sonnet running"}
