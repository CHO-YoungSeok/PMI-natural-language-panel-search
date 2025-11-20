# --- 새 검색 파이프라인 ---
from fastapi import FastAPI
from pydantic import BaseModel
import sys, os, time, re, json
from fastapi.middleware.cors import CORSMiddleware
from db_search import fts_search, vector_search, get_jsons_by_ids
from sonnet_api import preprocess_query, llm_filter_panel
from rrf_logic import rrf_rank
from query_vectorizer import get_query_vector
import numpy as np
from dotenv import load_dotenv

# 프로젝트 .env에 설정한 환경변수들 활성화
load_dotenv()

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
    count: int = 300

@app.post("/ask")
def search_pipeline(item: QueryItem):
    """
    1. Sonnet으로 쿼리 전처리(깔끔문장)
    2. fts, 벡터 검색 각각 top_k 추출
    3. RRF 융합
    4. Sonnet으로 패널 필터링
    5. JSON 반환
    """
    t0 = time.time()
    # 1. Sonnet 전처리
    clean_query = preprocess_query(item.query)
    print(f"clean query: {clean_query}")

    # 2. postgreSQL의 Full Text Search(FTS), 벡터 코사인 유사도 검색
    fts_results = fts_search(clean_query, top_k=item.count)
    print("----------------------------------------------")
    print("fts_result")
    print(fts_results)

    query_vec = get_query_vector(clean_query) 
    vector_results = vector_search(query_vec, top_k=item.count)
    print("----------------------------------------------")
    print("query_vec 변환 완료")

    # 3. RRF 융합
    # id 기준으로 결과 통합
    # 성능 개선: 순위를 딕셔너리로 미리 매핑 (O(1) lookup)

    # fts_ids = fts_results
    fts_ids = vector_results
    vector_ids = vector_results
    fts_rank_map = {id_: rank + 1 for rank, id_ in enumerate(fts_ids)}
    vector_rank_map = {id_: rank + 1 for rank, id_ in enumerate(vector_ids)}

    all_ids = list(set(fts_ids + vector_ids))

    # RRF 점수 계산
    rrf_input = []
    k = 60  # RRF 상수 (표준값)

    for id_ in all_ids:
        # 딕셔너리 lookup으로 순위 가져오기 (O(1))
        fts_rank = fts_rank_map.get(id_, len(fts_ids) + 1)
        vector_rank = vector_rank_map.get(id_, len(vector_ids) + 1)
        
        # 표준 RRF 공식: 1/(k + rank)
        rrf_score = (1.0 / (k + fts_rank)) + (1.0 / (k + vector_rank))
        
        rrf_input.append({
            "id": id_, 
            "rrf_score": rrf_score
        })

    # RRF 점수 기준 정렬 (높은 점수 우선)
    rrf_sorted = sorted(rrf_input, key=lambda x: x["rrf_score"], reverse=True)
    top_panels = rrf_sorted[:min(item.count * 2, len(rrf_sorted))]
    # for p in top_panels :
    #     print(p["id"])

    # 4. Sonnet으로 패널 필터링
    ## json_data_table에서 패널별 json데이터를 sonnet에 함께 넘긴다. 반환 값은 id만 받고, 
    ### 다시 sql id검색을 통해 온전한 json형태로 클라이언트에게 반환한다.
    panel_ids = [p["id"] for p in top_panels]
    panel_jsons = get_jsons_by_ids(panel_ids)
    print("complete : panel_jsons ")

    final_ids = llm_filter_panel(clean_query, panel_jsons).strip().split(' ')
    answers = get_jsons_by_ids(final_ids)
    for p in answers :
        print(p)

    # 5. 클라이언트에게 결과 반환
    t1 = time.time()
    return {
        "original_query": item.query,
        "clean_query": clean_query,
        "result": answers,
        "metrics": {
            "total_time": f"{t1-t0:.2f}s",
            "final_count": len(answers)
        }
    }
    
    # 결과 콘솔에 출력
    for p in answers :
        print(f"{p.id} : {p.info_text}")
    print(len(p))

@app.get("/")
def root():
    return {"message": "RRF Search API with Sonnet running"}
