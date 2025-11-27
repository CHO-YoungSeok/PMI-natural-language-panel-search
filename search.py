from fastapi import FastAPI
from pydantic import BaseModel
import sys, os, time, re, json
from fastapi.middleware.cors import CORSMiddleware
from db_search import bm25_search, vector_search, get_jsons_by_ids, has_field_info, extract_regions_from_query, extract_gender_from_query
from sonnet_api import preprocess_query, llm_filter_panel
from query_vectorizer import get_query_vector
import numpy as np
from dotenv import load_dotenv
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from io import StringIO
from contextlib import redirect_stdout

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
    count: int = 150  # RRF를 통해 산출할 개수 >  min(result_count, 150)
    k: int = 60  # RRF 상수 (기본값: 60)

@app.post("/ask")
def search_pipeline(item: QueryItem):
    """
    하이브리드 검색 파이프라인 (BM25 + Vector Search + RRF) - 멀티스레드 최적화

    1. Sonnet으로 쿼리 전처리 + 결과 개수 추출 (병렬)
    2. BM25 검색 + 벡터 검색 (병렬)
    3. RRF 융합으로 두 검색 결과 통합
    4. Sonnet으로 최종 패널 필터링
    5. JSON 반환
    """
    # Console 출력 캡처 시작
    console_log = StringIO()

    kst_time = datetime.now(ZoneInfo("Asia/Seoul"))
    log_line = "========================================"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"{kst_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} : 검색 시작"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = "========================================"
    print(log_line)
    console_log.write(log_line + "\n")

    t0 = time.time()

    # 1단계: Sonnet 전처리 (병렬 처리: 쿼리 정제 + 개수 추출 + 출생년도 추출)
    clean_query, result_count, birth_years = preprocess_query(item.query)

    # 거주지역 및 성별 추출 (clean_query 기반, 키워드 매칭)
    regions = extract_regions_from_query(clean_query)
    genders = extract_gender_from_query(clean_query)

    # RRF count 동적 계산: min(result_count * 3, item.count)
    rrf_count = min(result_count * 3, item.count)

    log_line = f"{time.time() - t0:.1f}s, [1단계 완료] clean query: {clean_query}\nresult count: {result_count}\nbirth years: {birth_years if birth_years is not None else 'None (no age filter)'}\nregions: {regions if regions is not None else 'None (no region filter)'}\ngenders: {genders if genders is not None else 'None (no gender filter)'}\nRRF count: {rrf_count}"
    print(log_line)
    console_log.write(log_line + "\n")

    # ⭐ 2단계 병렬화: BM25 검색 + 벡터 검색 동시 실행
    t01 = time.time()

    def vector_search_with_embedding(query, birth_years, regions, genders):
        """쿼리 벡터화 + 검색을 하나의 함수로 묶음"""
        query_vec = get_query_vector(query)
        return vector_search(query_vec, top_k=36000, birth_years=birth_years,
                           regions=regions, genders=genders)

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 두 검색 작업을 동시에 제출 (출생년도/거주지역/성별 필터링 적용)
        future_bm25 = executor.submit(bm25_search, clean_query, 36000, birth_years, regions, genders)
        future_vector = executor.submit(vector_search_with_embedding, clean_query, birth_years, regions, genders)

        # 모든 작업이 완료될 때까지 대기
        concurrent.futures.wait([future_bm25, future_vector])

        # 결과 가져오기
        bm25_results_ids = future_bm25.result()
        vector_results_ids = future_vector.result()

    log_line = "----------------------------------------------"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"{time.time() - t01:.1f}s, [병렬 2단계 완료] BM25 + 벡터 검색"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"bm25_result (BM25 기반) 완료: {len(bm25_results_ids)}개"
    print(log_line)
    console_log.write(log_line + "\n")

    bm25_top30 = get_jsons_by_ids(bm25_results_ids[:30])
    log_line = json.dumps(bm25_top30, indent=2, ensure_ascii=False)
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = "----------------------------------------------"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"vector_result (벡터 기반) 완료: {len(vector_results_ids)}개"
    print(log_line)
    console_log.write(log_line + "\n")

    vector_top30 = get_jsons_by_ids(vector_results_ids[:30])
    log_line = json.dumps(vector_top30, indent=2, ensure_ascii=False)
    print(log_line)
    console_log.write(log_line + "\n")

    # 2.5. 도메인 키워드 기반 필터링 (음용경험/OTT)
    # 쿼리에서 특정 도메인 키워드 감지
    domain_filters = []
    if any(keyword in clean_query for keyword in ['OTT', '넷플릭스', '웨이브', '티빙', '왓챠', '쿠팡플레이']):
        domain_filters.append('여러분이 현재 이용 중인 OTT 서비스는 몇 개인가요?')
    if any(keyword in clean_query for keyword in ['음용', '술', '맥주', '소주', '음주', '와인', '위스키']):
        domain_filters.append('음용경험 술')    

    # 도메인 필터 적용
    if domain_filters:
        log_line = "----------------------------------------------"
        print(log_line)
        console_log.write(log_line + "\n")

        log_line = f"도메인 키워드 감지: {domain_filters}"
        print(log_line)
        console_log.write(log_line + "\n")

        # BM25 결과 필터링
        bm25_results_ids = has_field_info(bm25_results_ids, domain_filters)
        log_line = f"BM25 필터링 후: {len(bm25_results_ids)}개"
        print(log_line)
        console_log.write(log_line + "\n")

        # Vector 결과 필터링
        vector_results_ids = has_field_info(vector_results_ids, domain_filters)
        log_line = f"Vector 필터링 후: {len(vector_results_ids)}개"
        print(log_line)
        console_log.write(log_line + "\n")

    # 3. RRF 융합
    bm25_ids = bm25_results_ids
    vector_ids = vector_results_ids
    bm25_rank_map = {id_: rank + 1 for rank, id_ in enumerate(bm25_ids)}
    vector_rank_map = {id_: rank + 1 for rank, id_ in enumerate(vector_ids)}

    all_ids = list(set(bm25_ids + vector_ids))
    log_line = "----------------------------------------------"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"rrf ALL list set : {len(all_ids)}개"
    print(log_line)
    console_log.write(log_line + "\n")

    # RRF 점수 계산
    rrf_input = []
    k = item.k  # RRF 상수 (파라미터로 전달됨)

    log_line = f"RRF k 값: {k}"
    print(log_line)
    console_log.write(log_line + "\n")

    for id_ in all_ids:
        bm25_rank = bm25_rank_map.get(id_, len(bm25_ids) + 1)
        vector_rank = vector_rank_map.get(id_, len(vector_ids) + 1)
        rrf_score = (1.0 / (k + bm25_rank)) + (1.0 / (k + vector_rank))

        rrf_input.append({
            "id": id_,
            "rrf_score": rrf_score
        })

    # RRF 점수 기준 정렬 (높은 점수 우선)
    rrf_sorted = sorted(rrf_input, key=lambda x: x["rrf_score"], reverse=True)
    top_panels = rrf_sorted[:min(rrf_count, len(rrf_sorted))]
    log_line = "----------------------------------------------"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"rrf 완료 : {len(top_panels)}개"
    print(log_line)
    console_log.write(log_line + "\n")

    rrf_top30 = get_jsons_by_ids([panel['id'] for panel in top_panels[:30]])
    log_line = json.dumps(rrf_top30, indent=2, ensure_ascii=False)
    print(log_line)
    console_log.write(log_line + "\n")

    # 4. result_count에 따라 LLM 필터링 수행 여부 결정
    log_line = "----------------------------------------------"
    print(log_line)
    console_log.write(log_line + "\n")

    if result_count > 100:
        # result_count > 100: LLM 필터링 건너뛰고 RRF 결과 바로 사용
        log_line = f"[LLM 필터링 건너뛰기] result_count({result_count}) > 100 → RRF 결과 직접 반환"
        print(log_line)
        console_log.write(log_line + "\n")

        # RRF 결과에서 result_count만큼 가져오기
        final_ids = [p['id'] for p in top_panels[:result_count]]

        log_line = f"RRF 상위 {len(final_ids)}개 패널 선택 완료"
        print(log_line)
        console_log.write(log_line + "\n")

    else:
        # result_count <= 100: 기존 LLM 필터링 로직 수행
        log_line = f"[LLM 필터링 진행] result_count({result_count}) ≤ 100"
        print(log_line)
        console_log.write(log_line + "\n")

        panel_ids = [p["id"] for p in top_panels]
        panel_jsons = get_jsons_by_ids(panel_ids)
        log_line = f"{time.time() - t01:.1f}s, complete : panel_jsons"
        print(log_line)
        console_log.write(log_line + "\n")

        llm_result = llm_filter_panel(clean_query, panel_jsons).strip()
        # 빈 결과 처리: 빈 문자열이면 빈 리스트, 아니면 split
        final_ids = llm_result.split() if llm_result else []

        if not final_ids:
            log_line = "[경고] LLM이 빈 결과를 반환했습니다."
            print(log_line)
            console_log.write(log_line + "\n")

        # [추가] 결과 부족 시 RRF 점수로 보충
        if len(final_ids) < result_count:
            shortage = result_count - len(final_ids)
            log_line = f"[보충 필요] LLM 결과 {len(final_ids)}개 / 요청 {result_count}개 → {shortage}개 부족"
            print(log_line)
            console_log.write(log_line + "\n")

            # 이미 선택된 ID 제외하고 RRF 점수 순으로 추가
            remaining_panels = [p for p in top_panels if p['id'] not in final_ids]
            additional_ids = [p['id'] for p in remaining_panels[:shortage]]
            final_ids.extend(additional_ids)

            log_line = f"[보충 완료] {len(additional_ids)}개 추가 → 총 {len(final_ids)}개"
            print(log_line)
            console_log.write(log_line + "\n")

    answers_full = get_jsons_by_ids(final_ids)

    # 결과 콘솔에 출력 (전체 결과)
    log_line = "----------------------------------------------"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = f"최종 검색 결과 (전체) : {len(answers_full)}개"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = json.dumps(answers_full, indent=2, ensure_ascii=False)
    print(log_line)
    console_log.write(log_line + "\n")

    # 사용자 요청 개수만큼 제한 (API 응답용)
    answers = answers_full[:min(result_count, len(answers_full))]

    log_line = f"\n사용자에게 반환되는 결과 : {len(answers)}개 (요청 개수: {result_count})"
    print(log_line)
    console_log.write(log_line + "\n")

    # 5. 클라이언트에게 결과 반환
    t1 = time.time()

    # 최종 시간 로깅
    log_line = f"\n전체 소요 시간: {t1-t0:.2f}s"
    print(log_line)
    console_log.write(log_line + "\n")

    log_line = "======================================== 검색 완료 ========================================\n"
    print(log_line)
    console_log.write(log_line + "\n")

    # Console 로그를 파일에 저장 (최근 검색이 상단에 오도록)
    try:
        results_file = "search_results.log"
        current_log = console_log.getvalue()

        # 기존 파일 내용 읽기
        existing_content = ""
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()

        # 최근 로그를 상단에 추가 (prepend)
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(current_log)
            if existing_content:
                f.write("\n" + existing_content)

        print(f"검색 결과가 {results_file}에 저장되었습니다.")
    except Exception as e:
        print(f"로그 저장 중 오류 발생: {e}")

    return {
        "original_query": item.query,
        "clean_query": clean_query,
        "result": answers,
        "console_log": console_log.getvalue(),
        "metrics": {
            "total_time": f"{t1-t0:.2f}s",
            "final_count": len(answers)
        }
    }
    
@app.get("/")
def root():
    return {"message": "RRF Search API with Sonnet running (Multi-threaded)"}
