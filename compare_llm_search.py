"""
LLM 모델 비교용 검색 파이프라인 (search.py와 동일한 로직)
Sonnet vs Gemini 성능 비교
"""
import sys
import os
import time
import json
from dotenv import load_dotenv
from db_search import bm25_search, vector_search, get_jsons_by_ids
from query_vectorizer import get_query_vector

# LLM API imports
import sonnet_api
import gemini_api

# 환경변수 로드
load_dotenv()


def search_pipeline(query: str, llm_module, llm_name: str, count: int = 150):
    """
    하이브리드 검색 파이프라인 (search.py와 동일한 로직)

    Args:
        query: 검색 쿼리
        llm_module: LLM 모듈 (sonnet_api 또는 gemini_api)
        llm_name: LLM 이름 (출력용)
        count: RRF 후보 패널 개수

    Returns:
        dict: 검색 결과
    """
    print(f"\n{'='*80}")
    print(f"  [{llm_name}] 검색 시작")
    print(f"{'='*80}")

    t0 = time.time()

    # 1. 쿼리 전처리
    clean_query = llm_module.preprocess_query(query)
    print(f"clean query: {clean_query}")

    # 2. (Python BM25 with 한국어 형태소 분석), 벡터 코사인 유사도 검색
    bm25_results_ids = bm25_search(clean_query, top_k=36000)
    print("----------------------------------------------")
    print(f"bm25_result (BM25 기반) 완료: {len(bm25_results_ids)}개")
    print(get_jsons_by_ids(bm25_results_ids[:3]))

    query_vec = get_query_vector(clean_query)
    vector_results_ids = vector_search(query_vec, top_k=36000)
    print("----------------------------------------------")
    print(f"query_vec 기반 완료 : {len(vector_results_ids)}개")
    print(get_jsons_by_ids(vector_results_ids[:3]))

    # 3. RRF 융합
    # id 기준으로 결과 통합
    # 성능 개선: 순위를 딕셔너리로 미리 매핑 (O(1) lookup)
    bm25_ids = bm25_results_ids
    vector_ids = vector_results_ids
    bm25_rank_map = {id_: rank + 1 for rank, id_ in enumerate(bm25_ids)}
    vector_rank_map = {id_: rank + 1 for rank, id_ in enumerate(vector_ids)}

    all_ids = list(set(bm25_ids + vector_ids))
    print("----------------------------------------------")
    print(f"rrf ALL list set : {len(all_ids)}개")

    # RRF 점수 계산
    rrf_input = []
    k = 40  # RRF 상수 (표준값)

    for id_ in all_ids:
        # 딕셔너리 lookup으로 순위 가져오기 (O(1))
        bm25_rank = bm25_rank_map.get(id_, len(bm25_ids) + 1)
        vector_rank = vector_rank_map.get(id_, len(vector_ids) + 1)

        # 표준 RRF 공식: 1/(k + rank)
        rrf_score = (1.0 / (k + bm25_rank)) + (1.0 / (k + vector_rank))

        rrf_input.append({
            "id": id_,
            "rrf_score": rrf_score
        })

    # RRF 점수 기준 정렬 (높은 점수 우선)
    rrf_sorted = sorted(rrf_input, key=lambda x: x["rrf_score"], reverse=True)
    top_panels = rrf_sorted[:min(count, len(rrf_sorted))]
    print("----------------------------------------------")
    print(f"rrf 완료 : {len(top_panels)}개")
    print(get_jsons_by_ids([panel['id'] for panel in top_panels[:3]]))

    # 4. LLM으로 패널 필터링
    ## json_data_table에서 패널별 json데이터를 LLM에 함께 넘긴다. 반환 값은 id만 받고,
    ### 다시 sql id검색을 통해 온전한 json형태로 클라이언트에게 반환한다.
    panel_ids = [p["id"] for p in top_panels]
    panel_jsons = get_jsons_by_ids(panel_ids)
    print("complete : panel_jsons ")

    final_ids = llm_module.llm_filter_panel(clean_query, panel_jsons).strip().split(' ')
    answers = get_jsons_by_ids(final_ids)

    # 결과 콘솔에 출력
    print("----------------------------------------------")
    print(f"최종 검색 결과 : {len(answers)}개")
    print(json.dumps(answers, ensure_ascii=False, indent=2))

    # 5. 결과 반환
    t1 = time.time()

    result = {
        "llm_name": llm_name,
        "original_query": query,
        "clean_query": clean_query,
        "result": answers,
        "metrics": {
            "total_time": f"{t1-t0:.2f}s",
            "final_count": len(answers)
        }
    }

    print(f"\n[{llm_name}] 총 소요 시간: {t1-t0:.2f}초")
    print(f"{'='*80}\n")

    return result


def compare_models(query: str, count: int = 150):
    """
    Sonnet과 Gemini 모델을 비교하여 검색 수행

    Args:
        query: 검색 쿼리
        count: RRF 후보 패널 개수
    """
    print("\n" + "="*80)
    print("  LLM 모델 비교 검색")
    print("="*80)
    print(f"원본 쿼리: {query}")
    print(f"RRF 후보 개수: {count}")
    print("="*80)

    # Sonnet 검색
    sonnet_result = search_pipeline(query, sonnet_api, "Sonnet", count)

    # Gemini 검색
    gemini_result = search_pipeline(query, gemini_api, "Gemini", count)

    # 최종 요약
    print("\n" + "="*80)
    print("  검색 완료 - 결과 요약")
    print("="*80)
    print(f"\n[Sonnet]")
    print(f"  - 전처리 쿼리: {sonnet_result['clean_query']}")
    print(f"  - 결과 개수: {sonnet_result['metrics']['final_count']}개")
    print(f"  - 소요 시간: {sonnet_result['metrics']['total_time']}")

    print(f"\n[Gemini]")
    print(f"  - 전처리 쿼리: {gemini_result['clean_query']}")
    print(f"  - 결과 개수: {gemini_result['metrics']['final_count']}개")
    print(f"  - 소요 시간: {gemini_result['metrics']['total_time']}")
    print("="*80 + "\n")

    return {
        "sonnet": sonnet_result,
        "gemini": gemini_result
    }


def main():
    """메인 실행 함수"""
    print("="*80)
    print("  LLM 모델 비교 검색 시스템 (Sonnet vs Gemini)")
    print("="*80)
    print("\n종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    while True:
        try:
            # 사용자 입력
            query = input("\n검색 쿼리를 입력하세요: ").strip()

            # 종료 명령
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n프로그램을 종료합니다.")
                break

            # 빈 입력 처리
            if not query:
                print("⚠️  쿼리를 입력해주세요.")
                continue

            # RRF 후보 개수 입력 (선택)
            count_input = input("RRF 후보 개수 (기본값: 150, Enter로 건너뛰기): ").strip()
            count = int(count_input) if count_input.isdigit() else 150

            # 검색 및 비교 수행
            results = compare_models(query, count)

            # JSON 파일로 저장 여부 확인
            save = input("\n결과를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"compare_result_{int(time.time())}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"✅ 결과가 {filename}에 저장되었습니다.")

        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
