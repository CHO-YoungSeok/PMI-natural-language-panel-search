"""
RRF k-value Performance Testing Script

This script automates testing of different RRF k values with multiple queries.
Results are saved to testResult/ directory with separate files for BM25, Vector, and RRF results.

Test configuration:
- 4 test queries
- k values: [60]
- Total experiments: 4

Output includes:
- BM25 top 100 results (separate file)
- Vector top 100 results (separate file)
- RRF top 100 results (separate file)
- Each query has its own set of files
"""
  
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from db_search import bm25_search, vector_search, get_jsons_by_ids
from query_vectorizer import get_query_vector
from sonnet_api import preprocess_query

# Test configuration
OUTPUT_DIR = "testResult"

TEST_QUERIES = [
    "서울 20대 남자 100명",
    "경기 30~40대 남자 술을 먹은 사람 50명",
    "서울, 경기 OTT 이용하는 젊은층 30명",
    "고등학교 이하 학력 중 휴대폰이 아이폰인 사람 30명"
]

K_VALUES = [60]


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}/")


def perform_search(query: str, k: int) -> dict:
    """
    Perform hybrid search and return BM25, Vector, and RRF results.

    Args:
        query: Search query string
        k: RRF k value

    Returns:
        dict: Search results containing bm25_ids, vector_ids, rrf_ids
    """
    # Preprocess query
    clean_query, _, _ = preprocess_query(query)
    print(f"    Clean query: {clean_query}")

    # Perform BM25 search (top 36000)
    print(f"    Performing BM25 search...")
    bm25_results_ids = bm25_search(clean_query, 36000)

    # Perform Vector search (top 36000)
    print(f"    Performing Vector search...")
    query_vec = get_query_vector(clean_query)
    vector_results_ids = vector_search(query_vec, top_k=36000)

    # Perform RRF fusion
    print(f"    Performing RRF fusion with k={k}...")
    bm25_ids = bm25_results_ids
    vector_ids = vector_results_ids
    bm25_rank_map = {id_: rank + 1 for rank, id_ in enumerate(bm25_ids)}
    vector_rank_map = {id_: rank + 1 for rank, id_ in enumerate(vector_ids)}

    all_ids = list(set(bm25_ids + vector_ids))

    # RRF score calculation
    rrf_input = []
    for id_ in all_ids:
        bm25_rank = bm25_rank_map.get(id_, len(bm25_ids) + 1)
        vector_rank = vector_rank_map.get(id_, len(vector_ids) + 1)
        rrf_score = (1.0 / (k + bm25_rank)) + (1.0 / (k + vector_rank))

        rrf_input.append({
            "id": id_,
            "rrf_score": rrf_score
        })

    # Sort by RRF score (highest first)
    rrf_sorted = sorted(rrf_input, key=lambda x: x["rrf_score"], reverse=True)
    rrf_ids = [item["id"] for item in rrf_sorted]

    return {
        "clean_query": clean_query,
        "bm25_ids": bm25_results_ids,
        "vector_ids": vector_results_ids,
        "rrf_ids": rrf_ids
    }


def save_results(query_idx: int, k: int, original_query: str, result: dict):
    """
    Save BM25, Vector, and RRF results to separate JSON files.

    Args:
        query_idx: Query index (0, 1, 2, 3)
        k: RRF k value
        original_query: Original user query
        result: Search results dict containing bm25_ids, vector_ids, rrf_ids
    """
    kst_time = datetime.now(ZoneInfo("Asia/Seoul"))

    # Common metadata
    metadata = {
        "timestamp": kst_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        "query_index": query_idx,
        "k_value": k,
        "original_query": original_query,
        "clean_query": result.get("clean_query", "")
    }

    # Save BM25 top 100 results
    bm25_top100_ids = result["bm25_ids"][:100]
    bm25_data = {
        **metadata,
        "search_type": "BM25",
        "total_results": len(result["bm25_ids"]),
        "top_100_count": len(bm25_top100_ids),
        "results": get_jsons_by_ids(bm25_top100_ids)
    }
    bm25_filename = f"query{query_idx}_k{k}_bm25.json"
    with open(os.path.join(OUTPUT_DIR, bm25_filename), 'w', encoding='utf-8') as f:
        json.dump(bm25_data, f, ensure_ascii=False, indent=2)
    print(f"    → Saved: {bm25_filename}")

    # Save Vector top 100 results
    vector_top100_ids = result["vector_ids"][:100]
    vector_data = {
        **metadata,
        "search_type": "Vector",
        "total_results": len(result["vector_ids"]),
        "top_100_count": len(vector_top100_ids),
        "results": get_jsons_by_ids(vector_top100_ids)
    }
    vector_filename = f"query{query_idx}_k{k}_vector.json"
    with open(os.path.join(OUTPUT_DIR, vector_filename), 'w', encoding='utf-8') as f:
        json.dump(vector_data, f, ensure_ascii=False, indent=2)
    print(f"    → Saved: {vector_filename}")

    # Save RRF top 100 results
    rrf_top100_ids = result["rrf_ids"][:100]
    rrf_data = {
        **metadata,
        "search_type": "RRF",
        "total_results": len(result["rrf_ids"]),
        "top_100_count": len(rrf_top100_ids),
        "results": get_jsons_by_ids(rrf_top100_ids)
    }
    rrf_filename = f"query{query_idx}_k{k}_rrf.json"
    with open(os.path.join(OUTPUT_DIR, rrf_filename), 'w', encoding='utf-8') as f:
        json.dump(rrf_data, f, ensure_ascii=False, indent=2)
    print(f"    → Saved: {rrf_filename}")


def run_tests():
    """Run all test combinations and save results."""
    print("=" * 80)
    print("RRF k-value Performance Testing")
    print("=" * 80)
    print(f"Test queries: {len(TEST_QUERIES)}")
    print(f"K values: {K_VALUES}")
    print(f"Total experiments: {len(TEST_QUERIES) * len(K_VALUES)}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("=" * 80)

    ensure_output_dir()

    total_tests = len(TEST_QUERIES) * len(K_VALUES)
    completed = 0

    # 전체 테스트 결과를 저장할 리스트
    all_results = []
    test_start_time = time.time()

    for query_idx, query in enumerate(TEST_QUERIES):
        print(f"\n[Query {query_idx}] {query}")

        for k in K_VALUES:
            completed += 1
            print(f"  [{completed}/{total_tests}] Testing k={k}...")

            try:
                start_time = time.time()
                result = perform_search(query, k)
                elapsed_time = time.time() - start_time

                save_results(query_idx, k, query, result)

                # Print summary
                bm25_count = len(result.get("bm25_ids", []))
                vector_count = len(result.get("vector_ids", []))
                rrf_count = len(result.get("rrf_ids", []))

                print(f"    ✓ BM25: {bm25_count}개, Vector: {vector_count}개, RRF: {rrf_count}개")
                print(f"    ✓ 소요시간: {elapsed_time:.2f}s")

                # 결과 수집
                all_results.append({
                    "query_index": query_idx,
                    "original_query": query,
                    "clean_query": result.get("clean_query", ""),
                    "k_value": k,
                    "bm25_count": bm25_count,
                    "vector_count": vector_count,
                    "rrf_count": rrf_count,
                    "elapsed_time": f"{elapsed_time:.2f}s",
                    "status": "success"
                })

                # 마지막 테스트가 아니면 60초(1분) 대기
                if completed < total_tests:
                    print(f"  ⏳ 다음 테스트까지 60초(1분) 대기 중...")
                    time.sleep(60)

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                import traceback
                traceback.print_exc()

                # 실패한 결과도 기록
                all_results.append({
                    "query_index": query_idx,
                    "original_query": query,
                    "k_value": k,
                    "status": "failed",
                    "error": str(e)
                })
                continue

    # 전체 테스트 소요 시간
    total_elapsed_time = time.time() - test_start_time

    # 최종 결과 요약 저장
    kst_time = datetime.now(ZoneInfo("Asia/Seoul"))
    summary = {
        "test_info": {
            "timestamp": kst_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "total_queries": len(TEST_QUERIES),
            "k_values": K_VALUES,
            "total_experiments": total_tests,
            "completed_experiments": len([r for r in all_results if r["status"] == "success"]),
            "failed_experiments": len([r for r in all_results if r["status"] == "failed"]),
            "total_elapsed_time": f"{total_elapsed_time:.2f}s"
        },
        "queries": TEST_QUERIES,
        "results": all_results
    }

    summary_filename = f"test_summary_{kst_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(OUTPUT_DIR, summary_filename), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Testing complete! Results saved to {OUTPUT_DIR}/")
    print(f"Summary saved: {summary_filename}")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
