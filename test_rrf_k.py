"""
RRF k-value Performance Testing Script - API Call Version

Tests the search API endpoint (/ask) with multiple queries.
Results are saved to testResult/ directory.

Test configuration:
- 4 test queries
- k values: [60]
- Uses FastAPI endpoint instead of direct function calls

Output includes:
- Full API responses with search results
- Summary report with metrics
"""

import json
import os
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# Test configuration
OUTPUT_DIR = "testResult"
API_URL = "http://localhost:8000/ask"  # FastAPI endpoint

TEST_QUERIES = [
    # "서울 20대 남자 100명 ",
    # "경기 30~40대 남자 술을 먹은 사람 50명 ",
    # "서울, 경기 OTT 이용하는 젊은층 30명 ",
    # "고등학교 이하 학력 중 월 소득이 300만원 이상이면서 휴대폰이 아이폰인 사람 30명",
    "서울 에 살 규 이쓰며서, 30대 남졍이고, 아이폰을 쓰는 사람 50명"
    # "경기 30~40대 남자 술을 먹은 사람 100명",
    # "서울 50대 이상 남자 중 술을 안먹는 사람 50명",
    # "서울, 경기 OTT 이용하는 젊은층 30명",
    # "OTT 이용하지 않는 젊은층 30명",
    # "애플(아이폰) 사용자 중 20대 여성 30명",
    # "전업주부이면서 자녀가 있는 사람 50명",
    # "대졸 이상 학력의 미혼 남성 40명",
    # "월 개인소득 300만원 이상인 직장인 45명",
    # "서울 에 살 규 이쓰며서, 30대 남졍이고, 한달에 소득이 500만원 이하인 샤람",
    # "경기도에 살고 기혼이며 자녀가 있고 월 가구소득 800만원 이상인 사람 30명",
    # "20대 미혼 여성 중 대학생이고 혼자 거주하며 아이폰을 사용하는 사람 20명",
    # "부산 거주, 40대 기혼 남성, 차량 보유자이며 월 가구소득 200만원 ~ 600만원 사이인 사람 50명",
    #  "IT 직무 종사자 중 월 개인소득 500만원 이상이고 미혼인 사람 18명",
]

K_VALUES = [60]


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def call_search_api(query: str, count: int, k: int) -> dict:
    """
    Call the search API endpoint.

    Args:
        query: Search query string
        count: Number of results to return
        k: RRF k value

    Returns:
        dict: API response containing results
    """
    payload = {
        "query": query,
        "count": count,
        "k": k
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()


def save_results(query_idx: int, k: int, original_query: str, api_response: dict):
    """
    Save API response to JSON file.

    Args:
        query_idx: Query index
        k: RRF k value
        original_query: Original user query
        api_response: Full API response
    """
    kst_time = datetime.now(ZoneInfo("Asia/Seoul"))

    # Remove console_log from saved data (keep for terminal output only)
    api_response_clean = {
        key: value for key, value in api_response.items()
        if key != "console_log"
    }

    # Add test metadata
    output_data = {
        "test_metadata": {
            "timestamp": kst_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "query_index": query_idx,
            "k_value": k,
            "original_query": original_query,
        },
        "api_response": api_response_clean
    }

    # Save to file
    filename = f"query{query_idx}_k{k}_result.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return filename


def run_tests():
    """Run all test combinations and save results."""
    print("=" * 80)
    print("RRF API Testing")
    print("=" * 80)
    print(f"API Endpoint: {API_URL}")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print(f"K values: {K_VALUES}")
    print(f"Total experiments: {len(TEST_QUERIES) * len(K_VALUES)}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("=" * 80)

    ensure_output_dir()

    total_tests = len(TEST_QUERIES) * len(K_VALUES)
    completed = 0
    all_results = []
    test_start_time = time.time()

    for query_idx, query in enumerate(TEST_QUERIES):
        print(f"\n[Query {query_idx}] {query}")

        for k in K_VALUES:
            completed += 1
            print(f"  [{completed}/{total_tests}] Testing k={k}...")

            try:
                start_time = time.time()

                # Call API endpoint
                api_response = call_search_api(query, count=150, k=k)

                elapsed_time = time.time() - start_time

                # Save results
                filename = save_results(query_idx, k, query, api_response)

                # Extract summary info
                result_count = len(api_response.get("result", []))
                metrics = api_response.get("metrics", {})

                print(f"    ✓ Results: {result_count}개")
                print(f"    ✓ API Time: {metrics.get('total_time', 'N/A')}")
                print(f"    ✓ Saved: {filename}")

                # Collect summary
                all_results.append({
                    "query_index": query_idx,
                    "original_query": query,
                    "clean_query": api_response.get("clean_query", ""),
                    "k_value": k,
                    "result_count": result_count,
                    "api_time": metrics.get('total_time', 'N/A'),
                    "total_elapsed": f"{elapsed_time:.2f}s",
                    "status": "success",
                    "filename": filename
                })

                # Wait between tests (except for the last one)
                if completed < total_tests:
                    wait_time = 30
                    print(f"  ⏳ Waiting {wait_time}s before next test...")
                    time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                print(f"    ✗ API Error: {e}")
                all_results.append({
                    "query_index": query_idx,
                    "original_query": query,
                    "k_value": k,
                    "status": "failed",
                    "error": str(e)
                })
                continue
            except Exception as e:
                print(f"    ✗ Error: {e}")
                all_results.append({
                    "query_index": query_idx,
                    "original_query": query,
                    "k_value": k,
                    "status": "failed",
                    "error": str(e)
                })
                continue

    # Calculate total test time
    total_elapsed_time = time.time() - test_start_time

    # Save summary report
    kst_time = datetime.now(ZoneInfo("Asia/Seoul"))
    summary = {
        "test_info": {
            "timestamp": kst_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "api_endpoint": API_URL,
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
    summary_filepath = os.path.join(OUTPUT_DIR, summary_filename)

    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Testing complete! Results saved to {OUTPUT_DIR}/")
    print(f"Summary: {summary_filename}")
    print(f"Success: {summary['test_info']['completed_experiments']}/{total_tests}")
    print(f"Failed: {summary['test_info']['failed_experiments']}/{total_tests}")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
