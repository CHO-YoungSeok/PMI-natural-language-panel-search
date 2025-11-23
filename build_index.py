"""
BM25 인덱스 빌더

이 스크립트는 데이터베이스에서 전체 문서를 읽어서 BM25 인덱스를 구축합니다.
한국어 형태소 분석과 전처리를 수행하여 검색 품질을 향상시킵니다.

사용법:
    python build_index.py

출력:
    bm25_index.pkl - BM25 인덱스와 문서 ID 매핑이 저장된 pickle 파일

주의사항:
    - 데이터베이스에 많은 문서가 있을 경우 시간이 오래 걸릴 수 있습니다
    - 메모리 사용량이 높을 수 있으니 충분한 메모리를 확보하세요
    - 데이터베이스 데이터가 변경되면 이 스크립트를 다시 실행해야 합니다
"""

import sys
import os
import pickle
import time
from typing import List, Tuple

# 현재 디렉토리의 db_search 모듈 임포트
from db_search import get_json_db_conn, preprocess_text

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("✗ rank_bm25 패키지가 설치되어 있지 않습니다.")
    print("  설치: pip install rank-bm25")
    sys.exit(1)

try:
    from kiwipiepy import Kiwi
except ImportError:
    print("✗ kiwipiepy 패키지가 설치되어 있지 않습니다.")
    print("  설치: pip install kiwipiepy")
    sys.exit(1)


def build_bm25_index(batch_size=5000) -> Tuple:
    """
    데이터베이스에서 배치 단위로 문서를 읽어 BM25 인덱스 구축 (메모리 효율적)

    Args:
        batch_size: 한 번에 처리할 문서 수 (기본값: 5000)

    Returns:
        Tuple: (bm25, doc_ids) - BM25 인덱스 객체와 문서 ID 리스트
    """
    print("=" * 80)
    print("BM25 인덱스 구축 시작 (배치 처리 모드)")
    print("=" * 80)
    print(f"배치 크기: {batch_size}개 문서")

    # 1. 데이터베이스 연결
    print("\n[1/5] 데이터베이스 연결 중...")
    try:
        conn = get_json_db_conn()
        cur = conn.cursor()
        print("✓ 데이터베이스 연결 성공")
    except Exception as e:
        print(f"✗ 데이터베이스 연결 실패: {e}")
        sys.exit(1)

    # 2. 전체 문서 개수 확인
    print("\n[2/5] 전체 문서 개수 확인 중...")
    try:
        cur.execute("SELECT COUNT(*) FROM wellcome1st_json_data")
        total_docs = cur.fetchone()[0]
        print(f"✓ 총 {total_docs}개 문서")
        print(f"  예상 배치 수: {(total_docs + batch_size - 1) // batch_size}개")
    except Exception as e:
        print(f"✗ 문서 개수 확인 실패: {e}")
        cur.close()
        conn.close()
        sys.exit(1)

    # 3. 배치 단위로 문서 처리
    print("\n[3/5] 배치 단위로 문서 전처리 중...")
    print("  (메모리 효율적으로 처리합니다...)")

    tokenized_docs = []
    doc_ids = []
    start_time = time.time()
    offset = 0
    total_processed = 0

    while offset < total_docs:
        try:
            # 배치 단위로 문서 조회
            cur.execute(
                "SELECT id, info_text FROM wellcome1st_json_data ORDER BY id LIMIT %s OFFSET %s",
                (batch_size, offset)
            )
            batch = cur.fetchall()

            if not batch:
                break

            # 배치 처리
            for doc_id, text in batch:
                tokens = preprocess_text(text)

                if tokens:
                    tokenized_docs.append(tokens)
                    doc_ids.append(doc_id)

                total_processed += 1

                # 진행률 표시 (1000개마다)
                if total_processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    docs_per_sec = total_processed / elapsed
                    remaining = (total_docs - total_processed) / docs_per_sec if docs_per_sec > 0 else 0
                    memory_mb = len(tokenized_docs) * 0.01  # 대략적인 메모리 사용량 추정

                    print(f"  진행: {total_processed}/{total_docs} "
                          f"({total_processed / total_docs * 100:.1f}%) "
                          f"[{docs_per_sec:.1f} docs/sec, "
                          f"남은시간: {remaining/60:.1f}분, "
                          f"메모리: ~{memory_mb:.0f}MB]")

            offset += batch_size

        except Exception as e:
            print(f"✗ 배치 처리 실패 (offset={offset}): {e}")
            cur.close()
            conn.close()
            sys.exit(1)

    cur.close()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n✓ 전처리 완료: {len(tokenized_docs)}개 문서 인덱싱 ({elapsed:.1f}초 소요)")

    # 제외된 문서 수 확인
    excluded = total_processed - len(tokenized_docs)
    if excluded > 0:
        print(f"  ⚠ {excluded}개 문서는 전처리 후 빈 토큰으로 제외되었습니다.")

    # 4. BM25 인덱스 생성
    print("\n[4/5] BM25 인덱스 생성 중...")
    try:
        start_time = time.time()
        bm25 = BM25Okapi(tokenized_docs)
        elapsed = time.time() - start_time
        print(f"✓ BM25 인덱스 생성 완료 ({elapsed:.2f}초 소요)")
    except Exception as e:
        print(f"✗ BM25 인덱스 생성 실패: {e}")
        print(f"  메모리 부족일 수 있습니다. batch_size를 줄여보세요 (현재: {batch_size})")
        sys.exit(1)

    return bm25, doc_ids


def save_index(bm25, doc_ids, output_file='bm25_index.pkl'):
    """
    BM25 인덱스를 pickle 파일로 저장

    Args:
        bm25: BM25 인덱스 객체
        doc_ids: 문서 ID 리스트
        output_file: 출력 파일 경로
    """
    print(f"\n인덱스 저장 중: {output_file}")

    try:
        with open(output_file, 'wb') as f:
            pickle.dump((bm25, doc_ids), f)

        # 파일 크기 확인
        file_size = os.path.getsize(output_file)
        size_mb = file_size / (1024 * 1024)

        print(f"✓ 인덱스 저장 완료")
        print(f"  파일 크기: {size_mb:.2f} MB")
        print(f"  저장 위치: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"✗ 인덱스 저장 실패: {e}")
        sys.exit(1)


def verify_index(index_file='bm25_index.pkl'):
    """
    저장된 인덱스 파일 검증

    Args:
        index_file: 검증할 인덱스 파일 경로
    """
    print(f"\n인덱스 파일 검증 중: {index_file}")

    try:
        with open(index_file, 'rb') as f:
            bm25, doc_ids = pickle.load(f)

        print(f"✓ 인덱스 로드 성공")
        print(f"  인덱싱된 문서 수: {len(doc_ids)}개")
        print(f"  샘플 문서 ID: {doc_ids[:5]}")

        # 간단한 검색 테스트
        from db_search import preprocess_text

        test_query = "운동 좋아하는 20대"
        query_tokens = preprocess_text(test_query)

        if query_tokens:
            scores = bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)),
                               key=lambda i: scores[i],
                               reverse=True)[:3]

            print(f"\n테스트 쿼리: '{test_query}'")
            print(f"전처리된 토큰: {query_tokens}")
            print(f"상위 3개 결과:")
            for i, idx in enumerate(top_indices, 1):
                print(f"  {i}. ID: {doc_ids[idx]}, Score: {scores[idx]:.4f}")

            print("\n✓ 인덱스 검증 완료: 정상 작동")
        else:
            print("⚠ 테스트 쿼리 전처리 결과가 비어있습니다.")

    except Exception as e:
        print(f"✗ 인덱스 검증 실패: {e}")
        sys.exit(1)


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print(" BM25 인덱스 빌더 for RRF Search (메모리 효율 모드)")
    print("=" * 80)

    # 배치 크기 설정 (메모리에 따라 조정 가능)
    # 작은 메모리 환경: 1000-2000
    # 일반 환경: 5000 (기본값)
    # 충분한 메모리: 10000+
    batch_size = 5000

    # 명령줄 인자로 배치 크기 지정 가능
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
            print(f"사용자 지정 배치 크기: {batch_size}")
        except ValueError:
            print(f"⚠ 잘못된 배치 크기: {sys.argv[1]}, 기본값({batch_size}) 사용")

    start_total = time.time()

    # 1. 인덱스 구축
    bm25, doc_ids = build_bm25_index(batch_size=batch_size)

    # 2. 인덱스 저장
    save_index(bm25, doc_ids)

    # 3. 인덱스 검증
    verify_index()

    elapsed_total = time.time() - start_total

    print("\n" + "=" * 80)
    print(f" 전체 작업 완료 (총 소요시간: {elapsed_total/60:.2f}분)")
    print("=" * 80)
    print("\n다음 단계:")
    print("  1. FastAPI 서버를 실행하세요: uvicorn search:app --reload")
    print("  2. 또는 테스트 스크립트 실행: python test_bm25.py")
    print("\n메모리 부족 시:")
    print("  - 배치 크기를 줄여서 실행: python build_index.py 2000")
    print("  - 또는: python build_index.py 1000")
    print("\n주의:")
    print("  - 데이터베이스 데이터가 변경되면 이 스크립트를 다시 실행하세요")
    print("  - 주기적으로 인덱스를 재구축하여 최신 상태를 유지하세요")
    print("")


if __name__ == "__main__":
    main()
