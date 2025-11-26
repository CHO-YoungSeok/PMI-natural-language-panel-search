import psycopg2
import numpy as np
import os
import re
import pickle
from typing import List, Dict
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# 프로젝트 .env에 설정한 환경변수들 활성화
load_dotenv()

DB_CONFIG = {
    'host': 'database-1.cpk06802so7a.ap-southeast-2.rds.amazonaws.com',
    'port': 5432,
    'user': 'postgres',
    'password': os.getenv("POSTGRES_DB_PASSWORD"),
    'dbname': 'postgres'
}

def get_json_db_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def get_vector_db_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


# Kiwi 초기화 (전역으로 한 번만 생성)
kiwi = Kiwi()

# 불용어 설정
stopwords = Stopwords()

# 커스텀 불용어 추가 (검색에 불필요한 단어들)
custom_stopwords = [
    ('이', 'JKS'), ('가', 'JKS'), ('을', 'JKO'), ('를', 'JKO'),
    ('은', 'JX'), ('는', 'JX'), ('의', 'JKG'), ('에', 'JKB'),
    ('에서', 'JKB'), ('으로', 'JKB'), ('로', 'JKB'),
    ('이다', 'VCP'), ('하다', 'XSV'), ('있다', 'VV'), ('없다', 'VA'),
    ('것', 'NNB'), ('수', 'NNB'), ('등', 'NNB'), ('중', 'NNB'),
    ('및', 'MAJ'), ('또는', 'MAJ'), ('그리고', 'MAJ'),
]

for word, pos in custom_stopwords:
    stopwords.add((word, pos))


def preprocess_text(text: str) -> List[str]:
    """
    한국어 텍스트를 BM25 검색에 최적화된 토큰으로 변환

    전처리 단계:
    1. 정규화 (특수문자, 이모티콘 제거)
    2. 형태소 분석
    3. 품사 필터링 (명사, 동사, 형용사만 추출)
    4. 불용어 제거
    5. 단일 문자 제거

    Args:
        text: 원본 텍스트

    Returns:
        List[str]: 전처리된 토큰 리스트
    """
    if not text or not isinstance(text, str):
        return []

    # 1. 정규화
    # 특수문자, 이모티콘 제거 (한글, 영문, 숫자만 유지)
    text = re.sub(r'[^가-힣A-Za-z0-9\s]', ' ', text)

    # 연속된 자음/모음 제거 (ㅋㅋㅋ, ㅠㅠ 등)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1{1,}', '', text)

    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return []

    # 2. 형태소 분석 (불용어 자동 제거)
    tokens = kiwi.tokenize(text, stopwords=stopwords)

    # 3. 품사 필터링
    # 의미있는 품사만 추출: 명사(NN*), 동사(VV), 형용사(VA), 영어(SL), 숫자(SN), 부사(MAG)
    filtered_tokens = []
    for token in tokens:
        # 품사 필터링
        if any(token.tag.startswith(tag) for tag in ['NN', 'VV', 'VA', 'SL', 'SN', 'MAG']):
            # 단일 문자 제거 (의미 없는 한글자 단어), 단 영어/숫자는 예외
            if len(token.form) >= 2 or token.tag in ['SL', 'SN']:
                filtered_tokens.append(token.form)

    return filtered_tokens


def bm25_search(query: str, top_k: int = 100) -> List[str]:
    """
    BM25 기반 검색 (전처리 강화, 캐시 사용)

    Args:
        query: 검색 쿼리 (자연어 문장)
        top_k: 반환할 최대 결과 개수

    Returns:
        List[str]: 관련도 높은 순으로 정렬된 ID 리스트
    """
    # 캐시된 인덱스 파일에서 로드
    cache_file = 'bm25_index.pkl'

    try:
        with open(cache_file, 'rb') as f:
            bm25, doc_ids = pickle.load(f)
            print(f"✓ BM25 인덱스 캐시 로드 완료: {len(doc_ids)}개 문서")
    except FileNotFoundError:
        print(f"✗ BM25 인덱스 캐시 파일({cache_file})을 찾을 수 없습니다.")
        print(f"  먼저 'python build_index.py'를 실행하여 인덱스를 생성하세요.")
        return []
    except Exception as e:
        print(f"✗ BM25 인덱스 로드 실패: {e}")
        return []

    # 쿼리 전처리 (동일한 전처리 파이프라인 적용)
    query_tokens = preprocess_text(query)

    print(f"원본 쿼리: {query}")
    print(f"전처리된 토큰: {query_tokens}")

    if not query_tokens:
        print("✗ 쿼리 전처리 결과가 비어있습니다.")
        return []

    # BM25 점수 계산
    scores = bm25.get_scores(query_tokens)

    # 상위 K개 반환
    top_indices = sorted(range(len(scores)),
                        key=lambda i: scores[i],
                        reverse=True)[:top_k]

    results = [doc_ids[i] for i in top_indices]

    # # 디버깅용: 상위 3개 점수 출력
    # print(f"\n상위 3개 결과:")
    # for i in range(min(3, len(top_indices))):
    #     idx = top_indices[i]
    #     print(f"  {i+1}. ID: {doc_ids[idx]}, Score: {scores[idx]:.4f}")

    return results



def vector_search(query_vec: list, top_k: int = 100) -> List[str]:
    """
    wellcome1st_vector_data 테이블에서 벡터 유사도 기반 검색
    """
    conn = get_vector_db_conn()
    cur = conn.cursor()
    
    try:
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec)
        
        sql = """
            SELECT fk_id, embedding <=> %s AS distance
            FROM wellcome1st_vector_data
            ORDER BY distance ASC
            LIMIT %s
        """
        cur.execute(sql, (query_vec, top_k)) 
        results = [r[0] for r in cur.fetchall()]
        return results

    finally:
        cur.close()
        conn.close()

def get_jsons_by_ids(ids: list):
    """
    ID 리스트를 받아서 해당하는 JSON 데이터 반환
    Args:
        ids: 검색할 ID 리스트
    Returns:
        [{"id": "...", "info_text": {...}}, ...]
    """
    # 빈 리스트인 경우 빈 리스트 반환
    if not ids:
        return []

    conn = get_json_db_conn()
    cur = conn.cursor()

    try:
        # WHERE IN 절 사용 (tuple로 변환)
        sql = """
            SELECT id, info_text
            FROM wellcome1st_json_data
            WHERE id IN %s
        """
        # %s에 tuple을 전달 (리스트를 tuple로 감싸기)
        cur.execute(sql, (tuple(ids),))

        # 결과를 딕셔너리 리스트로 변환
        results = [{"id": r[0], "info_text": r[1]} for r in cur.fetchall()]
        return results

    finally:
        cur.close()
        conn.close()

