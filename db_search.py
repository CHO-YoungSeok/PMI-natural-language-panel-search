import psycopg2
import numpy as np
import os
from typing import List, Dict
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

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

# 한 문장에 대해서는 BM25와 성능 차이가 거의 없음. 
## wellcome1st_json_data 테이블에서 PostgreSQL 내장 Full-Text Search 기반 텍스트 검색
## - 가장 유사한 순으로 id를 리스트로 반환
def fts_search(query: str, top_k: int = 100) -> List[str]:
    conn = get_json_db_conn()
    cur = conn.cursor()

    try :
        sql = """
            SELECT id
            FROM wellcome1st_json_data
            WHERE to_tsvector('simple', info_text) @@ plainto_tsquery('simple', %s)
            ORDER BY ts_rank_cd(to_tsvector('simple', info_text), plainto_tsquery('simple', %s)) DESC
            LIMIT %s
        """
        cur.execute(sql, (query, query, top_k))
        results = [r[0] for r in cur.fetchall()]
        return results

    finally : 
        cur.close()
        conn.close()

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

