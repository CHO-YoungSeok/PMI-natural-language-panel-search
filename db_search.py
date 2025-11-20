import psycopg2
import numpy as np
from typing import List, Dict

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'your_user',
    'password': 'your_password',
    'dbname': 'your_db'
}

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def bm25_search(query: str, top_k: int = 100) -> List[Dict]:
    """
    wellcome1st_json_data 테이블에서 BM25 기반 텍스트 검색
    """
    conn = get_db_conn()
    cur = conn.cursor()
    sql = f"""
        SELECT id, info,
        ts_rank_cd(to_tsvector('korean', info::text), plainto_tsquery('korean', %s)) AS rank
        FROM wellcome1st_json_data
        WHERE to_tsvector('korean', info::text) @@ plainto_tsquery('korean', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    cur.execute(sql, (query, query, top_k))
    results = [ {'id': r[0], 'info': r[1], 'bm25_score': r[2]} for r in cur.fetchall() ]
    cur.close()
    conn.close()
    return results

def vector_search(query_vec: np.ndarray, top_k: int = 100) -> List[Dict]:
    """
    wellcome1st_vector_data 테이블에서 벡터 유사도 기반 검색
    """
    conn = get_db_conn()
    cur = conn.cursor()
    sql = f"""
        SELECT fk_id, embedding <=> %s AS distance
        FROM wellcome1st_vector_data
        ORDER BY distance ASC
        LIMIT %s
    """
    cur.execute(sql, (query_vec.tolist(), top_k))
    results = [ {'fk_id': r[0], 'distance': r[1]} for r in cur.fetchall() ]
    cur.close()
    conn.close()
    return results
