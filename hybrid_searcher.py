import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# vectorizer 함수 (query 임베딩 생성) 외부에서 import 필요
from query_vectorizer import get_query_vector

def hybrid_search(query_text: str, top_k: int = 100):
    # DB 연결
    conn = psycopg2.connect(
        host="database-1.cpk06802so7a.ap-southeast-2.rds.amazonaws.com",
        port=5432,
        user="postgres",
        password=os.getenv("POSTGRES_DB_PASSWORD"),
        dbname="postgres"
    )
    register_vector(conn)

    cur = conn.cursor()

    # 쿼리 벡터 생성 (1024차원, float 리스트)
    query_vec = get_query_vector(query_text)
    # 정규화
    query_vec = query_vec / np.linalg.norm(query_vec)

    # pgvector의 '<=>': 코사인 거리 기반 오더(작을수록 유사)
    sql = """
    SELECT fk_id, embedding <=> %s AS cosine_distance
    FROM wellcome1st_vector_data
    ORDER BY cosine_distance ASC
    LIMIT %s;
    """
    
    cur.execute(sql, (query_vec.tolist(), top_k))
    results = cur.fetchall()  # [(fk_id, distance), ...]

    cur.close()
    conn.close()

    # 임베딩 결과 형식 가공
    vector_results = []
    for fk_id, distance in results:
        vector_results.append({
            "id": fk_id,
            "score": 1 - distance,  # 코사인 거리 → 유사도(1-거리)
            "distance": distance
        })

    return vector_results
