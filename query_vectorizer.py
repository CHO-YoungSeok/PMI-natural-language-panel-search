# search_logic/query_vectorizer.py

# KURE-v1 모델 로드를 위해 sentence_transformers 라이브러리 사용
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# --- 1. 모델 로딩 (전역 변수로 한 번만 로드하여 재사용) ---
print("KURE-v1 모델 로딩 중... (최초 로딩 시 시간 소요)")
start_model_load_time = time.time()
try:
    KURE_MODEL = SentenceTransformer("nlpai-lab/KURE-v1")
    print(f"KURE-v1 모델 로드 성공! ({time.time() - start_model_load_time:.2f}초 소요)")
except Exception as e:
    print(f"🚨 KURE-v1 모델 로드 오류: {e}")
    KURE_MODEL = None


# --- 2. 쿼리 벡터화 함수 ---
def get_query_vector(query_text: str) -> list:
    """
    자연어 질문을 KURE-v1 모델로 1024차원 벡터로 변환하여 리스트 형태로 반환합니다.
    """
    if KURE_MODEL is None:
        raise ValueError("KURE-v1 모델이 로드되지 않았습니다. 환경을 확인하세요.")

    start_time = time.time()
    embeddings = KURE_MODEL.encode([query_text])
    query_vector = embeddings[0].tolist()

    print(f"[Vectorization] '{query_text[:20]}...' 벡터화 완료. ({time.time()-start_time:.4f}초)")
    return query_vector
