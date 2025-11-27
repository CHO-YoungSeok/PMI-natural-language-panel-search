import psycopg2
import numpy as np
import os
import re
import pickle
from typing import List, Dict, Optional
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

# 도메인 특화 사용자 사전 추가 (복합명사를 단일 토큰으로 인식)
# 음용경험/OTT 검색 정확도 향상을 위한 핵심 키워드
core_keywords = [
    'OTT', 'OTT구독', 'OTT서비스',
    '음용경험', '음주경험'
]

for keyword in core_keywords:
    kiwi.add_user_word(keyword, 'NNG', 1.0)  # 일반명사로 등록, 높은 우선순위

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


# 한국 17개 광역시도 리스트 (거주지역 필터링용)
REGIONS = [
    "서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]


def extract_regions_from_query(query: str) -> Optional[List[str]]:
    """
    쿼리에서 거주지역 키워드 추출 (키워드 매칭 방식)

    Args:
        query: 전처리된 검색 쿼리 (clean_query)

    Returns:
        Optional[List[str]]: 발견된 지역 리스트 또는 None (지역 미언급 시)

    Example:
        >>> extract_regions_from_query("서울 및 경기 20대")
        ['서울', '경기']
        >>> extract_regions_from_query("OTT 이용자 100명")
        None
    """
    if not query:
        return None

    found_regions = []
    for region in REGIONS:
        if region in query:
            found_regions.append(region)

    return found_regions if found_regions else None


def extract_gender_from_query(query: str) -> Optional[List[str]]:
    """
    쿼리에서 성별 키워드 추출 (키워드 매칭 방식, DB 값으로 정규화)

    Args:
        query: 전처리된 검색 쿼리 (clean_query)

    Returns:
        Optional[List[str]]: 발견된 성별 리스트 (DB 값: "남자", "여자") 또는 None

    Example:
        >>> extract_gender_from_query("서울 20대 남성")
        ['남자']
        >>> extract_gender_from_query("경기 여성 및 남성")
        ['여자', '남자']
        >>> extract_gender_from_query("서울 20대")
        None
    """
    if not query:
        return None

    found_genders = []

    # "남성", "남자" → "남자" (DB 값으로 정규화)
    if any(keyword in query for keyword in ["남성", "남자"]):
        found_genders.append("남자")

    # "여성", "여자" → "여자" (DB 값으로 정규화)
    if any(keyword in query for keyword in ["여성", "여자"]):
        found_genders.append("여자")

    return found_genders if found_genders else None


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


def filter_by_birth_years(ids: list, birth_years_str: Optional[str]) -> List[str]:
    """
    패널 ID 리스트를 출생년도로 필터링

    Args:
        ids: 필터링할 패널 ID 리스트
        birth_years_str: 공백으로 구분된 출생년도 문자열 또는 None (필터링 안함)

    Returns:
        필터링된 ID 리스트 (birth_years_str이 None이면 원본 리스트 반환)
    """
    if birth_years_str is None or not ids:
        return ids

    # 출생년도 문자열을 리스트로 파싱
    birth_years = birth_years_str.split()

    conn = get_json_db_conn()
    cur = conn.cursor()

    try:
        # info_text JSON에서 '출생년도' 필드를 추출하여 필터링
        # PostgreSQL의 jsonb 연산자 사용
        sql = """
            SELECT id
            FROM wellcome1st_json_data
            WHERE id = ANY(%s)
            AND (info_text::jsonb->>'출생년도') = ANY(%s)
        """
        cur.execute(sql, (ids, birth_years))
        filtered_ids = [r[0] for r in cur.fetchall()]

        print(f"출생년도 필터링: {len(ids)}개 → {len(filtered_ids)}개 (출생년도: {len(birth_years)}개)")

        return filtered_ids

    except Exception as e:
        print(f"✗ 출생년도 필터링 실패: {e}")
        # 필터링 실패 시 원본 리스트 반환
        return ids

    finally:
        cur.close()
        conn.close()


def filter_by_regions(ids: List[str], regions: Optional[List[str]]) -> List[str]:
    """
    패널 ID 리스트를 거주지역으로 필터링 (OR 조건)

    Args:
        ids: 필터링할 패널 ID 리스트
        regions: 거주지역 리스트 (예: ["서울", "경기"]) 또는 None (필터링 안함)

    Returns:
        필터링된 ID 리스트 (regions가 None이면 원본 리스트 반환)

    Example:
        >>> filter_by_regions(['w001', 'w002', 'w003'], ['서울', '경기'])
        ['w001', 'w003']  # 서울 또는 경기 거주자만
    """
    if regions is None or not ids:
        return ids

    conn = get_json_db_conn()
    cur = conn.cursor()

    try:
        # info_text JSON에서 '거주지역' 필드를 추출하여 필터링 (OR 조건: ANY)
        sql = """
            SELECT id
            FROM wellcome1st_json_data
            WHERE id = ANY(%s)
            AND (info_text::jsonb->>'거주지역') = ANY(%s)
        """
        cur.execute(sql, (ids, regions))
        filtered_ids = [r[0] for r in cur.fetchall()]

        print(f"거주지역 필터링: {len(ids)}개 → {len(filtered_ids)}개 (지역: {regions})")

        return filtered_ids

    except Exception as e:
        print(f"✗ 거주지역 필터링 실패: {e}")
        # 필터링 실패 시 원본 리스트 반환
        return ids

    finally:
        cur.close()
        conn.close()


def filter_by_gender(ids: List[str], genders: Optional[List[str]]) -> List[str]:
    """
    패널 ID 리스트를 성별로 필터링 (OR 조건)

    Args:
        ids: 필터링할 패널 ID 리스트
        genders: 성별 리스트 (예: ["남자"], ["여자"], ["남자", "여자"]) 또는 None (필터링 안함)

    Returns:
        필터링된 ID 리스트 (genders가 None이면 원본 리스트 반환)

    Example:
        >>> filter_by_gender(['w001', 'w002', 'w003'], ['남자'])
        ['w001', 'w003']  # 남자만
    """
    if genders is None or not ids:
        return ids

    conn = get_json_db_conn()
    cur = conn.cursor()

    try:
        # info_text JSON에서 '성별' 필드를 추출하여 필터링 (OR 조건: ANY)
        sql = """
            SELECT id
            FROM wellcome1st_json_data
            WHERE id = ANY(%s)
            AND (info_text::jsonb->>'성별') = ANY(%s)
        """
        cur.execute(sql, (ids, genders))
        filtered_ids = [r[0] for r in cur.fetchall()]

        print(f"성별 필터링: {len(ids)}개 → {len(filtered_ids)}개 (성별: {genders})")

        return filtered_ids

    except Exception as e:
        print(f"✗ 성별 필터링 실패: {e}")
        # 필터링 실패 시 원본 리스트 반환
        return ids

    finally:
        cur.close()
        conn.close()


def bm25_search(
    query: str,
    top_k: int = 100,
    birth_years: Optional[str] = None,
    regions: Optional[List[str]] = None,
    genders: Optional[List[str]] = None
) -> List[str]:
    """
    BM25 기반 검색 (전처리 강화, 캐시 사용, 다중 필터링)

    Args:
        query: 검색 쿼리 (자연어 문장)
        top_k: 반환할 최대 결과 개수
        birth_years: 공백으로 구분된 출생년도 문자열 (옵션)
        regions: 거주지역 리스트 (예: ["서울", "경기"]) (옵션)
        genders: 성별 리스트 (예: ["남자"], ["여자"]) (옵션)

    Returns:
        List[str]: 관련도 높은 순으로 정렬된 ID 리스트 (모든 필터링 적용)
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

    # 출생년도 필터링 적용
    if birth_years is not None:
        results = filter_by_birth_years(results, birth_years)

    # 거주지역 필터링 적용
    if regions is not None:
        results = filter_by_regions(results, regions)

    # 성별 필터링 적용
    if genders is not None:
        results = filter_by_gender(results, genders)

    return results



def vector_search(
    query_vec: list,
    top_k: int = 100,
    birth_years: Optional[str] = None,
    regions: Optional[List[str]] = None,
    genders: Optional[List[str]] = None
) -> List[str]:
    """
    wellcome1st_vector_data 테이블에서 벡터 유사도 기반 검색 (다중 필터링 지원)

    Args:
        query_vec: 쿼리 임베딩 벡터
        top_k: 반환할 최대 결과 개수
        birth_years: 공백으로 구분된 출생년도 문자열 (옵션)
        regions: 거주지역 리스트 (예: ["서울", "경기"]) (옵션)
        genders: 성별 리스트 (예: ["남자"], ["여자"]) (옵션)

    Returns:
        List[str]: 유사도 높은 순으로 정렬된 ID 리스트 (모든 필터링 적용)
    """
    conn = get_vector_db_conn()
    cur = conn.cursor()

    try:
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec)

        # 필터가 하나라도 있으면 JOIN 사용
        has_filters = birth_years is not None or regions is not None or genders is not None

        if has_filters:
            # 동적으로 WHERE 조건 구성
            where_conditions = []
            params = [query_vec]

            if birth_years is not None:
                birth_year_list = birth_years.split()
                where_conditions.append("(j.info_text::jsonb->>'출생년도') = ANY(%s)")
                params.append(birth_year_list)

            if regions is not None:
                where_conditions.append("(j.info_text::jsonb->>'거주지역') = ANY(%s)")
                params.append(regions)

            if genders is not None:
                where_conditions.append("(j.info_text::jsonb->>'성별') = ANY(%s)")
                params.append(genders)

            where_clause = " AND ".join(where_conditions)
            params.append(top_k)

            sql = f"""
                SELECT v.fk_id, v.embedding <=> %s AS distance
                FROM wellcome1st_vector_data v
                JOIN wellcome1st_json_data j ON v.fk_id = j.id
                WHERE {where_clause}
                ORDER BY distance ASC
                LIMIT %s
            """
            cur.execute(sql, params)
        else:
            # 필터링 없음: 기존 로직
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


def has_field_info(ids: List[str], field_keywords: List[str]) -> List[str]:
    """
    정확한 JSON 필드 키 매칭으로 도메인 필터링

    Changes from previous version:
    - LIKE '%keyword%' → Exact match using jsonb ? operator
    - Supports multiple exact keys per domain (OR condition)
    - More reliable: only matches exact question keys from DB

    Args:
        ids: Panel ID list to filter
        field_keywords: Exact JSONB key strings (e.g., "여러분이 현재 이용 중인 OTT 서비스는 몇 개인가요?")

    Returns:
        Filtered panel IDs that have at least one of the specified keys with non-null, non-empty values

    Example:
        >>> ids = ['w001', 'w002', 'w003']
        >>> has_field_info(ids, ['여러분이 현재 이용 중인 OTT 서비스는 몇 개인가요?'])
        ['w001', 'w003']  # Only panels with exact OTT field
    """
    if not ids or not field_keywords:
        return ids

    conn = get_json_db_conn()
    cur = conn.cursor()

    try:
        # Build OR condition for multiple exact keys
        # Each keyword is an exact key string, not a substring to match
        conditions = []
        for exact_key in field_keywords:
            conditions.append(f"""
                (
                    (info_text::jsonb ? %s)
                    AND (info_text::jsonb->>%s) IS NOT NULL
                    AND (info_text::jsonb->>%s) != ''
                )
            """)

        where_clause = " OR ".join(conditions)

        # Flatten the parameter list: each exact_key appears 3 times in the query
        params = [ids]
        for exact_key in field_keywords:
            params.extend([exact_key, exact_key, exact_key])

        sql = f"""
            SELECT id
            FROM wellcome1st_json_data
            WHERE id = ANY(%s)
            AND ({where_clause})
        """
        cur.execute(sql, params)
        filtered_ids = [r[0] for r in cur.fetchall()]

        # Enhanced logging
        print(f"[도메인 필터] 정확한 키 매칭 모드")
        print(f"[도메인 필터] 검색 키: {field_keywords}")
        print(f"[도메인 필터] 입력: {len(ids)}개 → 출력: {len(filtered_ids)}개")
        if len(ids) > 0:
            print(f"[도메인 필터] 필터링 비율: {len(filtered_ids)/len(ids)*100:.1f}%")

        return filtered_ids

    except Exception as e:
        print(f"✗ 도메인 필터링 실패: {e}")
        # Return original list on failure
        return ids
    finally:
        cur.close()
        conn.close()

