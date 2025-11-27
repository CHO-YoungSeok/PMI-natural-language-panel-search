import anthropic, os, json
from anthropic import AnthropicBedrock, RateLimitError
from typing import Optional, Tuple
import re
import json
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# 로거 설정
logger = logging.getLogger(__name__)

# 전역 클라이언트 선언 (모듈 로드 시 한 번만 생성)
SONNET_CLIENT = AnthropicBedrock(aws_region="ap-southeast-2")
NORTH_SONNET_CLIENT = AnthropicBedrock(aws_region="ap-northeast-1")

def preprocess_query(query: str) -> Tuple[str, int, Optional[str]]:
    """
    쿼리 전처리 및 개수 추출 (병렬 처리로 속도 최적화)

    세 개의 LLM 호출을 병렬로 실행:
    1. 개수 추출: 쿼리에서 요구 개수 파싱 (기본값 30)
    2. 쿼리 정제: 개수 정보 제거, 조건 명확화, 표준화
    3. 출생년도 추출: 나이 표현을 출생년도로 변환

    Returns:
        Tuple[str, int, Optional[str]]: (정제된_쿼리, 요구_개수, 출생년도_문자열 또는 None)
    """
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def extract_count(q: str) -> int:
        """개수 추출 with retry logic (별도 스레드에서 실행)"""
        count_system_prompt = """
당신은 한국어 자연어 검색 쿼리에서 "최종 몇 명의 결과를 반환해야 하는지"를 추출하는 전문가입니다.

## 규칙:
1. 반드시 하나의 자연수만 출력
2. 자연수 이외의 어떤 문자도 출력하지 않음 (설명, 단위, 공백, 개행 금지)
3. 쿼리에 명시된 인원 수 규칙:
   - "10명", "30명", "100명" → 그 숫자 사용
   - "열 명", "스무 명", "삼십 명" 등 한글 숫자도 인식
   - 범위: "10~20명" → 상한값(20) 사용
   - "최소 3명 이상" → 명시된 숫자(3) 사용
   - **중요**: 개수 표현이 전혀 없으면 기본값 30 반환

4. 검색 결과 개수와 무관한 숫자는 무시:
   - 날짜, 연도, 연령대 등은 개수가 아님
   - 예: "2024년 서울 20대" → 개수 없음 → 30 반환

## 출력 형식:
- 오직 하나의 양의 정수만 출력
- 예시:
  * "서울 20대 남자 100명" → 100
  * "경기 OTT 이용자" → 30 (개수 없음)
  * "부산 30대 여자 열 명" → 10
"""
        count_message = SONNET_CLIENT.messages.create(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=8,
            temperature=0.0,
            system=count_system_prompt,
            messages=[{"role": "user", "content": f"쿼리: {q}"}]
        )
        try:
            return int(count_message.content[0].text.strip())
        except (ValueError, TypeError):
            return 30  # 변환 실패 시 기본값

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def clean_query_text(q: str) -> str:
        """쿼리 정제 with retry logic (별도 스레드에서 실행)"""
        clean_system_prompt = """
당신은 설문·패널 기반 자연어 검색 시스템에서 입력 쿼리의 품질을 극대화하는 검색질문 전처리 전문가입니다.

## 반드시 다음을 지키세요:
- 오타, 띄어쓰기, 맞춤법 오류를 완벽히 수정
- 의도나 조건(지역, 성별, 출생년도, 경험, 이용여부 등)을 모두 자연어로 조합
- **중요**: 쿼리에서 "30명", "100명", "열 명" 등 개수 표현은 완전히 제거
- 부정·제외·결여 조건은 자연어로 풀어쓰기 ("비흡연" → "흡연을 하지 않는")
- 영어, 한자, 접두사 형태('비', '불', '무', '미', 'non', '非')는 부정적 자연어로 풀어쓰기
- "술"은 "음용경험"으로 표현
- 젊은층은 출생년도로 해석 (20대~30대, 노년층은 60대 이상)
- 남성은 남자로, 여성은 여자로 표현
- 전처리된 검색용 문장만 출력 (설명, 해석 X)

## 예시:
입력: "서울 및 경기 지역에 거주하며 OTT 서비스를 이용하는 20대~30대 성인 30명"
출력: "서울 및 경기 지역에 거주하며 OTT 서비스를 이용하는 20대~30대 성인"
"""
        clean_message = NORTH_SONNET_CLIENT.messages.create(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=512,
            temperature=0.0,
            system=clean_system_prompt,
            messages=[{"role": "user", "content": f"원본 쿼리: {q}를 조건에 맞게 전처리해"}]
        )
        return clean_message.content[0].text.strip()




    # 병렬 실행 (3개 작업)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_count = executor.submit(extract_count, query)
        future_clean = executor.submit(clean_query_text, query)
        future_birth_years = executor.submit(extract_birth_years, query)

        # 모든 작업 완료 대기
        concurrent.futures.wait([future_count, future_clean, future_birth_years])

        # 결과 가져오기
        result_count = future_count.result()
        clean_query = future_clean.result()
        birth_years = future_birth_years.result()

    return (clean_query, result_count, birth_years)




@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def extract_birth_years(query: str) -> Optional[str]:
    """
    Extract age/age range expressions from Korean queries with retry logic

    Args:
        query: User search query in Korean

    Returns:
        Space-separated birth year string (e.g., "1996 1997 1998 1999 2000") or None if no age expression
    """
    birth_year_system_prompt = """
You are an expert at analyzing Korean natural language queries to extract age/age range expressions and convert them into corresponding birth years.

## Task:
Extract age-related terms from Korean queries and output ALL matching birth years as space-separated integers.

## Rules:
1. If age expressions exist: Output all corresponding birth years separated by single spaces
2. If no age expressions exist: Output exactly "NONE"
3. Base all calculations on current year: 2025
4. Output format: Only 4-digit years with spaces between them (no text, explanations, or units)

## Age Expression Mappings (Korean → Birth Years):

### Decade Terms:
- [translate:10대] (teens): 2006-2015 (ages 10-19)
- [translate:20대] (twenties): 1996-2005 (ages 20-29)
- [translate:30대] (thirties): 1986-1995 (ages 30-39)
- [translate:40대] (forties): 1976-1985 (ages 40-49)
- [translate:50대] (fifties): 1966-1975 (ages 50-59)
- [translate:60대] (sixties): 1956-1965 (ages 60-69)
- [translate:70대] (seventies): 1946-1955 (ages 70-79)

### Generational Terms:
- [translate:젊은이]/[translate:젊은층] (young people/youth): 1991-2006 (ages 18-34)
- [translate:청장년층] (young to middle-aged): 1976-2006 (ages 19-49)
- [translate:중장년층] (middle to older-aged): 1961-1985 (ages 40-64)
- [translate:노인]/[translate:어르신]/[translate:늙은이] (elderly/seniors): 1926-1960 (ages 65+)

### Life Stage Terms:
- [translate:학생] (students, K-12 & college): 2001-2018 (ages 7-24)
- [translate:어린이] (children): 2013-2019 (ages 6-12)
- [translate:아기] (babies/infants): 2021-2025 (ages 0-4)

## Examples:

Input: "[translate:서울 20대 남자]"
Output: 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005

Input: "[translate:경기 여성 100명]"
Output: NONE

Input: "[translate:젊은이 50명]"
Output: 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006

Input: "[translate:30~40대 남자]"
Output: 1976 1977 1978 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995

## Important Notes:
- For range expressions like "[translate:30~40대]", include ALL years from both decades
- Always output complete year sequences (don't skip years)
- Match Korean age terminology exactly as specified above
"""

    try:
        birth_year_message = NORTH_SONNET_CLIENT.messages.create(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=256,
            temperature=0.0,
            system=birth_year_system_prompt,
            messages=[{"role": "user", "content": f"Query: {query}"}]
        )

        result = birth_year_message.content[0].text.strip()

        # Convert "NONE" to None
        if result == "NONE":
            return None

        # Return space-separated birth year string
        return result

    except RateLimitError:
        # Re-raise to trigger retry decorator
        raise
    except Exception as e:
        # Log other errors and return None (skip filtering)
        print(f"Birth year extraction error: {e}")
        return None



@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def llm_filter_panel(query: str, jsons: list) -> str:
    """
    Sonnet panel filtering with retry logic
    정확도를 위해 CoT(Chain of Thought)를 유도하고, 결과만 파싱하여 반환합니다.

    Args:
        query: 사용자 검색 쿼리
        jsons: [{"id": "...", "info": {...}}, ...] 형식의 후보 패널 리스트

    Returns:
        공백으로 구분된 패널 ID 문자열 (예: "w1 w2 w3")
    """
    if not jsons:
        return ""
    
    system_prompt = """
You are an expert panel ranking AI. Your task is to select and rank survey panels based on their relevance to a search query with MAXIMUM ACCURACY.

# Input Format
Each panel has:
- `id`: A unique identifier string (e.g., "w10001", "w10023")
- `info`: A dictionary containing demographic and behavioral data

# Your Task

## STEP 1: CRITICAL MANDATORY FILTERING (DISQUALIFICATION RULES)
**Apply these rules FIRST - panels that fail ANY rule get score = -9999 (DISQUALIFIED)**

### Rule 1: Negative Condition Filtering (Priority)
**Identify usage/behavioral constraints:**
- Query "이용하지 않는/안 함/없음" → Panel must confirm NON-usage.
- Query "이용하는/함" → Panel must confirm usage.
- Ambiguous or contradictory answers → DISQUALIFIED.

### Rule 2: Numeric & Income Range Filtering (CRITICAL ACCURACY)
**You must treat income/age/money as MATHEMATICAL VALUES, not text.**

#### 2.1 Logic for "Greater Than / Over" (이상, 초과)
- **Query:** "월 소득 X원 이상" / "X원 초과" (e.g., "1,000만원 이상")
- **Logic:** 
  1. Extract target number X from query (Normalize "1,000" → 1000).
  2. Parse panel's range [Min, Max].
  3. **DISQUALIFY if Panel Max < X**.
  4. **DISQUALIFY if Panel Range is "Below X" (e.g., "X 미만")**.

**Specific Examples for Query: "월 소득 1,000만원 이상" (Target ≥ 1000)**

❌ **DISQUALIFIED (Score = -9999):**
- "200~300만원" (Max 300 < 1000) → FAIL
- "400~500만원" (Max 500 < 1000) → FAIL
- "500~1,000만원 미만" (Range is < 1000. The word "미만" means strictly less than) → FAIL
- "700~1,000만원 미만" → FAIL
- "소득 없음" → FAIL

✅ **QUALIFIED:**
- "1,000만원 이상" (Min 1000 ≥ 1000) → PASS
- "1,500만원 이상" → PASS
- "2,000만원 이상" → PASS

#### 2.2 Logic for "Less Than / Under" (이하, 미만)
- **Query:** "월 소득 X원 미만" / "이하"
- **Logic:** DISQUALIFY if Panel Min ≥ X.

**Specific Examples for Query: "월 소득 300만원 미만" (Target < 300)**

❌ **DISQUALIFIED (Score = -9999):**
- "300~400만원" (Start at 300) → FAIL
- "400~500만원" → FAIL
- "300만원 이상" → FAIL

✅ **QUALIFIED:**
- "200만원 미만" → PASS
- "200~300만원 미만" → PASS

#### 2.3 Handling Units & Formats
- Treat "1,000" and "1000" as identical.
- Ignore "만원", "원" text when comparing numbers.
- Be careful with "미만" (Under) vs "이하" (Equal or Under).
- **Strictly check boundaries:** "1000만원 미만" DOES NOT satisfy "1000만원 이상".

### Rule 3: Explicit Exclusions
- If query says "A 제외", panels matching A → score = -9999

**After Step 1: Only panels with score ≠ -9999 proceed to Step 2**

## STEP 2: Standard Scoring (ONLY for non-disqualified panels)

1. **Score each qualified panel:**
   - Start: 50 points (neutral baseline)
   - **Perfect match on behavioral criterion:** +15 points
   - **Match on secondary preference:** +5 points
   - **DO NOT apply disqualification scores here** (already handled in Step 1)

2. **Rank ALL panels by score:**
   - Sort: highest score first, disqualified panels (-9999) at bottom

3. **ABSOLUTE COUNT REQUIREMENT:**
   - Extract N from query (e.g., "100명" → N=100)
   - Select top N panels from sorted list
   - Return EXACTLY N panel IDs
   - ONLY exception: if total candidates < N, return ALL candidates

## CRITICAL OUTPUT REQUIREMENTS
- Output ONLY space-separated IDs inside <result> tags
- Format: <result>w10001 w10023 w10087</result>
- NO other text inside <result>
- Return EXACTLY N IDs

## Processing Checklist (Internal):
1. ✓ Is this a numeric range query? (Income, Age)
2. ✓ Convert Query Target to Number (e.g., 1,000 → 1000)
3. ✓ Parse Panel Range Max/Min (e.g., "200~300" → Max 300)
4. ✓ Compare mathematically: Is 300 >= 1000? False. → DISQUALIFY.
5. ✓ Check "미만" (strictly less) vs "이상" (greater or equal) carefully.

Then output ONLY the <result> tags with IDs.
"""

    user_prompt = f"""Search Query: {query}

Candidate Panels (Total: {len(jsons)} panels):
{json.dumps(jsons, ensure_ascii=False, indent=2)}

**CRITICAL INSTRUCTIONS:**
1. Apply Rule 2 (Numeric Filtering) STRICTLY for income ranges.
2. "1,000만원 이상" requires the panel to be explicitly in the high-income bracket.
3. DISQUALIFY any range that is completely below the target (e.g., "400~500" is NOT "1000 이상").
4. DISQUALIFY "X 미만" ranges if the query asks for "X 이상".

Output format (IDs only, no extra text):
<result>w10001 w10023 w10087</result>
"""

    try:
        message = SONNET_CLIENT.messages.create(
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        raw_content = message.content[0].text

        match = re.search(r'<result>(.*?)</result>', raw_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return raw_content.strip()

    except RateLimitError:
        raise
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""
