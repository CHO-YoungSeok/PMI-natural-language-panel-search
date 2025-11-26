import anthropic, os, json
from anthropic import AnthropicBedrock

# 전역 클라이언트 선언 (모듈 로드 시 한 번만 생성)
SONNET_CLIENT = AnthropicBedrock(aws_region="ap-southeast-2")
NORTH_SONNET_CLIENT = AnthropicBedrock(aws_region="ap-northeast-1")

def preprocess_query(query: str) -> tuple[str, int]:
    """
    쿼리 전처리 및 개수 추출 (병렬 처리로 속도 최적화)
 
    두 개의 LLM 호출을 병렬로 실행:
    1. 개수 추출: 쿼리에서 요구 개수 파싱 (기본값 30)
    2. 쿼리 정제: 개수 정보 제거, 조건 명확화, 표준화

    Returns:
        tuple[str, int]: (정제된_쿼리, 요구_개수)
    """
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures  

    def extract_count(q: str) -> int:
        """개수 추출 (별도 스레드에서 실행)"""
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
        count_message = NORTH_SONNET_CLIENT.messages.create(
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

    def clean_query_text(q: str) -> str:
        """쿼리 정제 (별도 스레드에서 실행)"""
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

    # 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_count = executor.submit(extract_count, query)
        future_clean = executor.submit(clean_query_text, query)

        # 모든 작업 완료 대기
        concurrent.futures.wait([future_count, future_clean])

        # 결과 가져오기
        result_count = future_count.result()
        clean_query = future_clean.result()

    return (clean_query, result_count)

import re
import json

# 실제 사용 시 적절한 클라이언트 객체로 교체하세요
# from anthropic import Anthropic
# SONNET_CLIENT = Anthropic(api_key="...")

def llm_filter_panel(query: str, jsons: list) -> str:
    """
    Sonnet을 사용하여 사용자 쿼리에 부합하는 패널을 필터링하고 랭킹을 매깁니다.
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
You are an expert panel ranking AI. Your task is to select and rank survey panels based on their relevance to a search query.

# Input Format
Each panel has:
- `id`: A unique identifier string (e.g., "w10001", "w10023")
- `info`: A dictionary containing demographic and behavioral data

# Your Task

## CRITICAL: Absolute Location Filtering
**This is the HIGHEST priority rule - override all other scoring:**

1. Extract location criteria from query (e.g., "서울", "경기", "부산", "인천", etc.)
2. For EACH panel:
   - Check if panel's location field in `info` matches the query's location requirement
   - Location matching rules:
     * Query mentions "서울" → ONLY accept panels with "서울" in location field
     * Query mentions "경기" → ONLY accept panels with "경기" in location field
     * Query mentions "인천" → ONLY accept panels with "인천" in location field
     * Partial matches NOT allowed (e.g., "경기" ≠ "서울", "인천" ≠ "서울")
   - If location DOES NOT match: IMMEDIATELY assign score = -9999 (disqualified)
   - If location matches OR no location specified in query: proceed with normal scoring

3. Examples:
   - Query: "서울에 거주하는 20대"
     * Panel location "서울" → Proceed with normal scoring ✓
     * Panel location "경기" → score = -9999 (DISQUALIFIED) ✗
     * Panel location "인천" → score = -9999 (DISQUALIFIED) ✗
   - Query: "경기 30대 남성"
     * Panel location "경기" → Proceed with normal scoring ✓
     * Panel location "서울" → score = -9999 (DISQUALIFIED) ✗

**This location filtering rule applies BEFORE any other scoring criteria.**

## Standard Scoring (applied AFTER location filtering)

1. Analyze the search query to extract:
   - Target criteria (location, age, gender, behaviors, etc.)
   - Required count N (e.g., "20명" = 20, "100명" = 100, "30명" = 30)

2. Score each panel (only if not disqualified by location):
   - Start with 50 points (neutral)
   - +10 points: Panel perfectly matches a critical criterion (age, gender)
   - +5 points: Panel matches a secondary criterion (behavior, preference)
   - -100 points: Panel directly contradicts a criterion (e.g., query wants "male" but panel is "female")
   - -10 points: Panel contradicts a secondary criterion
   - Ignore the `id` field for matching - only use `info` values

3. Rank ALL panels by score (highest first, disqualified panels at bottom)

4. **ABSOLUTE COUNT REQUIREMENT**:
   - Extract requested count N from query (e.g., "100명" → N=100)
   - After scoring ALL panels (including disqualified ones):
     * Sort by score (highest first)
     * Select top N panels
   - You MUST return EXACTLY N panel IDs
   - ONLY exception: if total candidates < N, return ALL candidates
   - Even if top N includes low/negative scores, still return them
   - If the query asks for "30명", return EXACTLY 30 panel IDs
   - If the query asks for "100명", return EXACTLY 100 panel IDs

# CRITICAL OUTPUT REQUIREMENTS
- You MUST output ONLY space-separated IDs inside <result> tags
- Format: <result>w10001 w10023 w10087</result>
- ABSOLUTELY NO other text, explanations, numbers, labels, punctuation, or line breaks inside <result> tags
- Only IDs separated by single spaces
- Return EXACTLY N IDs (the number requested in the query)

Example for "30명" query:
<result>w10001 w10023 w10087 w10045 w10099 w10123 w10145 w10167 w10189 w10201 w10223 w10245 w10267 w10289 w10301 w10323 w10345 w10367 w10389 w10401 w10423 w10445 w10467 w10489 w10501 w10523 w10545 w10567 w10589 w10601</result>
"""

    user_prompt = f"""Search Query: {query}

Candidate Panels (Total: {len(jsons)} panels):
{json.dumps(jsons, ensure_ascii=False, indent=2)}

Select the most relevant panels and output ONLY their IDs in this exact format.
Remember: Return EXACTLY the number of panels requested in the query.
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
        
        # 정규표현식으로 <result> 태그 안의 내용만 추출
        match = re.search(r'<result>(.*?)</result>', raw_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # 태그가 없을 경우(예외적 상황) 전체 텍스트에서 공백 정리 후 반환 시도
            return raw_content.strip()

    except Exception as e:
        print(f"LLM Error: {e}")
        return ""
