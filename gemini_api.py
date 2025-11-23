import os
from dotenv import load_dotenv
from google import genai

# .env 파일 로드
load_dotenv()

def get_gemini_client():
    """
    Gemini API 클라이언트를 반환
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

    client = genai.Client(api_key=api_key)
    return client


def preprocess_query(query: str) -> str:
    """
    Gemini를 사용해 '깔끔문장' 생성
    """
    client = get_gemini_client()

    system_prompt = """당신은 검색 쿼리 전처리 전문가입니다.
사용자가 입력한 질문을 분석하여 다음 작업을 수행하세요:
1. 오타 및 맞춤법 오류 수정
2. 불필요한 조사나 표현 제거
3. 명확하고 간결한 검색용 문장으로 변환
4. 핵심 의도는 유지하되 검색에 최적화된 형태로 재구성
***반드시 전처리된 문장만 출력하세요. 추가 설명은 하지 마세요.** *
"""

    user_prompt = f"""다음 쿼리를 전처리하세요.: 원본 쿼리: {query} """

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=f"{system_prompt}\n\n{user_prompt}"
    )

    return response.text.strip()


def llm_filter_panel(query: str, jsons: list) -> list:
    """
    Gemini를 사용해서 최종 패널 필터링 및 랭킹

    Args:
        query: 사용자 검색 쿼리
        jsons: [{"id": "...", "info": {...}}, ...] 형식의 후보 패널 리스트

    Returns:
        필터링되고 랭킹된 패널 리스트 (각 패널의 id를 ' '(공백)로 구분한다)
    """
    if not jsons:
        return []

    client = get_gemini_client()

    system_prompt = """
당신은 사용자의 검색 쿼리에 가장 잘 맞는 패널을 선별하는 AI 어시스턴트입니다.
사용자의 검색 의도를 정확히 파악하고, 가장 관련성 높은 패널의 id를 골라내는 것이 당신의 임무입니다.

# 입력 데이터 설명
- 너에게 주어지는 데이터는 "패널 정보"이며, 각 패널은 다음과 같은 JSON 구조를 가진다.
{
  "id": "패널의 고유 ID (string)",
  "info": {
    "<질문/필드 이름>": "<응답 값>",
    ...
  }
}
- info의 key는 설문 문항/속성 이름이고, info의 value는 그 문항에 대한 응답이다.
- 의미 판단과 검색 쿼리 매칭에 사용할 수 있는 정보는 오직 info 내부의 key와 value이다.

# 작업 지시사항
- id 필드는 단순 식별용으로만 사용하며, 의미 판단에는 절대 사용하지 마라.
- value가 문자열, 리스트, 혹은 또 다른 객체일 수 있으며, 이 값들을 자연어로 읽고 해석해라.
- 검색 쿼리와 가장 관련성이 높은 패널 순으로 정렬해 반환하라.

# 조건 기반 점수 규칙
- 사용자의 검색 쿼리는 여러 개의 "조건"(예: 지역, 연령, 성별, 행동/경험 등)으로 이루어져 있다고 가정하라.
- 먼저 쿼리에서 이러한 조건들을 스스로 추출하라.
- 각 후보 패널에 대해 info 내부의 모든 key와 value를 읽고, 각 조건에 대해 다음 기준으로 판단하라.
- 조건을 만족하는 명확한 근거가 info 안에 있으면 +1점
- 조건과 반대되는 근거가 있으면 확실하게 결과에서 제외하라
- 각 패널의 최종 점수는 만족한 조건 수로 계산하라.
- 점수가 높은 패널일수록 검색 쿼리와 더 잘 맞는 것으로 간주하고 상위에 두어라.
- 특히, 쿼리에서 중요한 키워드(예: "술을 먹은", "반려동물을 키우는" 등)가 info 어딘가에 전혀 언급되지 않으면,
  그 조건을 만족했다고 보지 말고, 해당 조건에 대해서는 0점 또는 불리한 점수를 주어라.
- 쿼리에 숫자가 포함되어 있어도, 모든 조건을 충분히 만족하는 패널 수가 그 숫자보다 적다면
  억지로 개수를 맞추려고 관련성이 낮은 패널을 추가하지 마라.
  이 경우, 조건을 만족하는 패널의 id만 모두 반환하라.
- 해당되는 패널 수가 100명보다 적다면, 존재하는 패널의 id만 모두 반환하라.

# 반환 형식
- 검색 쿼리에 대한 적합도가 높은 순으로 id를 나열하라.
- id만 공백 한 칸으로 구분하여 한 줄로 반환하라. 예시: "w10001 w10023 w10087"
- 다른 설명, 이유, 문장, 마크다운, 따옴표는 절대 포함하지 마라.
- 쿼리에 명시된 숫자가 있으면, 조건을 만족하는 패널 중에서 최대 그 숫자까지 id를 반환하라. (예: "3명 추천해줘" → 최대 3개 id)
- 쿼리에 숫자가 없으면, 관련성이 높은 id를 최대 100개까지 반환하라.
"""


    user_prompt = f"""## 검색 쿼리 : ###{query} 에 가장 부합###하는 패널들을 상위로 선별하세요.
    후보 패널 : {jsons}
    """


    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=f"{system_prompt}\n\n{user_prompt}"
    )

    return response.text.strip()
