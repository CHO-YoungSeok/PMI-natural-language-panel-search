import anthropic, os, json
from anthropic import AnthropicBedrock

def get_sonnet_client():
    client = AnthropicBedrock(aws_region="ap-southeast-2")
    return client

def preprocess_query(query: str) -> str:
    """
    Claude Sonnet을 사용해 '깔끔문장' 생성
    """
    client = get_sonnet_client()
    
    system_prompt = """당신은 사용자의 입력 쿼리를 전문적으로 다듬는 AI 어시스턴트입니다.
다음 규칙을 엄격히 따르세요:
1. 한국어로 생성할 것
2. 오타와 문법 오류를 수정할 것
3. 패널을 특정할 수 있는 명확하고 구체적인 문장으로 변환할 것
4. 불필요한 말을 제거하고 핵심 의도만 간결하게 표현할 것
5. 원본 쿼리의 의미를 변경하지 말 것"""
    
    user_prompt = f"""다음 쿼리를 깔끔하고 명확한 문장으로 변환해주세요:
원본 쿼리: {query}
변환된 문장만 출력하세요. 추가 설명은 불필요합니다."""
    
    message = client.messages.create(
        model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=1024,
        system=system_prompt,  # system 파라미터로 전달
        messages=[
            {"role": "user", "content": user_prompt}  # user_prompt 사용
        ]
    )
    
    return message.content[0].text.strip()


def llm_filter_panel(query: str, jsons: list) -> list:
    """
    Sonnet을 사용해서 최종 패널 필터링 및 랭킹
    
    Args:
        query: 사용자 검색 쿼리
        jsons: [{"id": "...", "info": {...}}, ...] 형식의 후보 패널 리스트
    
    Returns:
        필터링되고 랭킹된 패널 리스트 (각 패널의 id를 ' '(공백)로 구분한다)
    """    
    if not jsons:
        return []

    client = get_sonnet_client()

    system_prompt = """당신은 사용자의 입력 쿼리를 전문적으로 다듬는 AI 어시스턴트입니다. 사용자의 검색 의도를 정확히 파악하고 가장 관련성 높은 패널을 우선적으로 선별하는 것이 당신의 임무입니다.
# 작업 지시사항
사용자 검색 쿼리와 후보 패널 리스트가 주어집니다. 다음 기준에 따라 패널을 필터링하고 랭킹하세요.

- [{"id" : ~, "info_text" : ~ } ... ] 형태이므로 id를 기준으로 해당 id의 info_text가 컴색 쿼리와 가장 유사한 것을 선별.
- 검색 쿼리와 가장 관련성이 높은 패널만 선택
- 쿼리에 명시된 숫자가 있으면 해당 숫자만큼 반환 (예: "3명 추천해줘" → 3명)
- 검색 쿼리에 적합도가 높은 순으로 ***반드시 id를 ' '로 구분***해서 반환할 것.
"""
    
    user_prompt = f"""## 검색 쿼리 : ###{query} 에 가장 부합###하는 패널들을 상위로 선별하세요.
    후보 패널 : {jsons}
    """
    
    message = client.messages.create(
        model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=1024,
        system=system_prompt,  # system 파라미터로 전달
        messages=[
            {"role": "user", "content": user_prompt}  # user_prompt 사용
        ]
    )
    
    return message.content[0].text.strip()

