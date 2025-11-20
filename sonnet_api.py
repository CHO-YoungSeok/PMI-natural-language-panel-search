import anthropic, os

def get_sonnet_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY 환경변수가 없습니다.")
    return anthropic.Anthropic(api_key=api_key)

def preprocess_query(query: str) -> str:
    """
    Claude Sonnet을 사용해 '깔끔문장' 생성
    """
    client = get_sonnet_client()
    system_prompt = "입력 쿼리를 오타 없이 깔끔한 문장으로 변환하세요."
    user_prompt = f"질문: {query}"
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return message.content[0].text.strip()

def filter_panels(clean_query: str, panels: list) -> list:
    """
    Claude Sonnet을 사용해 패널 리스트 필터링
    """
    client = get_sonnet_client()
    context_text = "\n".join([str(p['id']) for p in panels])
    system_prompt = "질문 조건에 맞는 패널 id만 줄바꿈으로 구분해 출력하세요."
    user_prompt = f"질문: {clean_query}\n패널리스트 목록:\n{context_text}"
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=700,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return message.content[0].text.strip().splitlines()
