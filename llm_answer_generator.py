import anthropic, os, json, time

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def generate_llm_answer(clean_query: str, panelists: list) -> str:
    """
    Claude Sonnet 4.5를 사용해 최적의 패널 id 리스트를 반환합니다.
    """

    t0 = time.time()
    # 주요 필드만 포함해서 데이터 축약
    context_text = ""
    for i, panel in enumerate(panelists, 1):
        panel_id = panel["data"].get("id", f"panel_{i}")
        context_text += f"{panel_id}\n"

    system_prompt = (
        "당신은 패널 데이터 분석 전문가입니다.\n"
        "- 입력된 질문 조건에 부합하는 패널 id만 반환하세요.\n"
        "- 질문에 반환 수가 명시되어 있지 않으면 20명을 기본값으로 하세요.\n"
        "- id만 출력하고, 줄바꿈으로 구분하세요."
        "- 질문에 가장 잘 부합하는 사람을 상위에 우선적으로 출력하세요."
    )

    user_prompt = f"""질문: {clean_query}
패널리스트 목록:
{context_text}
질문 조건에 맞는 패널만 id로 출력해주세요.
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        answer = message.content[0].text.strip()
        print(f"[LLM Answer] 소요시간: {time.time()-t0:.2f}s, 답변: {len(answer)}글자")
        return answer
    except Exception as e:
        print(f"[LLM Answer Error] API Key: {os.environ.get('ANTHROPIC_API_KEY', 'None')} / {e}")
        return f"답변 생성 중 오류 발생: {str(e)}"
