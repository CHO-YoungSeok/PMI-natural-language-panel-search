import anthropic
import os

# Claude API 클라이언트 초기화
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")  # 환경변수에서 API 키 가져오기
)

def getCleanQuery(raw_query: str) -> str:
    """
    Claude Sonnet 4.5를 사용하여 사용자 쿼리를 전처리합니다.
    오타, 불필요한 표현을 제거하고 명확한 문장으로 변환합니다.
    """
    
    system_prompt = """당신은 검색 쿼리 전처리 전문가입니다. 
사용자가 입력한 질문을 분석하여 다음 작업을 수행하세요:
1. 오타 및 맞춤법 오류 수정
2. 불필요한 조사나 표현 제거
3. 명확하고 간결한 검색용 문장으로 변환
4. 핵심 의도는 유지하되 검색에 최적화된 형태로 재구성

반드시 전처리된 문장만 출력하세요. 추가 설명은 하지 마세요."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",  # Claude Sonnet 4.5
            max_tokens=200,
            temperature=0.3,  # 일관성 있는 출력을 위해 낮은 temperature
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"다음 질문을 전처리하세요: {raw_query}"
                }
            ]
        )
        
        # 응답에서 텍스트 추출
        clean_query = message.content[0].text.strip()
        print(f"[Query Preprocessing] 원본: {raw_query}")
        print(f"[Query Preprocessing] 전처리: {clean_query}")
        
        return clean_query
        
    except Exception as e:
        print(f"[Query Preprocessing Error] {e}")
        # 오류 발생 시 원본 쿼리 반환
        return raw_query
