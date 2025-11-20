from fastapi import FastAPI
from pydantic import BaseModel
import sys, os, time, re
from fastapi.middleware.cors import CORSMiddleware
from db_search import fts_search, vector_search
from sonnet_api import preprocess_query, llm_filter_panel
from rrf_logic import rrf_rank
from query_vectorizer import get_query_vector
import numpy as np
import anthropic, os
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


# msg = preprocess_query('서ㅇㅁ울에 사는 20대 여잨 중 운동 즉 엑설사이즈를 좋아하는 사람 20명')
# print(msg)

# print(os.getenv("POSTGRES_DB_PASSWORD"))


query_text = "운동 좋아하는 20대 여성"
query_vec = get_query_vector(query_text)
results = vector_search(query_vec, top_k=10)

for r in results:
    print(r)
