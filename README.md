# RRF Search - 한국어 하이브리드 검색 시스템

PMI RAG (Retrieval-Augmented Generation) 시스템으로, BM25와 Vector Search를 결합한 하이브리드 검색을 제공합니다.

## 주요 기능

- **하이브리드 검색**: BM25 + Vector Similarity Search
- **RRF (Reciprocal Rank Fusion)**: 두 검색 결과를 효과적으로 통합
- **한국어 최적화**: Kiwi 형태소 분석기 기반 전처리
- **LLM 기반 필터링**: Claude Sonnet을 사용한 쿼리 전처리 및 결과 필터링
- **고성능**: 인덱스 캐싱으로 빠른 검색 속도

## 시스템 아키텍처

```
사용자 쿼리
    │
    ▼
┌──────────────────────────────────┐
│   Claude Sonnet 쿼리 전처리       │
│   (오타 수정, 불필요한 단어 제거)  │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────────┐
│   BM25  │  │   Vector    │
│ Search  │  │   Search    │
│(형태소)  │  │ (KURE-v1)   │
└────┬────┘  └──────┬──────┘
     │              │
     └──────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  RRF 융합     │
    │  (k=60)       │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Claude Sonnet │
    │  필터링 & 랭킹 │
    └───────┬───────┘
            │
            ▼
       최종 결과
```

## 빠른 시작

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일 생성)
echo "POSTGRES_DB_PASSWORD=your_password" > .env
```

### 2. BM25 인덱스 구축

**중요**: 첫 실행 전 반드시 인덱스를 구축해야 합니다.

```bash
python build_index.py
```

이 과정은 다음을 수행합니다:
- 데이터베이스에서 전체 문서 로드
- 한국어 형태소 분석 및 전처리
- BM25 인덱스 생성
- `bm25_index.pkl` 파일로 저장

**예상 소요 시간**:
- 1만 문서: ~3분
- 5만 문서: ~15분

### 3. 서버 실행

```bash
uvicorn search:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 검색 테스트

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "운동 좋아하는 20대 여성",
    "count": 300
  }'
```

## 프로젝트 구조

```
RRF_Search/
├── search.py                  # FastAPI 메인 애플리케이션
├── db_search.py              # 검색 함수들 (fts_search, vector_search 등)
├── rrf_logic.py              # RRF 알고리즘 구현
├── query_vectorizer.py       # KURE-v1 벡터화
├── sonnet_api.py             # Claude Sonnet API 통합
├── build_index.py            # BM25 인덱스 빌더 (중요!)
│
├── bm25_index.pkl            # BM25 인덱스 캐시 (생성됨)
├── .env                      # 환경 변수
├── requirements.txt          # Python 패키지
│
├── index.html                # 웹 프론트엔드 UI
│
├── test_bm25.py             # BM25 검색 테스트
├── test_fts_korean.py       # 한국어 FTS 진단
├── llmPreprocessTest.py     # LLM 전처리 테스트
│
└── 문서/
    ├── README.md            # 이 파일
    ├── INDEX_BUILD_GUIDE.md # 인덱스 구축 상세 가이드
    ├── BM25_SETUP.md        # BM25 설정 가이드 (ParadeDB)
    └── FTS_KOREAN_SETUP.md  # 한국어 FTS 설정 가이드
```

## 검색 방법 비교

| 방법 | 기술 | 장점 | 단점 |
|------|------|------|------|
| **FTS (현재 사용)** | Python BM25 + Kiwi | 한국어 최적화, 형태소 분석, 빠름 | 인덱스 재구축 필요 |
| **Vector Search** | KURE-v1 embeddings | 의미론적 유사도, 동의어 처리 | 짧은 쿼리에 약함 |
| **RRF** | 랭크 융합 | 두 방법의 장점 결합 | - |

## 한국어 전처리 과정

`preprocess_text()` 함수가 수행하는 단계:

1. **정규화**: 특수문자, 이모티콘 제거
   ```
   "운동 좋아하구!! ㅋㅋㅋ ♥" → "운동 좋아하구"
   ```

2. **형태소 분석**: Kiwi 사용
   ```
   "운동 좋아하는" → [('운동', 'NNG'), ('좋아하', 'VV'), ('는', 'ETM')]
   ```

3. **품사 필터링**: 명사, 동사, 형용사만 추출
   ```
   [('운동', 'NNG'), ('좋아하', 'VV')] → ['운동', '좋아하']
   ```

4. **불용어 제거**: 조사, 접속사 등 제거
   ```
   ['이', '운동', '을', '좋아하'] → ['운동', '좋아하']
   ```

5. **단일 문자 제거**: 의미 없는 한글자 제거
   ```
   ['운', '동', '좋아하'] → ['좋아하']
   ```

## API 엔드포인트

### POST /ask

하이브리드 검색 수행

**요청**:
```json
{
  "query": "운동 좋아하는 20대 여성",
  "count": 300
}
```

**응답**:
```json
{
  "original_query": "운동 좋아하는 20대 여성",
  "clean_query": "운동 좋아하는 20대 여성",
  "result": [
    {
      "id": "uuid-1234",
      "info_text": {...}
    }
  ],
  "metrics": {
    "total_time": "2.34s",
    "final_count": 10
  }
}
```

### GET /

헬스 체크

**응답**:
```json
{
  "message": "RRF Search API with Sonnet running"
}
```

## 인덱스 업데이트

데이터베이스에 새 문서가 추가되거나 변경된 경우:

```bash
# 기존 인덱스 백업 (선택)
mv bm25_index.pkl bm25_index.pkl.backup

# 새 인덱스 구축
python build_index.py

# 서버 재시작 (FastAPI reload 모드라면 자동)
```

### 자동 업데이트 (Cron)

```bash
# crontab -e
# 매일 새벽 3시에 인덱스 재구축
0 3 * * * cd /path/to/RRF_Search && python build_index.py
```

## 성능 최적화

### 검색 속도

| 문서 수 | 인덱스 로드 | 검색 시간 | 총 응답 시간 |
|--------|-----------|----------|-------------|
| 1만 | ~200ms | ~20ms | ~2-3초 |
| 5만 | ~1s | ~50ms | ~2-3초 |
| 10만 | ~2s | ~100ms | ~3-4초 |

*총 응답 시간에는 LLM 호출 시간(~1-2초) 포함*

### 메모리 사용

| 문서 수 | 인덱스 크기 | 메모리 사용 |
|--------|-----------|-----------|
| 1만 | ~30 MB | ~200 MB |
| 5만 | ~150 MB | ~800 MB |
| 10만 | ~300 MB | ~1.5 GB |

## 테스트

### 1. BM25 검색 테스트

```bash
python test_bm25.py
```

다음을 테스트합니다:
- BM25, FTS, Vector 검색 비교
- RRF 융합 결과
- Sonnet 전처리 + 검색 파이프라인

### 2. 한국어 FTS 진단

```bash
python test_fts_korean.py
```

다음을 확인합니다:
- FTS 토큰화 방식
- 다양한 검색 방법 비교
- pg_trgm 사용 가능 여부

## 문제 해결

### 1. 인덱스 파일이 없음

```
✗ BM25 인덱스 캐시 파일(bm25_index.pkl)을 찾을 수 없습니다.
```

**해결**: `python build_index.py` 실행

### 2. 패키지 미설치

```
ModuleNotFoundError: No module named 'kiwipiepy'
```

**해결**: `pip install -r requirements.txt`

### 3. 데이터베이스 연결 실패

```
✗ 데이터베이스 연결 실패: connection refused
```

**확인**:
- `.env` 파일에 올바른 비밀번호 설정
- 데이터베이스 서버 실행 중인지 확인
- 네트워크 연결 확인

### 4. 메모리 부족

**해결**:
- 서버 메모리 증설
- 배치 크기 조정 (build_index.py)
- Redis 캐시 사용 (추후 개선)

## 기술 스택

### 백엔드
- **FastAPI**: REST API 프레임워크
- **PostgreSQL**: 메인 데이터베이스
- **pgvector**: 벡터 유사도 검색

### NLP & 검색
- **Kiwi (kiwipiepy)**: 한국어 형태소 분석
- **rank-bm25**: BM25 알고리즘 구현
- **KURE-v1**: 한국어 문장 임베딩 (1024차원)

### LLM
- **Claude Sonnet 4.5**: 쿼리 전처리 및 결과 필터링
- **AWS Bedrock**: LLM 호스팅 (ap-southeast-2)

### 데이터베이스
- **AWS RDS PostgreSQL**: 데이터 저장소 (ap-southeast-2)

## 다음 단계 개선 사항

- [ ] Redis 기반 인덱스 캐싱 (멀티프로세스 지원)
- [ ] 증분 인덱스 업데이트 (전체 재구축 불필요)
- [ ] 멀티프로세싱 기반 인덱스 구축 (속도 향상)
- [ ] 검색 결과 캐싱 (동일 쿼리 재사용)
- [ ] 로깅 및 모니터링 추가
- [ ] A/B 테스트를 위한 검색 방법 선택 옵션

## 참고 자료

- **Kiwi**: https://github.com/bab2min/kiwipiepy
- **rank-bm25**: https://github.com/dorianbrown/rank_bm25
- **RRF 논문**: Cormack et al. (2009) - Reciprocal Rank Fusion
- **BM25 알고리즘**: https://en.wikipedia.org/wiki/Okapi_BM25
- **FastAPI**: https://fastapi.tiangolo.com

## 라이선스

이 프로젝트는 PMI 프로젝트의 일부입니다.

## 지원

문제가 발생하면 다음 문서를 참고하세요:
- [인덱스 구축 가이드](INDEX_BUILD_GUIDE.md)
- [BM25 설정 가이드](BM25_SETUP.md)
- [한국어 FTS 설정](FTS_KOREAN_SETUP.md)
