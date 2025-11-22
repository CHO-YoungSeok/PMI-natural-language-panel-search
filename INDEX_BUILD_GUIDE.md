# BM25 인덱스 구축 가이드

## 개요

이 프로젝트는 한국어 최적화된 BM25 검색을 위해 사전에 인덱스를 구축하는 방식을 사용합니다.
인덱스를 미리 구축하면 검색 시 매번 형태소 분석을 할 필요가 없어 응답 속도가 크게 향상됩니다.

## 아키텍처

```
┌─────────────────┐
│  build_index.py │  (1회 실행)
│                 │
│  1. DB 조회     │
│  2. 형태소 분석 │
│  3. 전처리      │
│  4. BM25 구축   │
│  5. Pickle 저장 │
└────────┬────────┘
         │
         ▼
  bm25_index.pkl
         │
         ▼
┌────────┴────────┐
│   db_search.py  │  (검색 시 사용)
│                 │
│  1. 캐시 로드   │
│  2. 쿼리 전처리 │
│  3. BM25 검색   │
│  4. 결과 반환   │
└─────────────────┘
```

## 설치

### 1. 패키지 설치

```bash
# requirements.txt 기반 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install kiwipiepy rank-bm25 psycopg2-binary python-dotenv
```

### 2. 환경 변수 설정

`.env` 파일에 데이터베이스 비밀번호 설정:

```bash
POSTGRES_DB_PASSWORD=your_password_here
```

## 인덱스 구축

### 첫 실행

처음 시스템을 시작할 때 인덱스를 구축해야 합니다:

```bash
python build_index.py
```

### 실행 과정

```
================================================================================
BM25 인덱스 구축 시작
================================================================================

[1/4] 데이터베이스 연결 중...
✓ 데이터베이스 연결 성공

[2/4] 문서 데이터 로드 중...
✓ 총 15000개 문서 로드 완료

[3/4] 형태소 분석 및 전처리 중...
  (이 과정은 시간이 걸릴 수 있습니다...)
  진행: 1000/15000 (6.7%) [50.2 docs/sec, 남은시간: 4.7분]
  진행: 2000/15000 (13.3%) [52.1 docs/sec, 남은시간: 4.2분]
  ...

✓ 전처리 완료: 14950개 문서 인덱싱 (280.5초 소요)
  ⚠ 50개 문서는 전처리 후 빈 토큰으로 제외되었습니다.

[4/4] BM25 인덱스 생성 중...
✓ BM25 인덱스 생성 완료 (2.34초 소요)

인덱스 저장 중: bm25_index.pkl
✓ 인덱스 저장 완료
  파일 크기: 45.23 MB
  저장 위치: /path/to/RRF_Search/bm25_index.pkl

인덱스 파일 검증 중: bm25_index.pkl
✓ 인덱스 로드 성공
  인덱싱된 문서 수: 14950개
  샘플 문서 ID: ['uuid-1', 'uuid-2', 'uuid-3', 'uuid-4', 'uuid-5']

테스트 쿼리: '운동 좋아하는 20대'
전처리된 토큰: ['운동', '좋아하', '20', '대']
상위 3개 결과:
  1. ID: uuid-1234, Score: 12.4567
  2. ID: uuid-5678, Score: 10.2345
  3. ID: uuid-9012, Score: 9.8765

✓ 인덱스 검증 완료: 정상 작동

================================================================================
 전체 작업 완료 (총 소요시간: 4.88분)
================================================================================
```

### 성능 지표

| 문서 수 | 전처리 시간 | 인덱스 생성 | 파일 크기 | 총 소요 |
|--------|-----------|-----------|----------|---------|
| 1,000 | ~20초 | ~0.5초 | ~3 MB | ~21초 |
| 5,000 | ~1.5분 | ~1초 | ~15 MB | ~1.6분 |
| 10,000 | ~3분 | ~2초 | ~30 MB | ~3.2분 |
| 50,000 | ~15분 | ~10초 | ~150 MB | ~15.5분 |

*실제 성능은 하드웨어와 문서 길이에 따라 다를 수 있습니다.*

## 전처리 과정

`preprocess_text()` 함수가 각 문서에 수행하는 전처리:

### 1. 정규화
```python
# 원본: "운동 좋아하구!! 여행두 좋아요ㅋㅋㅋ ♥♥"
# 결과: "운동 좋아하구 여행두 좋아요"
```

### 2. 형태소 분석
```python
# Kiwi를 사용한 형태소 분석
# "운동 좋아하는" → [('운동', 'NNG'), ('좋아하', 'VV'), ('는', 'ETM')]
```

### 3. 품사 필터링
```python
# 의미있는 품사만 추출:
# - 명사(NN*): 일반명사, 고유명사, 의존명사
# - 동사(VV): 동사
# - 형용사(VA): 형용사
# - 영어(SL): 영어 단어
# - 숫자(SN): 숫자
# - 부사(MAG): 일반부사
```

### 4. 불용어 제거
```python
# 검색에 불필요한 단어 제거:
# ['이', '가', '을', '를', '은', '는', '의', '에', '있다', '없다', ...]
```

### 5. 단일 문자 제거
```python
# 의미 없는 한글자 단어 제거 (단, 영어/숫자는 보존)
# "운동 좋아하는 사 람" → ['운동', '좋아하', '사람']
```

## 인덱스 업데이트

### 언제 인덱스를 재구축해야 하나?

다음과 같은 경우 인덱스를 재구축해야 합니다:

1. **데이터베이스에 새 문서 추가**
   - 새로운 패널 데이터가 추가된 경우

2. **문서 내용 변경**
   - 기존 문서의 `info_text`가 수정된 경우

3. **검색 품질 개선**
   - 불용어 리스트 변경
   - 전처리 로직 개선

### 재구축 방법

```bash
# 1. 기존 인덱스 백업 (선택사항)
mv bm25_index.pkl bm25_index.pkl.backup

# 2. 새 인덱스 구축
python build_index.py

# 3. 서버 재시작
# (FastAPI는 다음 요청 시 자동으로 새 인덱스 로드)
```

### 자동 업데이트 (선택사항)

Cron을 사용하여 주기적으로 인덱스 재구축:

```bash
# crontab -e
# 매일 새벽 3시에 인덱스 재구축
0 3 * * * cd /path/to/RRF_Search && /usr/bin/python build_index.py
```

## 문제 해결

### 1. 메모리 부족

```
MemoryError: Unable to allocate array
```

**해결책**:
- 서버 메모리 증설
- 또는 배치 처리 방식으로 변경 (문서를 나눠서 처리)

```python
# build_index.py에서 배치 크기 조정
BATCH_SIZE = 5000  # 한 번에 처리할 문서 수
```

### 2. 데이터베이스 연결 실패

```
✗ 데이터베이스 연결 실패: connection refused
```

**확인 사항**:
- `.env` 파일에 올바른 비밀번호 설정
- 데이터베이스 서버 실행 중인지 확인
- 네트워크 연결 확인

### 3. 패키지 미설치

```
ModuleNotFoundError: No module named 'kiwipiepy'
```

**해결책**:
```bash
pip install kiwipiepy rank-bm25
```

### 4. 인덱스 파일이 없음

```
✗ BM25 인덱스 캐시 파일(bm25_index.pkl)을 찾을 수 없습니다.
```

**해결책**:
```bash
python build_index.py
```

### 5. 형태소 분석 느림

**최적화 방법**:
- Kiwi는 이미 최적화된 라이브러리입니다
- 멀티프로세싱 사용 가능:

```python
from multiprocessing import Pool

def process_doc(doc):
    return preprocess_text(doc[1])

with Pool(processes=4) as pool:
    tokenized_docs = pool.map(process_doc, documents)
```

## 검색 사용

인덱스 구축 후 검색 사용:

### Python에서 직접 사용

```python
from db_search import fts_search

# BM25 검색
results = fts_search("운동 좋아하는 20대 여성", top_k=10)
print(f"검색 결과: {results}")
```

### FastAPI 서버 실행

```bash
uvicorn search:app --host 0.0.0.0 --port 8000 --reload
```

### API 호출

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "운동 좋아하는 20대 여성", "count": 300}'
```

## 캐시 전략

### 현재 구현: 파일 기반 캐시

```python
# db_search.py
with open('bm25_index.pkl', 'rb') as f:
    bm25, doc_ids = pickle.load(f)
```

**장점**:
- 구현 간단
- 서버 재시작 시에도 캐시 유지

**단점**:
- 로드 시간 있음 (수 초)
- 멀티프로세스 환경에서 비효율적

### 대안: Redis 캐시 (추후 개선)

```python
import redis
import pickle

r = redis.Redis(host='localhost', port=6379)

# 저장
r.set('bm25_index', pickle.dumps((bm25, doc_ids)))

# 로드
bm25, doc_ids = pickle.loads(r.get('bm25_index'))
```

**장점**:
- 빠른 로드 속도
- 멀티프로세스 공유 가능
- 분산 환경 지원

## 성능 모니터링

### 인덱스 크기 확인

```bash
ls -lh bm25_index.pkl
```

### 검색 속도 측정

```python
import time

start = time.time()
results = fts_search("운동 좋아하는 20대", top_k=100)
elapsed = time.time() - start

print(f"검색 시간: {elapsed*1000:.2f}ms")
print(f"결과 수: {len(results)}")
```

### 예상 성능

| 문서 수 | 인덱스 로드 | 검색 시간 | 메모리 사용 |
|--------|-----------|----------|-----------|
| 1,000 | ~50ms | ~5ms | ~50 MB |
| 10,000 | ~200ms | ~20ms | ~200 MB |
| 50,000 | ~1s | ~50ms | ~800 MB |
| 100,000 | ~2s | ~100ms | ~1.5 GB |

## 참고 자료

- **Kiwi (한국어 형태소 분석기)**: https://github.com/bab2min/kiwipiepy
- **rank-bm25 (BM25 구현)**: https://github.com/dorianbrown/rank_bm25
- **BM25 알고리즘**: https://en.wikipedia.org/wiki/Okapi_BM25
- **형태소 분석 품사 태그**: https://github.com/bab2min/kiwipiepy#품사-태그
