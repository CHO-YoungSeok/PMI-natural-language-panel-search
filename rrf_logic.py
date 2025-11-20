# rrf_logic.py

def rrf_rank(results, k=60):
    """
    Reciprocal Rank Fusion (RRF) 점수 계산
    results: [{ "id": ..., "score": ..., "combined_score": ..., "distance": ... }]
    """
    ranked = []

    for idx, r in enumerate(results):
        # idx는 0부터 시작하므로 rank = idx + 1
        rank = idx + 1

        # 기존 score + 사전 보정 점수(combined_score) 모두 반영 가능
        base_score = r.get("combined_score", r.get("score", 0))

        # RRF 점수 계산
        rrf_score = 1 / (k + rank)

        ranked.append({
            "id": r["id"],
            "data": r["data"],
            "distance": r["distance"],
            "score": base_score,
            "rrf_score": rrf_score
        })

    return ranked
