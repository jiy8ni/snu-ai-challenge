"""hard_score → 증강 가중치 공유 수식 (stdlib 전용).

hard_cases.py(재채굴 CSV 생성, pandas 의존)와 학습 시점 오버레이(vl_dataset의
hard_cases_path)가 같은 수식을 쓰도록 한 곳에 둔다. 가중치는 전부 모델 자신의
예측 기반이며 외부 API와 무관하다.
"""

import math


def clamp01(value):
    return min(1.0, max(0.0, float(value)))


def repeats_from_score(score, max_repeats):
    """물리 복제 수 1..max_repeats. score 0→1, score 1→max_repeats."""
    max_repeats = int(max_repeats)
    repeats = 1 + int(math.ceil(clamp01(score) * (max_repeats - 1)))
    return max(1, min(max_repeats, repeats))


def prob_from_score(score, base_prob, hard_prob):
    """캡션 증강 확률 보간: score 0→base_prob, score 1→hard_prob."""
    return clamp01(base_prob + (hard_prob - base_prob) * clamp01(score))
