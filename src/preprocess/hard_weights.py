"""hard_score → 증강 가중치 공유 수식 (stdlib 전용).

hard_cases.py(pandas 의존)와 llm_caption_augment.py(RunPod에서 stdlib만으로
실행되는 계약) 양쪽이 같은 수식을 쓰도록 한 곳에 둔다. 예전에는 두 파일이
각자 수식을 들고 있어(max_extra_repeats vs max_repeats-1) 기본값에서만
우연히 일치했다.
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
