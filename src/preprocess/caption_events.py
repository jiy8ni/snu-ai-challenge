"""캡션을 시간 순서의 이벤트 절로 분해한다.

corpus 통계 (train 9,535개 기준):
  then 5,718 / as 3,747 / while 1,536 / followed by 978 / finally 928 /
  before(중간) 687 / ';' 552 / after(중간) 133 / next 103

순서 규칙:
- "A then B" / "A followed by B" / "A; finally B" 등: 서술 순서 = 시간 순서
- "A before B": A가 먼저 -> 서술 순서 = 시간 순서 (역전 없음)
- "A after B": B가 먼저 -> 역전
- 문두 "After X, Y": X가 먼저 -> 서술 순서 유지 / 문두 "Before X, Y": 역전
- "as / while": 동시 서술 -> 별도 이벤트로 나누지 않는다 (같은 절에 유지)

용도: ① No_ordering 판별 피처 (이벤트 수 vs 프레임 수 4)
      ② CoT 학습 타깃 자동 생성 (이벤트 절 + 정답 순서 결합)
"""

import re

# 서술 순서 = 시간 순서인 순차 연결어 (분할 지점)
_SEQ_SPLIT = re.compile(
    r"""
    ,?\s*(?:and\s+)?\bthen\b,?\s*
    | ,?\s+followed\s+by\s+
    | ,?\s*(?:and\s+)?\bfinally\b,?\s*
    | ,?\s*(?:and\s+)?\bnext\b,\s*
    | ,?\s+ending\s+with\s+
    | ,?\s*(?:and\s+)?\bafterwards?\b,?\s+
    | ,?\s*\bsubsequently\b,?\s*
    | ;\s*
    | ,?\s+before\s+          # "A before B" -> A 먼저, 서술 순서 유지
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "A after B" -> B가 먼저 (역전). 문두 "After X, Y"는 별도 처리.
_MID_AFTER = re.compile(r",?\s+after\s+", re.IGNORECASE)
_LEADING_AFTER = re.compile(r"^\s*after\s+(.+?),\s*(.+)$", re.IGNORECASE | re.DOTALL)
_LEADING_BEFORE = re.compile(r"^\s*before\s+(.+?),\s*(.+)$", re.IGNORECASE | re.DOTALL)


def _clean(clause):
    return clause.strip(" ,.;").strip()


def split_events(sentence):
    """캡션 -> 시간 순서대로 정렬된 이벤트 절 리스트."""
    if not sentence or not sentence.strip():
        return []

    # 문두 종속절 처리
    parts = None
    m = _LEADING_AFTER.match(sentence)
    if m:  # "After X, Y" -> X 먼저
        parts = [m.group(1), m.group(2)]
    else:
        m = _LEADING_BEFORE.match(sentence)
        if m:  # "Before X, Y" -> Y 먼저 (역전)
            parts = [m.group(2), m.group(1)]
    segments = parts if parts is not None else [sentence]

    events = []
    for seg in segments:
        # 순차 연결어로 분할 (서술 순서 = 시간 순서)
        chunks = _SEQ_SPLIT.split(seg)
        for chunk in chunks:
            if not chunk or not chunk.strip():
                continue
            # 절 내부의 "A after B" -> [B, A]로 역전
            sub = _MID_AFTER.split(chunk)
            for clause in reversed(sub) if len(sub) > 1 else sub:
                clause = _clean(clause)
                if clause:
                    events.append(clause)
    return events


def n_events(sentence):
    return len(split_events(sentence))
