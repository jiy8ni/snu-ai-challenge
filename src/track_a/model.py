"""Track A: 동결 SigLIP2 임베딩 위 Set-to-Rank 헤드.

설계 원칙 — **입력 프레임 순열에 대한 등변성(equivariance)을 구조로 보장**한다:
  - 프레임 토큰에는 위치 인코딩·슬롯 파생 피처를 절대 넣지 않는다 (집합으로 취급).
  - 프레임별 스칼라 피처는 대칭 함수(다른 프레임들에 대한 통계)만 허용.
  - 게이트 풀링은 순열 불변 연산(mean)만 사용.
등변이므로 학습 시 순열 증강과 추론 시 TTA가 모두 불필요하다.
검증: tests/test_track_a_model.py (입력 셔플 -> rank 로짓 동일 셔플, 게이트 불변).

출력:
  rank_logits (B, 4, 4): [i, r] = "입력 프레임 i의 시간 순위가 r+1"의 로짓
  gate_logit  (B,):      UNORDERABLE(No_ordering) 게이트
"""

import torch
import torch.nn as nn

N_FRAMES = 4
TYPE_FRAME, TYPE_EVENT, TYPE_CAPTION = 0, 1, 2


class OrderHead(nn.Module):
    def __init__(
        self,
        d_emb=768,
        d_model=256,
        n_scalars=8,
        n_layers=3,
        n_heads=8,
        d_ffn=512,
        dropout=0.1,
        max_events=6,
        use_events=True,
        use_scalars=True,
    ):
        super().__init__()
        self.use_events = use_events
        self.use_scalars = use_scalars
        self.n_scalars = n_scalars

        self.frame_proj = nn.Linear(d_emb + (n_scalars if use_scalars else 0), d_model)
        self.text_proj = nn.Linear(d_emb, d_model)  # 이벤트·캡션 공유
        self.type_emb = nn.Embedding(3, d_model)
        self.event_pos = nn.Embedding(max_events, d_model)  # 서술 순서 = 시간 순서 신호

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ffn, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.rank_head = nn.Linear(d_model, N_FRAMES)
        self.gate_head = nn.Sequential(
            nn.Linear(2 * d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, img, scalars, events, event_mask, caption):
        """img (B,4,D) | scalars (B,4,S) | events (B,E,D) | event_mask (B,E) bool(유효=True)
        | caption (B,D)"""
        b = img.size(0)
        frame_in = torch.cat([img, scalars], dim=-1) if self.use_scalars else img
        frames = self.frame_proj(frame_in) + self.type_emb.weight[TYPE_FRAME]

        cap = self.text_proj(caption).unsqueeze(1) + self.type_emb.weight[TYPE_CAPTION]

        if self.use_events:
            n_ev = events.size(1)
            ev = (
                self.text_proj(events)
                + self.type_emb.weight[TYPE_EVENT]
                + self.event_pos.weight[:n_ev].unsqueeze(0)
            )
            tokens = torch.cat([frames, ev, cap], dim=1)
            pad = torch.cat(
                [
                    torch.zeros(b, N_FRAMES, dtype=torch.bool, device=img.device),
                    ~event_mask,
                    torch.zeros(b, 1, dtype=torch.bool, device=img.device),
                ],
                dim=1,
            )
        else:
            tokens = torch.cat([frames, cap], dim=1)
            pad = torch.zeros(b, tokens.size(1), dtype=torch.bool, device=img.device)

        enc = self.encoder(tokens, src_key_padding_mask=pad)
        frame_out = enc[:, :N_FRAMES]

        rank_logits = self.rank_head(frame_out)                      # (B, 4, 4)
        pooled = frame_out.mean(dim=1)                               # 순열 불변
        gate_logit = self.gate_head(torch.cat([pooled, enc[:, -1]], dim=-1)).squeeze(-1)
        return rank_logits, gate_logit
