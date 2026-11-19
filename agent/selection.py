"""Turn a day's scores into a list split by company size.

The lists show 10 large caps (Nifty 100), 10 mid caps (Midcap 150) and 10
small caps (Smallcap 250). Microcaps are left out: they are harder to trade
in size and riskier for mid- and long-term holding.

When fewer than 10 stocks of a size pass the main filter (for the mid-term
list, the trend template), the gap is filled with the next best scores of the
same size and those rows are marked as filled.
"""

from __future__ import annotations

import pandas as pd

SEGMENTS = ("large", "mid", "small")
PER_SEGMENT = 10


def pick_by_segment(primary: pd.Series, fallback: pd.Series | None, segment: pd.Series,
                    n: int = PER_SEGMENT) -> list[dict]:
    """One day's picks: [{symbol, segment, score, rank, filled}], large first.

    primary  - scores of stocks that pass every filter (NaN elsewhere)
    fallback - scores used to fill a short segment (NaN where not allowed);
               None means no filling.
    """
    seg = segment.reindex(primary.index)
    out = []
    for name in SEGMENTS:
        mine = seg.eq(name)
        top = primary[mine].dropna().nlargest(n)
        rows = [(s, v, False) for s, v in top.items()]
        if fallback is not None and len(rows) < n:
            pool = fallback.reindex(primary.index)[mine].drop(top.index, errors="ignore")
            rows += [(s, v, True) for s, v in pool.dropna().nlargest(n - len(rows)).items()]
        out += [dict(symbol=s, segment=name, score=float(v), rank=i, filled=f)
                for i, (s, v, f) in enumerate(rows, 1)]
    return out
