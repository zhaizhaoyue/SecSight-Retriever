from __future__ import annotations
from typing import Dict, Any, Iterable, List

# Simple metadata-based filtering utilities

def match_meta(meta: Dict[str, Any], ticker=None, year=None, form=None):
    if ticker and (str(meta.get("ticker") or "").upper() != str(ticker).upper()):
        return False
    if form and (str(meta.get("form") or "").upper() != str(form).upper()):
        return False
    # Year: try fy or year
    if year is not None:
        fy = meta.get("fy") or meta.get("year")
        try:
            if int(fy) != int(year):
                return False
        except Exception:
            return False
    return True


def filter_hits(hits: Iterable[Dict[str, Any]], ticker=None, year=None, form=None) -> List[Dict[str, Any]]:
    return [h for h in hits if match_meta(h.get("meta", {}), ticker, year, form)]