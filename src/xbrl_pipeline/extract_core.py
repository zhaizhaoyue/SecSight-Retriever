# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any, Optional
from arelle.ModelDtsObject import ModelConcept

logger = logging.getLogger(__name__)

def _period_tuple(ctx) -> Dict[str, Any]:
    if ctx.isForeverPeriod:
        return {"start": None, "end": None, "instant": None, "is_forever": True}
    if ctx.isInstantPeriod:
        return {
            "start": None,
            "end": None,
            "instant": getattr(ctx, "instantDatetime", None).date() if getattr(ctx, "instantDatetime", None) else None,
            "is_forever": False,
        }
    if ctx.isStartEndPeriod:
        return {
            "start": getattr(ctx, "startDatetime", None).date() if getattr(ctx, "startDatetime", None) else None,
            "end": getattr(ctx, "endDatetime", None).date() if getattr(ctx, "endDatetime", None) else None,
            "instant": None,
            "is_forever": False,
        }
    return {"start": None, "end": None, "instant": None, "is_forever": False}

def extract_contexts(model_xbrl) -> List[Dict[str, Any]]:
    out = []
    for ctx in model_xbrl.contexts.values():
        dims = []
        if ctx.qnameDims:
            for dimQn, dimVal in ctx.qnameDims.items():
                member = getattr(dimVal, "memberQname", None)
                dims.append({
                    "dimension": str(dimQn),
                    "member": str(member) if member else str(dimVal)
                })
        ent = None
        try:
            ent = ctx.entityIdentifier[1] if ctx.entityIdentifier else None
        except Exception:
            ent = None
        p = _period_tuple(ctx)
        out.append({
            "context_id": ctx.id,
            "entity": ent,
            "period_start": p["start"],
            "period_end": p["end"],
            "period_instant": p["instant"],
            "is_forever": p["is_forever"],
            "dimensions": dims
        })
    logger.info("contexts extracted: %d", len(out))
    return out

def extract_units(model_xbrl) -> List[Dict[str, Any]]:
    out = []
    for unit in model_xbrl.units.values():
        measures = {"numerators": [], "denominators": []}
        if unit.measures:
            if len(unit.measures) > 0 and unit.measures[0]:
                measures["numerators"] = [str(q) for q in unit.measures[0]]
            if len(unit.measures) > 1 and unit.measures[1]:
                measures["denominators"] = [str(q) for q in unit.measures[1]]
        out.append({
            "unit_id": unit.id,
            "measures": measures
        })
    logger.info("units extracted: %d", len(out))
    return out

def extract_facts(model_xbrl) -> List[Dict[str, Any]]:
    out = []
    for f in model_xbrl.facts:
        concept: Optional[ModelConcept] = f.concept
        if concept is None:
            continue
        rec = {
            "fact_oid": f.objectId(),                 # [TRANSLATED]
            "qname": str(concept.qname),              # us-gaap:Revenues
            "value_raw": f.value,
            "is_numeric": bool(f.isNumeric),
            "decimals": getattr(f, "decimals", None),
            "unit_id": f.unitID if f.isNumeric else None,
            "context_id": f.contextID,
            "label_role": getattr(f, "preferredLabel", None),
            "footnote_refs": list(getattr(f, "footnoteRefs", []) or []),
            "doc_order": getattr(f, "sourceline", None),
        }
        out.append(rec)
    logger.info("facts extracted: %d", len(out))
    return out
