# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any
from arelle import XbrlConst

logger = logging.getLogger(__name__)

def _collect_edges(model_xbrl, arcrole: str) -> List[Dict[str, Any]]:
    rs = model_xbrl.relationshipSet(arcrole)
    if not rs:
        return []
    edges = []
    for rel in rs.modelRelationships:
        frm = rel.fromModelObject
        to = rel.toModelObject
        if not frm or not to:
            continue
        edges.append({
            "from_qname": str(getattr(frm, "qname", "")),
            "to_qname": str(getattr(to, "qname", "")),
            "weight": getattr(rel, "weight", None),
            "order": getattr(rel, "order", None),
            "preferred_label": getattr(rel, "preferredLabel", None),
            "arcrole": arcrole
        })
    return edges

def extract_calc_edges(model_xbrl) -> List[Dict[str, Any]]:
    e = _collect_edges(model_xbrl, XbrlConst.summationItem)
    logger.info("calc edges: %d", len(e))
    return e

def extract_pre_edges(model_xbrl) -> List[Dict[str, Any]]:
    e = _collect_edges(model_xbrl, XbrlConst.parentChild)
    logger.info("pre edges: %d", len(e))
    return e

def extract_def_edges(model_xbrl) -> List[Dict[str, Any]]:
    arcs = [
        XbrlConst.domainMember,
        XbrlConst.dimensionDomain,
        XbrlConst.all,
        XbrlConst.notAll,
        XbrlConst.hypercubeDimension
    ]
    out = []
    for a in arcs:
        out += _collect_edges(model_xbrl, a)
    logger.info("def edges: %d", len(out))
    return out
