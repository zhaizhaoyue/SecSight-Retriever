# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any, Iterable
from arelle import XbrlConst

logger = logging.getLogger(__name__)

DEFAULT_ROLES: Iterable[str] = (
    XbrlConst.standardLabel,
    XbrlConst.documentationLabel,
    "http://www.xbrl.org/2003/role/terseLabel",
    "http://www.xbrl.org/2003/role/verboseLabel",
    "http://www.xbrl.org/2003/role/totalLabel",
    "http://www.xbrl.org/2003/role/negatedLabel",
)

DEFAULT_LANGS: Iterable[str] = ("en-US", "en")

def extract_labels(model_xbrl,
                   roles: Iterable[str] = DEFAULT_ROLES,
                   langs: Iterable[str] = DEFAULT_LANGS) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for qname, concept in model_xbrl.qnameConcepts.items():
        for role in roles:
            rs = model_xbrl.relationshipSet(XbrlConst.conceptLabel, role=role)
            if not rs:
                continue
            for rel in rs.fromModelObject(concept):
                res = rel.toModelObject
                if res is None:
                    continue
                lang = getattr(res, "xmlLang", None)
                if langs and lang not in langs:
                    continue
                out.append({
                    "qname": str(qname),
                    "role": role,
                    "lang": lang,
                    "text": res.text
                })
    logger.info("labels extracted: %d", len(out))
    return out
