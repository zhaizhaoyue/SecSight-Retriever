# -*- coding: utf-8 -*-
import os
import logging
from typing import Tuple, Optional, Dict, Any
from arelle import Cntlr, FileSource

logger = logging.getLogger(__name__)

def load_model_xbrl(entry_path: str):
    """
    [TRANSLATED] Arelle [TRANSLATED] iXBRL HTML / XBRL [TRANSLATED] / *_htm.xml。
    Arelle [TRANSLATED] DTS [TRANSLATED] xsd / cal / pre / def / lab [TRANSLATED]。
    """
    if not os.path.exists(entry_path):
        raise FileNotFoundError(entry_path)
    cntlr = Cntlr.Cntlr(logFileName=None)
    fs = FileSource.openFileSource(entry_path, cntlr)
    model_xbrl = cntlr.modelManager.load(filesource=fs)
    if model_xbrl is None:
        raise RuntimeError("Arelle [TRANSLATED]，[TRANSLATED]。")
    logger.info("Loaded model: facts=%d, contexts=%d, units=%d",
                len(model_xbrl.facts), len(model_xbrl.contexts), len(model_xbrl.units))
    return cntlr, model_xbrl

def guess_filing_meta(entry_path: str) -> Dict[str, Any]:
    """
    [TRANSLATED] filing [TRANSLATED]（[TRANSLATED]，[TRANSLATED]）。
    [TRANSLATED]：…/0000320193-20-000096/aapl-20200926_htm.xml
    """
    base = os.path.basename(entry_path)
    parent = os.path.basename(os.path.dirname(entry_path))
    return {
        "source_path": os.path.abspath(entry_path),
        "accession_no": parent if "-" in parent else None,
        "ticker_hint": base.split("-")[0] if "-" in base else None,
    }

def close(cntlr):
    """[TRANSLATED] Arelle [TRANSLATED]（[TRANSLATED]）。"""
    try:
        cntlr.modelManager.close()
    except Exception:
        pass
