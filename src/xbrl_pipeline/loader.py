# -*- coding: utf-8 -*-
import os
import logging
from typing import Tuple, Optional, Dict, Any
from arelle import Cntlr, FileSource

logger = logging.getLogger(__name__)

def load_model_xbrl(entry_path: str):
    """
    使用 Arelle 加载 iXBRL HTML / XBRL 实例 / *_htm.xml。
    Arelle 的 DTS 发现会自动把 xsd / cal / pre / def / lab 全部加载进来。
    """
    if not os.path.exists(entry_path):
        raise FileNotFoundError(entry_path)
    cntlr = Cntlr.Cntlr(logFileName=None)
    fs = FileSource.openFileSource(entry_path, cntlr)
    model_xbrl = cntlr.modelManager.load(filesource=fs)
    if model_xbrl is None:
        raise RuntimeError("Arelle 加载失败，请检查入口文件与依赖。")
    logger.info("Loaded model: facts=%d, contexts=%d, units=%d",
                len(model_xbrl.facts), len(model_xbrl.contexts), len(model_xbrl.units))
    return cntlr, model_xbrl

def guess_filing_meta(entry_path: str) -> Dict[str, Any]:
    """
    从路径猜测 filing 元信息（可选，用于落库附带字段）。
    例如：…/0000320193-20-000096/aapl-20200926_htm.xml
    """
    base = os.path.basename(entry_path)
    parent = os.path.basename(os.path.dirname(entry_path))
    return {
        "source_path": os.path.abspath(entry_path),
        "accession_no": parent if "-" in parent else None,
        "ticker_hint": base.split("-")[0] if "-" in base else None,
    }

def close(cntlr):
    """释放 Arelle 资源（可选）。"""
    try:
        cntlr.modelManager.close()
    except Exception:
        pass
