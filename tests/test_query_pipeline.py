import json
from types import SimpleNamespace
import sys                                                                                                                                
from pathlib import Path                                                                                                                  
                                                                                                                                               
ROOT = Path(__file__).resolve().parents[1]                                                                                                
if str(ROOT) not in sys.path:                                                                                                             
    sys.path.insert(0, str(ROOT))   
import pytest

from src import cli
from src.retrieval import pipeline as qp


class _StubHybrid:
    def __init__(self, records):
        self._records = records
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return list(self._records)


def _make_dirs(tmp_path):
    index_dir = tmp_path / "index"
    chunk_dir = tmp_path / "chunk"
    index_dir.mkdir()
    chunk_dir.mkdir()
    return index_dir, chunk_dir


def test_run_query_returns_records(monkeypatch, tmp_path):
    index_dir, chunk_dir = _make_dirs(tmp_path)
    records = [{"id": "RID-1", "score": 0.9, "snippet": "hello", "meta": {"ticker": "AAPL"}}]
    stub = _StubHybrid(records)
    monkeypatch.setattr(qp, "_build_hybrid", lambda req: stub)

    result = qp.run_query(qp.QueryRequest(
        query="What is revenue?",
        index_dir=index_dir,
        content_dir=chunk_dir,
        topk=5,
    ))

    assert result.query == "What is revenue?"
    assert result.records == records
    assert result.answer is None
    assert stub.calls and stub.calls[0][0] == "What is revenue?"


def test_run_query_with_llm(monkeypatch, tmp_path):
    index_dir, chunk_dir = _make_dirs(tmp_path)
    records = [{"id": "RID-2", "score": 0.8}]
    stub = _StubHybrid(records)
    monkeypatch.setattr(qp, "_build_hybrid", lambda req: stub)

    expected_answer = {"answer": "Answer", "citations": [{"rid": "RID-2", "quote": "42"}]}
    called = {}

    def fake_answer_with_llm(query, recs, llm, max_ctx_tokens):
        called["args"] = (query, recs, max_ctx_tokens)
        return expected_answer

    monkeypatch.setattr(qp, "answer_with_llm", fake_answer_with_llm)

    result = qp.run_query(qp.QueryRequest(
        query="Another question",
        index_dir=index_dir,
        content_dir=chunk_dir,
        llm_base_url="http://localhost",
        llm_model="dummy",
        llm_api_key="key",
    ))

    assert result.answer == expected_answer
    assert called["args"][0] == "Another question"
    assert called["args"][2] == 2400


def test_cli_command_query_json(monkeypatch, tmp_path, capsys):
    index_dir, chunk_dir = _make_dirs(tmp_path)

    query_result = qp.QueryResult(
        query="Sample",
        records=[{"id": "RID-3", "score": 1.0, "snippet": "text", "meta": {"ticker": "AAPL"}}],
        answer={"answer": "LLM answer", "citations": [{"rid": "RID-3", "quote": "sample"}]},
    )

    captured_request = {}

    def fake_execute_query(req):
        captured_request["index_dir"] = req.index_dir
        captured_request["query"] = req.query
        return query_result

    monkeypatch.setattr(cli, "execute_query", fake_execute_query)

    args = SimpleNamespace(
        query="Sample",
        index_dir=str(index_dir),
        chunk_dir=str(chunk_dir),
        bm25_meta=None,
        dense_model="m",
        dense_device="cpu",
        topk=5,
        bm25_topk=100,
        dense_topk=100,
        ce_candidates=128,
        rrf_k=60.0,
        w_bm25=2.0,
        w_dense=2.0,
        ce_weight=0.5,
        rerank_model="ce",
        rerank_device=None,
        ticker=None,
        form=None,
        year=None,
        llm_base_url=None,
        llm_model=None,
        llm_api_key=None,
        max_context_tokens=2400,
        loose_filters=False,
        snippet_chars=0,
        json_out=True,
    )

    cli.command_query(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["answer"]["answer"] == "LLM answer"
    assert captured_request["index_dir"] == index_dir
