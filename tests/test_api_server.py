from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_crawl_simple_stub(monkeypatch, tmp_path):
    def _stub_run_simple(query, max_results, anchor, download_count, out_dir):
        assert query == "PETase"
        assert max_results == 3
        assert anchor is None
        assert download_count == 0
        assert out_dir == tmp_path / "run_test"
        return {"metadata": "meta.json", "downloads": "log.json", "count": 1}

    monkeypatch.setattr(api_server, "run_simple", _stub_run_simple)

    payload = {
        "query": "PETase",
        "max_results": 3,
        "download_count": 0,
        "output_root": str(tmp_path),
        "run_name": "run_test",
    }
    resp = client.post("/crawl/simple", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_dir"].endswith("run_test")
    assert body["result"]["count"] == 1
