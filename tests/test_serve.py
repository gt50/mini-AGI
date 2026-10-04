"""The chat endpoint trains on what it is sent, so who may send matters."""

import pytest


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("flask")
    import serve
    monkeypatch.setattr(serve, "ALLOWED_HOSTS", set(serve.LOOPBACK))
    return serve.app.test_client()


def test_refuses_non_json(client):
    r = client.post("/api/chat", data='{"messages": []}',
                    content_type="text/plain",
                    headers={"Host": "127.0.0.1:8080"})
    assert r.status_code == 415


def test_refuses_foreign_origin(client):
    r = client.post("/api/chat", json={"messages": []},
                    headers={"Host": "127.0.0.1:8080",
                             "Origin": "https://evil.example"})
    assert r.status_code == 403


def test_refuses_rebound_host(client):
    r = client.post("/api/chat", json={"messages": []},
                    headers={"Host": "evil.example:8080",
                             "Origin": "http://evil.example:8080"})
    assert r.status_code == 403
