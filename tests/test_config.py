import os

import pytest

from minagi import config


def test_take_config_flag(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  d_model: 7\n", encoding="utf-8")
    monkeypatch.delenv(config.ENV, raising=False)
    argv = ["train.py", "--config", str(p), "read", "x"]
    config.take_config_flag(argv)
    assert argv == ["train.py", "read", "x"]
    assert os.environ[config.ENV] == str(p)
    assert config.get(config.load(), "model.d_model") == 7

    argv = ["serve.py", f"--config={p}"]
    config.take_config_flag(argv)
    assert argv == ["serve.py"]


def test_missing_config_file_is_an_error():
    with pytest.raises(SystemExit):
        config.take_config_flag(["x", "--config", "no/such/file.yaml"])


def test_default_without_env(monkeypatch):
    monkeypatch.delenv(config.ENV, raising=False)
    assert config.path() == config.DEFAULT
