"""Settings endpoints, password hashing and config persistence."""

from __future__ import annotations

import base64
import httpx
import pytest
import yaml

from locallm_valet.api import create_app
from locallm_valet.config import load_config
from locallm_valet.crypto import generate_api_key, hash_password, is_hashed, verify_password

from .conftest import make_config


def _basic(user: str, pw: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ------------------------------------------------------------------- hashing


def test_hash_and_verify():
    stored = hash_password("s3cret")
    assert is_hashed(stored)
    assert stored.startswith("pbkdf2:sha256:260000$")
    assert verify_password("s3cret", stored)
    assert not verify_password("wrong", stored)
    # two hashes of the same password differ (random salt) but both verify
    assert verify_password("s3cret", hash_password("s3cret"))


def test_verify_plaintext_fallback():
    """Legacy configs store plaintext; it must keep authenticating."""
    assert verify_password("admin", "admin")
    assert not verify_password("admin", "admin2")
    assert not verify_password("", None)


def test_generate_api_key_format():
    key = generate_api_key()
    assert key.startswith("sk-")
    assert len(key) == len("sk-") + 32 * 2  # 32 hex bytes
    assert all(c in "0123456789abcdef" for c in key[3:])
    assert generate_api_key() != key


def test_crypto_cli_hash_and_verify(capsys):
    """``python -m locallm_valet.crypto hash`` prints a usable hash and
    ``verify`` round-trips it."""
    from locallm_valet.crypto import main

    assert main(["hash", "s3cret"]) == 0
    stored = capsys.readouterr().out.strip()
    assert is_hashed(stored)
    assert verify_password("s3cret", stored)

    assert main(["verify", "s3cret", stored]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["verify", "wrong", stored]) == 1
    assert "FAIL" in capsys.readouterr().out


# ------------------------------------------------------- first-launch defaults


_CONFIG_TEMPLATE = """
server:
  port: 8101
backend:
  port: 30123
usage:
  enabled: false
models:
  qwen:
    path: /tmp/qwen
"""


async def test_first_launch_defaults(tmp_path):
    """No credentials at all → admin/admin created as a PBKDF2 hash."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_CONFIG_TEMPLATE, encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.server.username == "" and cfg.server.password == ""
    app = create_app(config=cfg)
    assert cfg.server.username == "admin"
    assert is_hashed(cfg.server.password) and verify_password("admin", cfg.server.password)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        assert (await client.get("/gateway/settings/credentials")).status_code == 401
        r = await client.get("/gateway/settings/credentials", headers=_basic("admin", "admin"))
        assert r.status_code == 200
        assert r.json() == {"username": "admin", "has_password": True}

    saved = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert saved["server"]["password"].startswith("pbkdf2:sha256:")


async def test_plaintext_password_upgraded_on_startup(tmp_path):
    """A legacy plaintext password is hashed + persisted on first boot."""
    cfg_file = tmp_path / "config.yaml"
    data = yaml.safe_load(_CONFIG_TEMPLATE)
    data["server"]["username"] = "kevin"
    data["server"]["password"] = "plain-secret"
    cfg_file.write_text(yaml.safe_dump(data), encoding="utf-8")

    cfg = load_config(cfg_file)
    create_app(config=cfg)

    assert is_hashed(cfg.server.password)
    assert verify_password("plain-secret", cfg.server.password)
    saved = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert saved["server"]["username"] == "kevin"
    assert saved["server"]["password"].startswith("pbkdf2:")
    assert saved["server"]["port"] == 8101  # untouched sections survive


# ------------------------------------------------------------ settings API


def _settings_app(auth: bool = True):
    cfg = make_config()
    if auth:
        cfg.server.username = "admin"
        cfg.server.password = hash_password("admin-pw")
    return cfg, create_app(config=cfg)


async def test_basic_auth_login():
    _, app = _settings_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        r = await client.post("/gateway/settings/auth-check", headers=_basic("admin", "admin-pw"))
        assert r.status_code == 200
        assert r.json() == {"ok": True}


async def test_basic_auth_wrong_password():
    _, app = _settings_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        for headers in (
            _basic("admin", "nope"),
            _basic("someone-else", "admin-pw"),
            {},
        ):
            r = await client.post("/gateway/settings/auth-check", headers=headers)
            assert r.status_code == 401


async def test_settings_page_shell_open_but_data_gated():
    """The page shell renders without auth (login modal lives there); the
    data endpoints behind it demand Basic/Bearer credentials."""
    _, app = _settings_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        shell = await client.get("/gateway/settings")
        assert shell.status_code == 200
        assert "authedFetch" in shell.text
        # Settings is the one page whose data endpoints are all gated — it
        # validates stored credentials on load (pops the login modal on 401).
        assert "autoCheckCredentials();" in shell.text
        for path in ("/gateway/settings/api-keys", "/gateway/settings/models"):
            assert (await client.get(path)).status_code == 401
        assert (await client.get("/gateway/status")).status_code == 200  # open GET stays open


async def test_api_key_crud():
    cfg, app = _settings_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        h = _basic("admin", "admin-pw")
        r = await client.post("/gateway/settings/api-keys", headers=h)
        assert r.status_code == 200
        key = r.json()["key"]
        assert r.json()["masked"] == key[:8] + "***"

        listed = (await client.get("/gateway/settings/api-keys", headers=h)).json()
        assert [k["masked"] for k in listed["keys"]] == [key[:8] + "***"]

        # new Bearer key grants /v1 access immediately
        assert (await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})).status_code == 200

        # delete by prefix -> gone from the registry (no config file behind
        # this synthetic config, so only the in-memory side applies)
        r = await client.delete(f"/gateway/settings/api-keys/{key[:8]}", headers=h)
        assert r.status_code == 200
        listed = (await client.get("/gateway/settings/api-keys", headers=h)).json()
        assert listed["keys"] == []
        assert (await client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})).status_code == 401
        r = await client.delete(f"/gateway/settings/api-keys/{key[:8]}", headers=h)
        assert r.status_code == 404


async def test_change_credentials(tmp_path):
    cfg = make_config()
    cfg._config_path = str(tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text(_CONFIG_TEMPLATE.replace("8101", "8100"), encoding="utf-8")
    cfg.server.username = "admin"
    cfg.server.password = hash_password("old-pw")
    app = create_app(config=cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        r = await client.post(
            "/gateway/settings/credentials",
            json={"current_password": "wrong"},
            headers=_basic("admin", "old-pw"),
        )
        assert r.status_code == 400

        r = await client.post(
            "/gateway/settings/credentials",
            json={"current_password": "old-pw", "new_password": "new-secret"},
            headers=_basic("admin", "old-pw"),
        )
        assert r.status_code == 200

    # old password dead, new one works (hashed, persisted)
    assert verify_password("new-secret", cfg.server.password)
    assert not verify_password("old-pw", cfg.server.password)
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["server"]["password"].startswith("pbkdf2:")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        assert (await client.post("/gateway/settings/auth-check", headers=_basic("admin", "old-pw"))).status_code == 401
        assert (await client.post("/gateway/settings/auth-check", headers=_basic("admin", "new-secret"))).status_code == 200


async def test_change_username_only(tmp_path):
    """Changing ONLY the username keeps the existing password — the settings
    userForm sends no ``new_password`` and must not be rejected."""
    cfg = make_config()
    cfg._config_path = str(tmp_path / "config.yaml")
    original_hash = hash_password("same-pw")
    data = yaml.safe_load(_CONFIG_TEMPLATE.replace("8101", "8100"))
    data["server"]["username"] = "admin"
    data["server"]["password"] = original_hash
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg.server.username = "admin"
    cfg.server.password = original_hash
    app = create_app(config=cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        r = await client.post(
            "/gateway/settings/credentials",
            json={"current_password": "same-pw", "new_username": "kevin"},
            headers=_basic("admin", "same-pw"),
        )
        assert r.status_code == 200
        assert r.json() == {"username": "kevin"}

    # Username changed; the (unchanged) password still authenticates.
    assert cfg.server.username == "kevin"
    # Password must NOT be re-hashed by a username-only change.
    assert cfg.server.password == original_hash
    assert verify_password("same-pw", cfg.server.password)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        assert (await client.post("/gateway/settings/auth-check", headers=_basic("kevin", "same-pw"))).status_code == 200
        assert (await client.post("/gateway/settings/auth-check", headers=_basic("admin", "same-pw"))).status_code == 401

    # Persisted: username rewritten, the pre-existing password hash untouched.
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["server"]["username"] == "kevin"
    assert saved["server"]["password"] == original_hash
    assert saved["server"]["port"] == 8100  # untouched sections survive


async def test_model_backend_edit(tmp_path):
    cfg = make_config()
    cfg._config_path = str(tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text(_CONFIG_TEMPLATE.replace("8101", "8100"), encoding="utf-8")
    cfg.server.username = "admin"
    cfg.server.password = hash_password("pw")
    app = create_app(config=cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        h = _basic("admin", "pw")
        listing = (await client.get("/gateway/settings/models", headers=h)).json()
        assert {m["name"] for m in listing["models"]} == {"qwen", "gemma"}

        r = await client.put(
            "/gateway/settings/models/qwen",
            headers=h,
            json={"command_template": "llama-server -m {model_path}", "extra_args": ["--ctx-size", "8192"], "health_path": "/healthz"},
        )
        assert r.status_code == 200
        assert r.json()["health_path"] == "/healthz"

    spec = cfg.models["qwen"]
    assert spec.backend.command_template == "llama-server -m {model_path}"
    assert spec.backend.extra_args == ["--ctx-size", "8192"]
    assert spec.backend.health_path == "/healthz"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    backend = saved["models"]["qwen"]["backend"]
    assert backend["command_template"] == "llama-server -m {model_path}"
    assert backend["extra_args"] == ["--ctx-size", "8192"]
    assert backend["health_path"] == "/healthz"

    # unknown model -> 404; invalid body -> 400
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        assert (await client.put("/gateway/settings/models/nope", headers=h, json={})).status_code == 404
        assert (await client.put("/gateway/settings/models/qwen", headers=h, json={"extra_args": "--bad"})).status_code == 400


# ------------------------------------------------------- backwards compatibility


async def test_bearer_auth_still_works():
    """The old Bearer flow over /v1/* is unchanged."""
    cfg = make_config()
    cfg.server.api_keys = ["sk-legacy"]
    app = create_app(config=cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        assert (await client.get("/v1/models")).status_code == 401
        assert (await client.get("/v1/models", headers={"Authorization": "Bearer sk-legacy"})).status_code == 200
