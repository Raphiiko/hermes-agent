"""Request-local tool and reasoning policies ported from the private route overlay."""

from copy import deepcopy

import pytest
import toolsets

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _adapter(policy):
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"model_routes": {"alias": {"model": "route/model", **policy}}}))


@pytest.fixture
def runtime(monkeypatch):
    captured = {}
    config = {
        "platform_toolsets": {"api_server": ["terminal"]},
        "context": {"engine": "lcm"},
        "mcp_servers": {"alpha": {"command": "unused"}, "beta": {"command": "unused"}},
    }

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "openrouter", "api_key": "global-key"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model: {"enabled": True, "effort": "high"}))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", lambda provider, target_model=None: {"provider": provider, "api_key": f"key-{provider}"})
    monkeypatch.setattr(APIServerAdapter, "_ensure_session_db", lambda self: None)
    monkeypatch.setattr(APIServerAdapter, "_session_model_override_for", lambda *args: None)
    monkeypatch.setitem(toolsets.TOOLSETS, "test_plugin", {"tools": ["test_plugin_tool"], "includes": []})
    monkeypatch.setattr("hermes_cli.tools_config._gateway_plugin_toolsets", lambda: ([("test_plugin", "Test plugin", "")], set()))
    return captured, config


def test_route_toolsets_and_reasoning_are_normalized():
    route = _adapter({"toolsets": [" web ", "no_mcp"], "reasoning_effort": " HIGH "})._resolve_route("alias")
    assert route["toolsets"] == ["web", "no_mcp"]
    assert route["reasoning_effort"] == "high"


@pytest.mark.parametrize("value", [[], None, "web", "['web']", {}, False, 1, [""], ["web", 3], ["web", None]])
def test_invalid_or_empty_toolsets_keep_route_but_fail_closed(runtime, value):
    captured, _ = runtime
    adapter = _adapter({"toolsets": value})
    route = adapter._resolve_route("alias")
    assert route is not None
    adapter._create_agent(session_id="s1", route=route)
    assert captured["model"] == "route/model"
    assert captured["enabled_toolsets"] == []


def test_invalid_route_reasoning_is_dropped():
    assert _adapter({"reasoning_effort": "warp-nine"})._model_routes == {}


def test_route_overrides_toolsets_and_reasoning(runtime):
    captured, config = runtime
    before = deepcopy(config)
    adapter = _adapter({"provider": "openai-codex", "toolsets": ["web", "no_mcp"], "reasoning_effort": "low"})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["model"] == "route/model"
    assert captured["provider"] == "openai-codex"
    assert captured["reasoning_config"] == {"enabled": True, "effort": "low"}
    assert captured["enabled_toolsets"] == ["web"]
    assert config == before


def test_no_mcp_alone_resolves_to_no_tools(runtime):
    captured, _ = runtime
    adapter = _adapter({"toolsets": ["no_mcp"]})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["enabled_toolsets"] == []


@pytest.mark.parametrize("name", ["missing", "all", "*", "hermes", "hermes-api-server", "hermes-cli", "hermes-gateway", "safe", "debugging", "coding"])
def test_unknown_or_composite_toolset_fails_entire_list_closed(runtime, name):
    captured, _ = runtime
    adapter = _adapter({"toolsets": ["web", name]})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["enabled_toolsets"] == []


@pytest.mark.parametrize("name", ["all", "*", "safe", "coding", "hermes-api-server"])
def test_mcp_name_cannot_turn_a_composite_into_an_allow_list_entry(runtime, name):
    captured, config = runtime
    config["mcp_servers"][name] = {"command": "unused"}
    adapter = _adapter({"toolsets": ["web", name]})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["enabled_toolsets"] == []


def test_blank_session_model_does_not_suppress_route_policy(runtime):
    captured, _ = runtime
    adapter = _adapter({"toolsets": [], "reasoning_effort": "low"})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"), session_model="  ")
    assert captured["model"] == "route/model"
    assert captured["enabled_toolsets"] == []
    assert captured["reasoning_config"] == {"enabled": True, "effort": "low"}


@pytest.mark.parametrize("names, expected", [
    (["web"], ["web"]),
    (["web", "web"], ["web"]),
    (["context_engine"], ["context_engine"]),
    (["test_plugin"], ["test_plugin"]),
    (["alpha"], ["alpha"]),
    (["mcp-alpha"], ["alpha"]),
    (["web", "alpha", "context_engine", "test_plugin"], ["alpha", "context_engine", "test_plugin", "web"]),
    (["web", "alpha", "no_mcp"], ["web"]),
])
def test_only_explicit_context_plugin_and_mcp_toolsets_survive(runtime, names, expected):
    captured, _ = runtime
    adapter = _adapter({"toolsets": names})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["enabled_toolsets"] == expected


def test_omitted_toolsets_inherit_normal_platform_selection(runtime):
    from hermes_cli.tools_config import _get_platform_tools

    captured, config = runtime
    expected = sorted(_get_platform_tools(config, "api_server"))
    assert {"terminal", "context_engine", "test_plugin", "alpha", "beta"} <= set(expected)
    adapter = _adapter({"reasoning_effort": "low"})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["enabled_toolsets"] == expected


def test_global_disabled_toolsets_still_apply(runtime):
    captured, config = runtime
    config["agent"] = {"disabled_toolsets": ["web"]}
    adapter = _adapter({"toolsets": ["web", "terminal"]})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["enabled_toolsets"] == ["terminal"]


@pytest.mark.parametrize("effort, expected", [(False, {"enabled": False}), ("none", {"enabled": False}), ("low", {"enabled": True, "effort": "low"}), (None, {"enabled": True, "effort": "high"})])
def test_route_reasoning_is_request_local(runtime, effort, expected):
    captured, _ = runtime
    adapter = _adapter({"toolsets": [], "reasoning_effort": effort})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))
    assert captured["reasoning_config"] == expected
    adapter._create_agent(session_id="s2")
    assert captured["model"] == "global/model"
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert "terminal" in captured["enabled_toolsets"]


def test_explicit_request_reasoning_beats_route(runtime):
    captured, _ = runtime
    adapter = _adapter({"reasoning_effort": "low"})
    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"), model_options={"reasoning_effort": "none"})
    assert captured["reasoning_config"] == {"enabled": False}


@pytest.mark.parametrize("selection", ["slash", "persisted", "confirmed"])
def test_session_selection_skips_complete_static_route_policy(runtime, monkeypatch, selection):
    from hermes_cli.tools_config import _get_platform_tools

    captured, config = runtime
    adapter = _adapter({"provider": "route-provider", "api_key": "route-key", "base_url": "https://route.invalid", "toolsets": [], "reasoning_effort": "low"})
    kwargs = {}
    if selection == "slash":
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *args: {"model": "session/model", "provider": "session-provider", "api_key": "session-key"})
    elif selection == "persisted":
        kwargs["session_model"] = "session/model"
    else:
        kwargs.update(confirmed_runtime_lock=True, requested_model="session/model", requested_provider="session-provider")
        # Confirmed browser locks carry their own selection, never a static alias.
        kwargs["route"] = None
    kwargs.setdefault("route", adapter._resolve_route("alias"))
    adapter._create_agent(session_id="s1", **kwargs)
    assert captured["model"] == "session/model"
    assert captured["provider"] != "route-provider"
    assert captured["api_key"] != "route-key"
    assert captured.get("base_url") != "https://route.invalid"
    assert captured["enabled_toolsets"] == sorted(_get_platform_tools(config, "api_server"))
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}
