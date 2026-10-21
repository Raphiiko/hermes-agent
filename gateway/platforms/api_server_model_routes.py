"""Static model-route parsing and request-local tool selection."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_route_policy(route: dict[str, Any], cfg: dict[str, Any], alias: str) -> bool:
    from hermes_constants import parse_reasoning_effort

    if "toolsets" in cfg:
        names = cfg["toolsets"]
        if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() for name in names):
            logger.warning("api_server model_routes: route %r has invalid toolsets; using no tools", alias)
            names = []
        route["toolsets"] = [name.strip() for name in names]
    effort = cfg.get("reasoning_effort")
    if effort is not None:
        if parse_reasoning_effort(effort) is None:
            logger.warning("api_server model_routes: route %r has invalid reasoning_effort; dropping", alias)
            return False
        route["reasoning_effort"] = effort if isinstance(effort, (bool, dict)) else str(effort).strip().lower()
    return True


def parse_model_routes(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize aliases and route policies. Upstream credentials must never reach logs."""
    if not isinstance(raw, dict):
        if raw:
            logger.warning("api_server model_routes ignored: expected a mapping, got %s", type(raw).__name__)
        return {}
    routes: dict[str, dict[str, Any]] = {}
    for alias, cfg in raw.items():
        alias_str = str(alias).strip()
        if not alias_str or not isinstance(cfg, dict):
            logger.warning("api_server model_routes: dropping invalid route entry %r", alias_str or alias)
            continue
        route = {
            key: str(cfg[key]).strip()
            for key in ("model", "provider", "api_key", "base_url")
            if cfg.get(key) is not None and str(cfg[key]).strip()}
        if not route.get("model"):
            logger.warning("api_server model_routes: route %r has no 'model'; dropping", alias_str)
            continue
        if _normalize_route_policy(route, cfg, alias_str):
            routes[alias_str] = route
    return routes


def route_toolsets(config: dict[str, Any], names: Any) -> list[str]:
    """Resolve a strict allow-list without inheriting auto-enabled toolsets or composites."""
    from hermes_cli.tools_config import _get_platform_tools, enabled_mcp_server_names
    from toolsets import get_toolset

    if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() for name in names):
        return []
    mcp_servers = enabled_mcp_server_names(config)
    selected: set[str] = set()
    for raw_name in names:
        name = raw_name.strip()
        if name == "no_mcp":
            selected.add(name)
            continue
        if name.startswith("mcp-") and name not in mcp_servers and name[4:] in mcp_servers:
            name = name[4:]
        toolset = get_toolset(name)
        if (name in {"all", "*", "hermes"} or name.startswith("hermes-")
                or (toolset and (toolset.get("includes") or toolset.get("posture")))
                or (name not in mcp_servers and not toolset)):
            logger.warning("api_server model route has unknown or composite toolset %r; using no tools", name)
            return []
        selected.add(name)
    local_config = {**config, "platform_toolsets": {**(config.get("platform_toolsets") or {}), "api_server": sorted(selected)}}
    enabled = _get_platform_tools(local_config, "api_server", include_default_mcp_servers=False)
    return sorted(enabled & (selected - {"no_mcp"}))
