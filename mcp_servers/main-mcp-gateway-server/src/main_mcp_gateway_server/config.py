"""YAML 등록 파일에서 Gateway upstream MCP 서비스 정보를 읽어 검증한다."""

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


@dataclass(frozen=True)
class UpstreamConfig:
    """하나의 upstream MCP 서비스 연결 정보와 식별자를 보관한다."""

    service_id: str
    mcp_url: str
    health_url: str


def configured_upstreams() -> tuple[UpstreamConfig, ...]:
    """외부 YAML 파일의 upstream 등록 정보를 검증해 반환한다.

    `UPSTREAM_CONFIG_PATH`가 없으면 컨테이너 기본 경로를 사용한다. 파일에는 `upstreams` 배열과
    service_id, mcp_url, health_url 문자열이 필요하며, service_id 중복이나 잘못된 형식은 ValueError를 발생시킨다.
    """
    path = Path(os.getenv("UPSTREAM_CONFIG_PATH", "/app/config/upstreams.yaml"))
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"upstream 등록 파일을 읽을 수 없습니다: {path}") from error
    if not isinstance(parsed, dict) or not isinstance(parsed.get("upstreams"), list):
        raise ValueError("upstream 등록 파일에는 upstreams 배열이 필요합니다")

    upstreams: list[UpstreamConfig] = []
    service_ids: set[str] = set()
    for item in parsed["upstreams"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key] for key in ("service_id", "mcp_url", "health_url")):
            raise ValueError("각 upstream에는 비어 있지 않은 service_id, mcp_url, health_url이 필요합니다")
        service_id = item["service_id"]
        if service_id in service_ids:
            raise ValueError(f"중복된 service_id입니다: {service_id}")
        if not item["mcp_url"].startswith(("http://", "https://")) or not item["health_url"].startswith(("http://", "https://")):
            raise ValueError(f"upstream URL은 http 또는 https URL이어야 합니다: {service_id}")
        service_ids.add(service_id)
        upstreams.append(UpstreamConfig(service_id, item["mcp_url"], item["health_url"]))
    return tuple(upstreams)
