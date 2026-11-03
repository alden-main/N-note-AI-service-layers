from __future__ import annotations

import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from n_note_ai_service_layers import settings

ROOT = Path(__file__).parents[1]


def _upstream_hosts() -> list[str]:
    """Gateway에 등록된 MCP 서버의 호스트명(= compose 서비스 이름)을 읽는다."""
    upstreams = (ROOT / "mcp_servers" / "main-mcp-gateway-server" / "config" / "upstreams.yaml").read_text()
    return re.findall(r"^\s+mcp_url: http://([^:/]+)", upstreams, re.MULTILINE)


def _anchor_networks(compose: str) -> dict[str, list[str]]:
    """최상위 ``x-*: &anchor`` 블록의 ``networks:`` 목록을 읽는다."""
    anchors: dict[str, list[str]] = {}
    for name, body in re.findall(r"^x-[\w-]+: &([\w-]+)\n((?:[ #].*\n|\n)+)", compose, re.MULTILINE):
        block = re.search(r"^  networks:\n((?:    - .*\n)+)", body, re.MULTILINE)
        if block:
            anchors[name] = [line.strip()[2:] for line in block.group(1).splitlines()]
    return anchors


def _service_networks(compose: str) -> dict[str, list[str]]:
    """``services:`` 아래 각 서비스의 ``networks:`` 목록을 들여쓰기로 읽는다(YAML 의존성 없이)."""
    anchors = _anchor_networks(compose)
    networks: dict[str, list[str]] = {}
    service = None
    in_services = in_networks = False
    for line in compose.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_services = line.startswith("services:")
            service = None
            continue
        if not in_services:
            continue
        if indent == 2 and line.rstrip().endswith(":"):
            service, in_networks = line.strip()[:-1], False
            networks[service] = []
        elif indent == 4:
            in_networks = line.strip() == "networks:"
            merged = re.fullmatch(r"<<: \*([\w-]+)", line.strip())
            if merged and service:
                networks[service].extend(anchors.get(merged.group(1), []))
        elif in_networks and service and line.strip().startswith("- "):
            networks[service].append(line.strip()[2:])
    return networks


def _network_definitions(compose: str) -> dict[str, str]:
    """최상위 ``networks:`` 아래 네트워크별 정의를 문자열로 읽는다."""
    body = compose.split("\nnetworks:\n")[-1]
    sections = re.split(r"^  ([a-z0-9-]+):\n", body, flags=re.MULTILINE)
    return dict(zip(sections[1::2], sections[2::2]))


class DockerGatewayConfigurationTests(unittest.TestCase):
    def test_ai_service_uses_the_gateway_and_edge_networks_and_gateway_url(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("MCP_GATEWAY_URL: http://gateway:8080/mcp", compose)
        self.assertEqual(sorted(_service_networks(compose)["ai-service"]), ["ai-gateway-network", "n-note-edge"])

    def test_only_ai_service_joins_the_edge_network(self):
        networks = _service_networks((ROOT / "docker-compose.yml").read_text())
        self.assertEqual([name for name, items in networks.items() if "n-note-edge" in items], ["ai-service"])

    def test_edge_is_the_only_network_with_outside_access(self):
        definitions = _network_definitions((ROOT / "docker-compose.yml").read_text())
        self.assertEqual(sorted(definitions), ["ai-gateway-network", "mcp-gateway-network", "n-note-edge"])
        self.assertNotIn("internal: true", definitions["n-note-edge"])
        for name in ("ai-gateway-network", "mcp-gateway-network"):
            with self.subTest(network=name):
                self.assertIn(f"name: {name}\n", definitions[name])
                self.assertIn("internal: true", definitions[name])

    def test_every_service_declares_its_networks(self):
        # networks:를 빼면 Compose가 internal이 아닌 <프로젝트>_default 네트워크에 넣는다.
        networks = _service_networks((ROOT / "docker-compose.yml").read_text())
        self.assertIn("korean", networks)
        for name, items in networks.items():
            with self.subTest(service=name):
                self.assertTrue(items, f"{name} has no networks")

    def test_local_model_servers_stay_on_the_internal_gateway_network(self):
        networks = _service_networks((ROOT / "docker-compose.yml").read_text())
        for name in ("local-stt", "local-llm"):
            with self.subTest(service=name):
                self.assertEqual(networks[name], ["ai-gateway-network"])

    def test_only_the_gateway_bridges_ai_and_mcp_networks(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        networks = _service_networks(compose)
        self.assertEqual(sorted(networks["gateway"]), ["ai-gateway-network", "mcp-gateway-network"])
        for name in _upstream_hosts():
            with self.subTest(service=name):
                self.assertEqual(networks.get(name), ["mcp-gateway-network"])
        bridging = [name for name, items in networks.items() if "mcp-gateway-network" in items and "ai-gateway-network" in items]
        self.assertEqual(bridging, ["gateway"])

    def test_mcp_services_match_gateway_upstream_hosts_and_publish_no_ports(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        hosts = _upstream_hosts()
        self.assertGreaterEqual(len(hosts), 3)
        for name in hosts:
            with self.subTest(service=name):
                self.assertIn(f"\n  {name}:\n", compose)
        self.assertNotIn("ports:", compose)

    def test_local_model_servers_are_opt_in_profiles(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn('profiles: ["local-stt"]', compose)
        self.assertIn('profiles: ["local-llm"]', compose)

    def test_every_model_profile_points_local_services_at_compose_hosts(self):
        profiles = sorted((ROOT / "deploy" / "models").glob("*.env"))
        self.assertGreaterEqual(len(profiles), 6)
        for path in profiles:
            values = dict(line.split("=", 1) for line in path.read_text().splitlines() if line and not line.startswith("#"))
            with self.subTest(profile=path.stem):
                self.assertEqual(values["N_NOTE_MODEL_PROFILE"], path.stem)
                if "local-llm" in values["N_NOTE_COMPOSE_PROFILES"]:
                    self.assertEqual(values["N_NOTE_LLM_BASE_URL"], "http://local-llm:8000/v1")
                    self.assertEqual(values["N_NOTE_LLM_API"], "chat")
                    self.assertIn("N_NOTE_LOCAL_LLM_MODEL", values)
                if "local-stt" in values["N_NOTE_COMPOSE_PROFILES"]:
                    self.assertEqual(values["N_NOTE_STT_BASE_URL"], "http://local-stt:8000/v1")

    def test_settings_accept_every_blank_value_that_compose_passes(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        blank = dict.fromkeys(re.findall(r"^\s+([A-Z][A-Z0-9_]*): \$\{.*:-\}+\s*$", compose, re.MULTILINE), "")
        self.assertIn("N_NOTE_LLM_EXTRA_BODY", blank)
        with patch.dict(os.environ, blank, clear=True):
            self.assertEqual(settings.get().llm_extra_body, {})

    def test_standalone_mcp_composes_stay_off_the_shared_networks(self):
        # 같은 서비스 이름이 공유 네트워크에 두 번 등록되면 Gateway 호출이 어느 쪽으로 갈지 정해지지 않는다.
        composes = sorted((ROOT / "mcp_servers").glob("*/compose.yaml"))
        self.assertEqual(len(composes), 3)
        for path in composes:
            with self.subTest(compose=path.parent.name):
                compose = path.read_text()
                self.assertNotIn("mcp-gateway-network", compose)
                self.assertNotIn("ai-gateway-network", compose)
                self.assertNotIn("external: true", compose)
                self.assertIn("default:\n    internal: true\n", compose)

    def test_standalone_report_compose_uses_the_gateway_upstream_host(self):
        compose = (ROOT / "mcp_servers" / "main-mcp-server-document-analysis" / "compose.yaml").read_text()
        self.assertIn("\n  report:\n", compose)

    def test_install_checks_every_compose_network(self):
        install = (ROOT / "scripts" / "install.sh").read_text()
        for name in _network_definitions((ROOT / "docker-compose.yml").read_text()):
            with self.subTest(network=name):
                self.assertRegex(install, rf"for network in .*\b{name}\b")

    def test_install_removes_containers_of_renamed_services(self):
        self.assertIn("--remove-orphans", (ROOT / "scripts" / "install.sh").read_text())


if __name__ == "__main__":
    unittest.main()
