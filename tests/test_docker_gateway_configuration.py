from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


MCP_SERVICES = ("korean-standard-dictionary", "medical-term-review", "report-validation")


def _service_networks(compose: str) -> dict[str, list[str]]:
    """``services:`` 아래 각 서비스의 ``networks:`` 목록을 들여쓰기로 읽는다(YAML 의존성 없이)."""
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
        elif in_networks and service and line.strip().startswith("- "):
            networks[service].append(line.strip()[2:])
    return networks


class DockerGatewayConfigurationTests(unittest.TestCase):
    def test_ai_service_has_only_the_gateway_network_and_gateway_url(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("MCP_GATEWAY_URL: http://gateway:8080/mcp", compose)
        self.assertEqual(_service_networks(compose)["ai-service"], ["ai-gateway-network"])

    def test_only_the_gateway_bridges_ai_and_mcp_networks(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        networks = _service_networks(compose)
        self.assertEqual(sorted(networks["gateway"]), ["ai-gateway-network", "mcp-gateway-network"])
        for name in MCP_SERVICES:
            with self.subTest(service=name):
                self.assertEqual(networks[name], ["mcp-gateway-network"])
        bridging = [name for name, items in networks.items() if "mcp-gateway-network" in items and "ai-gateway-network" in items]
        self.assertEqual(bridging, ["gateway"])
        # MCP 네트워크는 외부 인터넷이 막힌 internal 네트워크다.
        self.assertIn("mcp-gateway-network:\n    name: mcp-gateway-network\n", compose)
        self.assertIn("internal: true", compose.split("\nnetworks:\n")[-1])

    def test_mcp_services_match_gateway_upstream_hosts_and_publish_no_ports(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        upstreams = (ROOT / "mcp_servers" / "main-mcp-gateway-server" / "config" / "upstreams.yaml").read_text()
        for name in MCP_SERVICES:
            with self.subTest(service=name):
                self.assertIn(f"http://{name}:8000/mcp", upstreams)
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


if __name__ == "__main__":
    unittest.main()
