from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


class DockerGatewayConfigurationTests(unittest.TestCase):
    def test_ai_service_has_only_the_gateway_network_and_gateway_url(self):
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("MCP_GATEWAY_URL: http://gateway:8080/mcp", compose)
        self.assertIn("- ai-gateway-network", compose)
        self.assertNotIn("mcp-gateway-network", compose)

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
