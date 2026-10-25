from __future__ import annotations

from pathlib import Path
import unittest


class DockerGatewayConfigurationTests(unittest.TestCase):
    def test_ai_service_has_only_the_gateway_network_and_gateway_url(self):
        compose = (Path(__file__).parents[1] / "docker-compose.yml").read_text()
        self.assertIn("MCP_GATEWAY_URL: http://gateway:8080/mcp", compose)
        self.assertIn("- ai-gateway-network", compose)
        self.assertNotIn("mcp-gateway-network", compose)


if __name__ == "__main__":
    unittest.main()
