import unittest


class TinyAgentTest(unittest.TestCase):
    def test_tiny_agent_returns_weather_observation(self):
        from architecture.tiny import TinyAgent
        agent = TinyAgent()

        self.assertEqual(agent.run("what the weather?"), "☀️")


if __name__ == "__main__":
    unittest.main()
