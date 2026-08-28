import unittest

from company_project import SceneAgent, SceneAgentBoundaryError


class EmptyScene(SceneAgent):
    name = "empty"
    description = "empty scene"

    def handle(self, task):
        return self.complete(task, output="ok")


class SceneAgentBoundariesTest(unittest.TestCase):
    def test_scene_agent_rejects_mounting_another_scene_agent(self):
        scene = EmptyScene(skills=[], tools=[])
        another_scene = EmptyScene(skills=[], tools=[])

        with self.assertRaises(SceneAgentBoundaryError):
            scene.mount_scene(another_scene)


if __name__ == "__main__":
    unittest.main()
