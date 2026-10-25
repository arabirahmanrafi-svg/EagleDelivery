from pathlib import Path
import tempfile
import unittest

from agents import bfs
from campus import CampusMap, DeliveryEnvironment


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def make_map(self, text):
        filename = Path(self.folder.name) / "test_map.txt"
        filename.write_text(text)
        return CampusMap(filename)

    def test_movement_reward_and_wall(self):
        campus = self.make_map("#####\n#S.W#\n#####\n")
        env = DeliveryEnvironment(campus, campus.location("S"), campus.location("W"))
        self.assertEqual(env.step("UP"), ((1, 1), -1, False))
        self.assertEqual(env.step("RIGHT"), ((1, 2), -1, False))
        self.assertEqual(env.step("RIGHT"), ((1, 3), 39, True))
        with self.assertRaises(ValueError):
            env.step("LEFT")
        self.assertEqual(env.reset(), (1, 1))

    def test_certain_slip(self):
        campus = self.make_map("#####\n#S.W#\n#####\n")
        env = DeliveryEnvironment(campus, campus.location("S"), campus.location("W"), slip=1)
        self.assertEqual(env.step("RIGHT"), ((1, 1), -1, False))

    def test_invalid_map(self):
        with self.assertRaises(ValueError):
            self.make_map("###\n#S.W#\n###\n")

    def test_duplicate_building(self):
        with self.assertRaises(ValueError):
            self.make_map("#####\n#SSW#\n#####\n")

    def test_bfs_and_unreachable(self):
        campus = self.make_map("#####\n#S.W#\n#####\n")
        answer = bfs(campus, campus.location("S"), campus.location("W"))
        self.assertEqual(answer.path, [(1, 1), (1, 2), (1, 3)])
        self.assertEqual(answer.expanded, 2)
        blocked = self.make_map("#####\n#S#W#\n#####\n")
        self.assertIsNone(bfs(blocked, blocked.location("S"), blocked.location("W")).path)

    def test_supplied_routes_exist(self):
        maps = Path(__file__).resolve().parents[1] / "maps"
        for filename in ("emu_campus.txt", "emu_detour.txt"):
            campus = CampusMap(maps / filename)
            for start, goal in (("S", "W"), ("L", "D"), ("H", "P")):
                answer = bfs(campus, campus.location(start), campus.location(goal))
                self.assertIsNotNone(answer.path, (filename, start, goal))


if __name__ == "__main__":
    unittest.main()
