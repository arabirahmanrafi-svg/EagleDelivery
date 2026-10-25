from pathlib import Path
import random
import tempfile
import unittest

import agents
from campus import ACTIONS, CampusMap


class SearchTests(unittest.TestCase):
    def test_manhattan(self):
        self.assertEqual(agents.manhattan((1, 2), (4, 6)), 7)
        self.assertEqual(agents.manhattan((4, 6), (1, 2)), 7)
        self.assertEqual(agents.manhattan((1, 2), (1, 2)), 0)

    def test_astar_small_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "small.txt"
            for text, length in [("#####\n#S.W#\n#####\n", 2),
                                 ("#####\n#S#W#\n#####\n", None)]:
                filename.write_text(text)
                campus = CampusMap(filename)
                start, goal = campus.location("S"), campus.location("W")
                answer = agents.astar(campus, start, goal)
                if length is None:
                    self.assertIsNone(answer.path)
                else:
                    self.assertEqual(answer.path, [start, (1, 2), goal])
                    self.assertEqual(answer.expanded, length)
                trivial = agents.astar(campus, start, start)
                self.assertEqual(trivial.path, [start])
                self.assertEqual(trivial.expanded, 0)

    def test_astar_supplied_maps(self):
        maps = Path(__file__).resolve().parents[1] / "maps"
        for filename in ("emu_campus.txt", "emu_detour.txt"):
            campus = CampusMap(maps / filename)
            for first, last in (("S", "W"), ("L", "D"), ("H", "P")):
                start, goal = campus.location(first), campus.location(last)
                baseline = agents.bfs(campus, start, goal)
                answer = agents.astar(campus, start, goal)
                self.assertIsNotNone(answer.path)
                self.assertEqual(len(answer.path), len(baseline.path))
                self.assertEqual((answer.path[0], answer.path[-1]), (start, goal))
                for state, successor in zip(answer.path, answer.path[1:]):
                    self.assertIn(successor, [n for a, n in campus.neighbors(state)])


class LearningTests(unittest.TestCase):
    def test_greedy_and_exploration(self):
        state = (1, 1)
        q = {(state, "RIGHT"): 5.0}
        rng = random.Random(10)
        for i in range(50):
            self.assertEqual(agents.epsilon_greedy(q, state, ACTIONS, 0, rng), "RIGHT")
        visited = {agents.epsilon_greedy(q, state, ACTIONS, 1, rng) for i in range(200)}
        self.assertEqual(visited, set(ACTIONS))
        tied = {agents.epsilon_greedy({}, state, ACTIONS, 0, rng) for i in range(200)}
        self.assertEqual(tied, set(ACTIONS))

    def test_unseen_zero_can_beat_negative_value(self):
        state = (1, 1)
        q = {(state, "RIGHT"): -1.0}
        action = agents.epsilon_greedy(q, state, ACTIONS, 0, random.Random(0))
        self.assertNotEqual(action, "RIGHT")

    def test_nonterminal_update(self):
        state, successor = (1, 1), (1, 2)
        q = {(state, "RIGHT"): 2.0, (successor, "UP"): 5.0}
        agents.q_learning_update(q, state, "RIGHT", -1, successor,
                                 ACTIONS, 0.5, 0.9, False)
        self.assertAlmostEqual(q[(state, "RIGHT")], 2.75)
        self.assertEqual(q[(successor, "UP")], 5.0)
        self.assertEqual(len(q), 2)

    def test_terminal_update_ignores_successor_values(self):
        state, successor = (1, 1), (1, 2)
        q = {(state, "RIGHT"): 2.0, (successor, "UP"): 999.0}
        agents.q_learning_update(q, state, "RIGHT", 39, successor,
                                 ACTIONS, 0.5, 0.9, True)
        self.assertAlmostEqual(q[(state, "RIGHT")], 20.5)

    def test_update_includes_unseen_actions(self):
        state, successor = (1, 1), (1, 2)
        q = {(successor, "UP"): -5.0}
        agents.q_learning_update(q, state, "RIGHT", -1, successor,
                                 ACTIONS, 1.0, 0.9, False)
        self.assertEqual(q[(state, "RIGHT")], -1.0)


if __name__ == "__main__":
    unittest.main()
