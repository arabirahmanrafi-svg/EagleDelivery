from collections import deque
import heapq
from itertools import count


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


def reconstruct_path(parents, goal):
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def bfs(campus, start, goal):
    frontier = deque([start])
    parents = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            if successor not in parents:
                parents[successor] = state
                frontier.append(successor)
    return SearchResult(None, expanded)


def manhattan(state, goal):
    """TODO 1: Return Manhattan distance between two (row, column) tuples."""
    # Add the absolute row difference and the absolute column difference.
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def astar(campus, start, goal):
    """TODO 2: Return SearchResult(path, expanded) for A* graph search."""
    # Use heapq as a priority queue and manhattan() as the heuristic.
    # Track best known g-costs and parent states in dictionaries.
    if start == goal:
        return SearchResult([start], 0)
    tie = count()
    frontier = [(manhattan(start, goal), next(tie), 0, start)]
    best_g = {start: 0}
    parents = {start: None}
    expanded = 0
    while frontier:
        f, order, g, state = heapq.heappop(frontier)
        if g != best_g[state]:
            continue  # stale entry, a cheaper route was found later
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            new_g = g + 1
            if successor not in best_g or new_g < best_g[successor]:
                best_g[successor] = new_g
                parents[successor] = state
                heapq.heappush(frontier, (new_g + manhattan(successor, goal),
                                          next(tie), new_g, successor))
    return SearchResult(None, expanded)


def epsilon_greedy(q_values, state, actions, epsilon, rng):
    """TODO 3: Choose an action with epsilon-greedy exploration."""
    # q_values is a dictionary. Its keys are (state, action) pairs.
    # Example key: ((1, 1), "RIGHT"). Unseen entries have value 0.0.
    if rng.random() < epsilon:
        return rng.choice(list(actions))
    values = [q_values.get((state, action), 0.0) for action in actions]
    best = max(values)
    return rng.choice([a for a, v in zip(actions, values) if v == best])


def q_learning_update(q_values, state, action, reward, next_state,
                      actions, alpha, gamma, terminal):
    """TODO 4: Update exactly one Q-table entry in place; return nothing."""
    # Get Q(state, action), defaulting to 0.0.
    # If terminal is True, the target is reward with no continuation term.
    # Otherwise target = reward + gamma * max Q(next_state, next_action).
    # Consider every action, including unseen entries with value 0.0.
    # Apply Q <- Q + alpha * (target - Q).
    current = q_values.get((state, action), 0.0)
    if terminal:
        target = reward
    else:
        target = reward + gamma * max(q_values.get((next_state, a), 0.0)
                                      for a in actions)
    q_values[(state, action)] = current + alpha * (target - current)