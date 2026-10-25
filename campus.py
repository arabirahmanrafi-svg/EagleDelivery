import random


# four actions available; hitting a wall leaves the robot in place
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
BUILDING_NAMES = {
    "S": "Student Center",
    "L": "Halle Library",
    "H": "Pray-Harrold",
    "D": "Downing Hall",
    "P": "Putnam Hall",
    "W": "Walton Hall",
}


class CampusMap:
    def __init__(self, filename):
        with open(filename) as source:
            self.rows = source.read().splitlines()
        if not self.rows or not self.rows[0]:
            raise ValueError("Map must not be empty.")
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        self.locations = {}
        self.filename = str(filename)
        allowed = set("#." + "".join(BUILDING_NAMES))
        for row_number, row in enumerate(self.rows):
            if len(row) != self.width:
                raise ValueError("All map rows must have the same width.")
            for column_number, cell in enumerate(row):
                if cell not in allowed:
                    raise ValueError("Use only #, ., S, L, H, D, P, and W in maps.")
                if cell in BUILDING_NAMES:
                    if cell in self.locations:
                        raise ValueError("Each building letter may appear only once.")
                    self.locations[cell] = (row_number, column_number)

    def location(self, letter):
        if letter not in self.locations:
            raise ValueError("Building " + letter + " is not present on this map.")
        return self.locations[letter]

    def passable(self, state):
        row, column = state
        return (0 <= row < self.height and 0 <= column < self.width
                and self.rows[row][column] != "#")

    def next_position(self, state, action):
        if action not in MOVES:
            raise ValueError("Unknown action: " + str(action))
        dr, dc = MOVES[action]
        candidate = (state[0] + dr, state[1] + dc)
        return candidate if self.passable(candidate) else state

    def neighbors(self, state):
        # ordered UP, DOWN, LEFT, RIGHT
        result = []
        for action in ACTIONS:
            successor = self.next_position(state, action)
            if successor != state:
                result.append((action, successor))
        return result

    def render(self, position=None, path=None):
        cells = [list(row) for row in self.rows]
        for row, column in path or []:
            if cells[row][column] == ".":
                cells[row][column] = "o"
        if position is not None:
            cells[position[0]][position[1]] = "@"
        return "\n".join("".join(row) for row in cells)


class DeliveryEnvironment:
    def __init__(self, campus, start, goal, slip=0.0, seed=0):
        if not campus.passable(start) or not campus.passable(goal):
            raise ValueError("Start and goal must be traversable cells.")
        if start == goal:
            raise ValueError("Delivery episodes require different start and goal cells.")
        if not 0 <= slip <= 1:
            raise ValueError("Slip probability must be between 0 and 1.")
        self.campus = campus
        self.start = start
        self.goal = goal
        self.slip = slip
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        self.position = self.start
        self.terminal = False
        return self.position

    def step(self, action):
        if self.terminal:
            raise ValueError("Episode has ended, call reset() before acting again.")
        if action not in ACTIONS:
            raise ValueError("Unknown action: " + str(action))
        if self.slip == 0 or self.random.random() >= self.slip:
            self.position = self.campus.next_position(self.position, action)
        reward = -1
        self.terminal = self.position == self.goal
        if self.terminal:
            reward += 40
        return self.position, reward, self.terminal
