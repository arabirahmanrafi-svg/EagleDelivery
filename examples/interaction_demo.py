from pathlib import Path
import sys

# allow this example to run from the starter project directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campus import CampusMap, DeliveryEnvironment

campus = CampusMap(Path(__file__).resolve().parents[1] / "maps" / "emu_campus.txt")
environment = DeliveryEnvironment(campus, campus.location("S"), campus.location("W"))
state = environment.reset()
print("Initial state:", state)
print(campus.render(position=state))

for action in ["UP", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]:
    next_state, reward, terminal = environment.step(action)
    print("state =", state, "action =", action, "next =", next_state,
          "reward =", reward, "terminal =", terminal)
    state = next_state
    if terminal:
        break
