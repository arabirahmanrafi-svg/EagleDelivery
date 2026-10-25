# EagleDelivery

COSC 461/561 Project 1. A simulated delivery robot carries one package between
building entrances on a grid map of the Eastern Michigan University campus. The
project has three parts:

- **Part A, search:** A* with a Manhattan-distance heuristic, compared with the
  provided BFS baseline on the campus map and on a detour map with one walkway closed.
- **Part B, learning:** tabular Q-learning with epsilon-greedy exploration. The agent
  learns only from interaction, with no access to the map or to BFS/A*.
- **Part C, custom map and experiment:** my own map, `maps/my_campus.txt`, and a
  controlled experiment on how the exploration rate epsilon affects learning on its
  S-W delivery task.

The report is [`report/EagleDelivery_Report.pdf`](report/EagleDelivery_Report.pdf).

## Requirements

Python 3.9 or newer. The simulator, tests, and experiments use only the standard
library. The plots need `matplotlib`, and rebuilding the report PDF needs `reportlab`:

```bash
python3 -m pip install matplotlib reportlab
```

## Files

| Path | Contents |
| --- | --- |
| `agents.py` | BFS (provided), plus my `manhattan`, `astar`, `epsilon_greedy`, and `q_learning_update` |
| `campus.py` | Map loading, movement, rewards, and resets (provided) |
| `run_project.py` | Show a map, run search, train and evaluate an agent (provided) |
| `summarize_results.py` | Evaluation summary and `learning_curve.csv` for a results folder (provided) |
| `make_plots.py` | Training-curve plots and training statistics for the report |
| `maps/` | `emu_campus.txt`, `emu_detour.txt`, and my `my_campus.txt` |
| `results/` | Raw training and evaluation CSVs, summaries, learning curves, and search results |
| `figures/` | Plots used in the report |
| `report/` | The report PDF and the script that builds it |

## Run commands

Run everything from the project folder.

Tests:

```bash
python3 -m unittest discover -v
```

Part A, search. A single example:

```bash
python3 run_project.py search --algorithm astar --start S --goal W
```

All BFS and A* runs, saved to `results/search_results.txt`:

```bash
for m in emu_campus emu_detour my_campus; do for t in S:W L:D H:P; do for a in bfs astar; do echo "$m ${t/:/-} $a: $(python3 run_project.py search --algorithm $a --map maps/$m.txt --start ${t%:*} --goal ${t#*:} | grep -E 'Expanded|Path length' | tr '\n' ' ')"; done; done; done > results/search_results.txt
```

Part B, baseline Q-learning on the campus map:

```bash
python3 run_project.py train --start S --goal W --output results/baseline_SW
python3 run_project.py train --start L --goal D --output results/baseline_LD
python3 run_project.py train --start H --goal P --output results/baseline_HP
python3 summarize_results.py results/baseline_SW
python3 summarize_results.py results/baseline_LD
python3 summarize_results.py results/baseline_HP
```

Part C, epsilon experiment on my map:

```bash
python3 run_project.py search --map maps/my_campus.txt --algorithm bfs
python3 run_project.py search --map maps/my_campus.txt --algorithm astar
python3 run_project.py train --map maps/my_campus.txt --epsilon 0.05 --output results/custom_e005
python3 run_project.py train --map maps/my_campus.txt --epsilon 0.2 --output results/custom_e020
python3 run_project.py train --map maps/my_campus.txt --epsilon 0.5 --output results/custom_e050
python3 summarize_results.py results/custom_e005
python3 summarize_results.py results/custom_e020
python3 summarize_results.py results/custom_e050
```

Plots, statistics, and report:

```bash
python3 make_plots.py
python3 report/build_report.py
```

`run_project.py` will not write into a folder that already has CSV files, so use a
new `--output` folder for each rerun.

## Experiment settings

| Setting | Part B baseline | Part C experiment |
| --- | --- | --- |
| Map | `maps/emu_campus.txt` | `maps/my_campus.txt` |
| Delivery tasks | S-W, L-D, H-P | S-W |
| Training episodes per seed | 2,000 | 2,000 |
| Evaluation episodes per seed | 100 (epsilon 0, frozen Q-table) | 100 (epsilon 0, frozen Q-table) |
| Max actions per episode | 200 | 200 |
| Seeds | 0, 1, 2 (fresh Q-table each) | 0, 1, 2 for every setting |
| Learning rate alpha | 0.3 | 0.3 |
| Discount gamma | 0.95 | 0.95 |
| Exploration epsilon | 0.2, constant | **0.05, 0.2, 0.5** (the varied factor) |
| Slip probability | 0.0 | 0.0 |

## Results summary

- **Search:** BFS and A* find routes of the same optimal length in all six required
  cases. A* expands fewer nodes in each one, from 22 vs 67 (campus S-W) to 60 vs 65
  (detour L-D).
- **Q-learning baseline:** all nine runs reach 100% evaluation success on the
  shortest route (18, 22, and 16 actions).
- **Epsilon experiment:** every epsilon setting learns the same 31-action route with
  100% success, and all three plateau at about episode 200. Higher epsilon mainly
  makes training more expensive: about 85k, 97k, and 147k total training actions for
  epsilon 0.05, 0.2, and 0.5.

## Custom map

```
###################
#S......#.........#
#.#####.#.#########
#.#...#...#.....#.#
#.#.#.#####.###.#.#
#...#.....L.#H#...#
###.#######.#.#####
#.....#.....#.....#
#.###.#.#####.###.#
#.#D#...#...P...#.#
#.#.#####.#####.#W#
#.................#
###################
```

The map is 13 × 19 with 3 independent loops. The 20-cell dead-end branch next to S
runs toward W's column and stops at (1, 17). The shortest S-W route is 31 actions;
the detours through P or through the corridor above W take 35 and 37 actions.
