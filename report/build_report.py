"""Build report/EagleDelivery_Report.pdf from the saved results and figures.

Run from the project folder after make_plots.py (needs: python3 -m pip install reportlab):
    python3 report/build_report.py
"""
import csv
from pathlib import Path
import re
import statistics

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, Preformatted, SimpleDocTemplate,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "Arabi Rahman Rafi"
REPO = "https://github.com/arabirahmanrafi-svg/EagleDelivery"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman",
                      fontSize=9.8, leading=11.8, spaceAfter=3.5)
heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName="Times-Bold",
                         fontSize=11.5, leading=13, spaceBefore=5, spaceAfter=2.5)
title = ParagraphStyle("title", parent=styles["Title"], fontName="Times-Bold",
                       fontSize=14, leading=16, spaceAfter=1)
byline = ParagraphStyle("byline", parent=body, alignment=TA_CENTER, spaceAfter=4)
caption = ParagraphStyle("caption", parent=body, fontSize=8.6, leading=10,
                         alignment=TA_CENTER, spaceAfter=5)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=9.6, spaceAfter=0,
                      alignment=TA_CENTER)
mono = ParagraphStyle("mono", fontName="Courier", fontSize=7.8, leading=8.6)


def para(text):
    return Paragraph(text, body)


def table(rows, widths):
    data = [[Paragraph(str(value), cell) for value in row] for row in rows]
    result = Table(data, colWidths=widths, hAlign="CENTER")
    result.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return result


def figure(filename, width):
    image = Image(str(ROOT / filename))
    image.drawHeight = width * image.imageHeight / image.imageWidth
    image.drawWidth = width
    return image


def read_search():
    # Lines look like: "emu_campus S-W bfs: Expanded: 67 Path length: 18"
    pattern = re.compile(r"(\S+) (\S-\S) (\w+): Expanded: (\d+) Path length: (\d+)")
    found = {}
    for line in (ROOT / "results" / "search_results.txt").read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            name, task, algorithm, expanded, length = match.groups()
            found[(name, task, algorithm)] = (int(length), int(expanded))
    return found


def read_runs(folder):
    runs = []
    for seed in (0, 1, 2):
        with open(ROOT / "results" / folder / f"training_seed{seed}.csv") as source:
            runs.append([{k: int(v) for k, v in row.items()}
                         for row in csv.DictReader(source)])
    return runs


def read_summary(folder):
    with open(ROOT / "results" / folder / "summary.csv") as source:
        return list(csv.DictReader(source))


def mean_sd(values, digits=1):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def training_stats(folder):
    runs = read_runs(folder)
    summary = read_summary(folder)
    return {
        "eval_success": statistics.mean(float(r["success_rate"]) for r in summary),
        "eval_steps": summary[0]["mean_steps_success"],
        "eval_return": summary[0]["mean_return"],
        "first": mean_sd([next(r["episode"] for r in run if r["success"]) for run in runs]),
        "success100": mean_sd([statistics.mean(r["success"] for r in run[:100])
                               for run in runs], 2),
        "truncated": mean_sd([sum(r["truncated"] for r in run) for run in runs]),
        "total": mean_sd([sum(r["steps"] for r in run) for run in runs], 0),
        "steps500": mean_sd([statistics.mean(r["steps"] for r in run[-500:])
                             for run in runs]),
        "return500": mean_sd([statistics.mean(r["return"] for r in run[-500:])
                              for run in runs]),
        "best": ", ".join(str(min(r["steps"] for r in run)) for run in runs),
    }


def main():
    search = read_search()
    my_map = (ROOT / "maps" / "my_campus.txt").read_text().rstrip("\n")
    custom_bfs = search[("my_campus", "S-W", "bfs")]
    custom_astar = search[("my_campus", "S-W", "astar")]

    story = [Paragraph("COSC 461/561 Project 1: Eagle Delivery", title),
             Paragraph(AUTHOR + " &nbsp;&middot;&nbsp; " + REPO, byline)]

    # Part A
    story.append(Paragraph("Part A: Planning a delivery with A*", heading))
    story.append(para(
        "<b>Setting up the problem.</b> <i>State:</i> where the robot is, written as "
        "(row, column). It can be on any walkway or building entrance. <i>Start "
        "state:</i> the cell of the starting building, for example S = (1, 1). "
        "<i>Actions:</i> move UP, DOWN, LEFT, or RIGHT. <i>Transition model:</i> a move "
        "takes the robot one cell in that direction. If that cell is a wall (#) or off "
        "the map, the robot stays where it is. <i>Goal test:</i> the robot is on the "
        "destination building; passing other buildings does not count. <i>Path "
        "cost:</i> the number of moves, where each move costs 1. A move into a wall "
        "wastes a step and goes nowhere, so the search ignores those moves."))
    story.append(para(
        "<b>Why BFS finds the shortest route.</b> BFS first checks every cell 1 move "
        "away, then every cell 2 moves away, and so on. Every move costs the same (1), "
        "so the first time BFS reaches the goal, it has used the fewest moves possible."))
    story.append(para(
        "<b>Why Manhattan distance is admissible.</b> The heuristic h(s) = "
        "|r<sub>s</sub> - r<sub>g</sub>| + |c<sub>s</sub> - c<sub>g</sub>| counts how "
        "many rows and columns the robot still has to cross. Each move changes the row "
        "or the column by only one, so the robot needs at least that many moves even if "
        "there were no walls. Walls can only make the trip longer. So h never guesses "
        "more than the real distance, which is what admissible means. It is also "
        "consistent: one move changes h by at most 1."))
    story.append(para(
        "<b>How my A* works.</b> A* keeps a priority queue (a heap) of cells to visit, "
        "ordered by f = g + h, where g is the number of moves so far and h is the "
        "Manhattan guess. Each entry is (f, order added, g, cell), so when two cells "
        "have the same f, the one added first comes out first. For every cell I store "
        "the best g found so far and its parent (the cell it came from). If I find a "
        "shorter way to a cell, I update both and add the cell to the queue again. When "
        "an old, out-of-date entry comes out of the queue, I skip it. I check for the "
        "goal when a cell comes out of the queue, and I count a cell as expanded when I "
        "look at its neighbors, the same way the BFS code does."))
    rows = [["Map", "Delivery task", "BFS length", "A* length", "BFS expanded",
             "A* expanded"]]
    for name, label in (("emu_campus", "Baseline"), ("emu_detour", "Detour")):
        for task in ("S-W", "L-D", "H-P"):
            bfs, astar = search[(name, task, "bfs")], search[(name, task, "astar")]
            rows.append([label, task, bfs[0], astar[0], bfs[1], astar[1]])
    story.append(table(rows, [0.9 * inch, 0.95 * inch] + [0.9 * inch] * 4))
    story.append(Paragraph("Table 1. BFS vs A* on the six required tasks "
                           "(results/search_results.txt).", caption))
    story.append(para(
        "<b>What the table shows.</b> BFS and A* always find routes of the same length, "
        "because both always find a shortest route. A* expanded fewer cells in all six "
        "tasks, but how much it saves changes a lot. A* saves the most when the "
        "Manhattan guess is close to the real distance. For baseline S-W, the guess from "
        "S is 14 and the real route is 18, so A* expanded only 22 cells compared with 67 "
        "for BFS. H-P is similar (16 vs 41). For L-D, the guess is 14 but the real route "
        "is 22, because walls force a big detour. A* then has to check many cells that "
        "look promising but are not on the route, so it saves less (62 vs 79). In the "
        "detour map, the closed walkway at (3, 6) turns a corridor that points toward W "
        "into a dead end. A* goes down that corridor first, finds it blocked, and backs "
        "out, so its count rises from 22 to 51, while BFS barely changes (67 to 70). "
        "Detour L-D is the closest case (60 vs 65). So A* is not always much better: "
        "when the guess is misleading it wastes work, and in some maps it could even "
        "expand more cells than BFS."))

    # Part B
    story.append(Paragraph("Part B: Learning a delivery policy with Q-learning", heading))
    story.append(para(
        "<b>How my code works.</b> <font face='Courier'>epsilon_greedy</font> picks a "
        "random action with probability &epsilon; (using <font face='Courier'>"
        "rng.random()</font> and <font face='Courier'>rng.choice()</font>). Otherwise it "
        "picks the action with the highest Q-value, where actions never tried count as "
        "0. If several actions tie, it picks one of them at random. "
        "<font face='Courier'>q_learning_update</font> first works out a target. At the "
        "goal, the target is just the reward. Otherwise it is the reward plus &gamma; "
        "times the best Q-value at the next cell. Then it moves Q(s, a) toward that "
        "target by a step of size &alpha;: Q(s, a) &larr; Q(s, a) + &alpha;(target - "
        "Q(s, a)). All 14 tests pass. <b>Settings:</b> 2,000 training episodes, 100 "
        "test episodes, at most 200 moves per episode, &alpha; = 0.3, &gamma; = 0.95, "
        "&epsilon; = 0.2, no slipping, and seeds 0, 1, 2, each starting with an empty "
        "Q-table."))
    rows = [["Task", "Shortest route", "Test success", "Test moves (return)",
             "First goal reached (episode)", "Success rate, episodes 1-100",
             "Episodes that hit the 200-move limit", "Training moves, last 500 episodes"]]
    for task, folder in (("S-W", "baseline_SW"), ("L-D", "baseline_LD"),
                         ("H-P", "baseline_HP")):
        s = training_stats(folder)
        rows.append([task, search[("emu_campus", task, "bfs")][0],
                     f"{s['eval_success']:.0%}", f"{s['eval_steps']} ({s['eval_return']})",
                     s["first"], s["success100"], s["truncated"], s["steps500"]])
    story.append(table(rows, [0.5 * inch, 0.65 * inch, 0.65 * inch, 0.85 * inch,
                              0.85 * inch, 0.95 * inch, 0.9 * inch, 0.9 * inch]))
    story.append(Paragraph(
        "Table 2. Baseline Q-learning on the campus map. Each value is the average "
        "&plusmn; standard deviation over seeds 0, 1, 2.", caption))
    story.append(para(
        "<b>What I found.</b> All nine runs (3 tasks &times; 3 seeds) delivered the "
        "package 100% of the time in testing, always on the shortest route (18, 22, and "
        "16 moves). So the test returns were 40 - L: 22, 18, and 24, the same for every "
        "seed. Training happened in three stages. (1) <i>Exploring:</i> at the start the "
        "robot wanders a lot. Many early episodes hit the 200-move limit (8 to 18 per "
        "run, all before episode 150), but it still reached the goal within its first 6 "
        "episodes. All Q-values start at 0 and every move costs -1, so any action the "
        "robot has already tried looks worse than one it has not tried yet. That pushes "
        "it to try new paths. (2) <i>Learning:</i> the number of moves per episode drops "
        "fast (Figure 1). It levels off at about episode 60 for H-P, 120 to 140 for S-W, "
        "and 140 for L-D. L-D is slowest because its route is the longest, so the goal "
        "reward takes longer to spread back to the start. After episode 150, every "
        "training episode reached the goal. (3) <i>Leveling off:</i> training episodes "
        "stay a bit longer than the shortest route (22.5 vs 18, 27.7 vs 22, 20.1 vs 16), "
        "because with &epsilon; = 0.2 about 1 in 5 moves is still random. A longer route "
        "has more chances for a random wrong move, so L-D has the biggest gap."))
    story.append(figure("figures/baseline_training.png", 4.6 * inch))
    story.append(Paragraph(
        "Figure 1. Average moves per training episode, in groups of 20 episodes "
        "(3 seeds, log scale), for the first 600 of 2,000 episodes.", caption))

    # Part C
    story.append(Paragraph("Part C: Custom map and exploration experiment", heading))
    map_table = Table([[Preformatted(my_map, mono), Paragraph(
        "<b>My map (maps/my_campus.txt, 13 &times; 19).</b> S is at (1, 1) and W is at "
        "(10, 17). <b>Dead ends:</b> the biggest is a 20-cell branch right next to S. It "
        "goes east along row 1, dips down through (3, 7)-(3, 9), then continues east to "
        "(1, 17), right above W, and stops. It looks like a way to W but goes nowhere. "
        "Smaller dead ends lead to (3, 17), H, and D. <b>Loops:</b> the map has 3 loops: "
        "one in the middle through L, one around P, and one around the corridor "
        "(7, 13)-(7, 17) next to W. <b>Route choices:</b> every route goes down column 1 "
        "first. The shortest then goes east along the bottom row and up into W (31 "
        "moves). Going past P takes 35 moves, and using the corridor above W takes 37. "
        f"<b>Search:</b> BFS {custom_bfs[0]} moves / {custom_bfs[1]} expanded; A* "
        f"{custom_astar[0]} / {custom_astar[1]}. The dead-end branch tricks A*: its "
        "cells look close to W, so A* explores the whole branch before it finds the "
        "real route.", body)]],
        colWidths=[1.55 * inch, 5.15 * inch])
    map_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(map_table)
    story.append(para(
        "<b>Question:</b> How does the exploration rate &epsilon; change how fast the "
        "robot learns, how costly training is, and the final route it learns? "
        "<b>Prediction:</b> I expected a higher &epsilon; to find the goal faster but to "
        "get lower rewards during training, because more random moves would take the "
        "robot off the route."))
    story.append(para(
        "<b>What I did.</b> I changed only &epsilon; (0.05, 0.2, 0.5) on the S-W task. "
        "Everything else stayed the same as in Part B: &alpha; = 0.3, &gamma; = 0.95, "
        "2,000 training episodes, a 200-move limit, no slipping, and seeds 0, 1, 2 with "
        "a new empty Q-table each time. That makes 9 training runs. After training, each "
        "run was tested for 100 episodes with &epsilon; = 0 and no more learning. I "
        "compared the settings using the summary files and the training CSVs."))
    rows = [["&epsilon;", "Test success / moves", "First goal reached (episode)",
             "Success rate, episodes 1-100", "Episodes that hit the limit",
             "Total training moves", "Training moves, last 500", "Reward, last 500",
             "Best training episode"]]
    for epsilon, folder in (("0.05", "custom_e005"), ("0.2", "custom_e020"),
                            ("0.5", "custom_e050")):
        s = training_stats(folder)
        rows.append([epsilon, f"{s['eval_success']:.0%} / {s['eval_steps']}", s["first"],
                     s["success100"], s["truncated"], s["total"], s["steps500"],
                     s["return500"], s["best"]])
    story.append(table(rows, [0.4 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch,
                              0.75 * inch, 0.95 * inch, 0.75 * inch, 0.8 * inch,
                              0.75 * inch]))
    story.append(Paragraph(
        "Table 3. My map, S-W task, for each &epsilon; (average &plusmn; standard "
        "deviation over seeds 0-2). The shortest route is 31 moves (return 9). \"Best "
        "training episode\" lists the shortest training episode for each seed.", caption))
    story.append(figure("figures/custom_epsilon_training.png", 4.6 * inch))
    story.append(Paragraph("Figure 2. Average moves per training episode, in groups of "
                           "20 episodes (3 seeds, log scale), for each &epsilon;.",
                           caption))
    story.append(para(
        "<b>Results.</b> (1) <i>The final route was the same.</i> All nine runs reached W "
        "100% of the time in testing, using the shortest 31-move route. (2) <i>Learning "
        "speed barely changed, so the first part of my prediction was wrong.</i> A higher "
        "&epsilon; did not find the goal sooner: the first delivery came at episode 27, "
        "21, and 32 on average, and the results varied more between seeds than between "
        "settings. All three settings leveled off at about episode 180 to 220 (Figure 2). "
        "(3) <i>Training cost changed a lot, as I predicted.</i> As &epsilon; went up, "
        "the total moves in training went from 85k to 97k to 147k, and the average "
        "reward late in training dropped from +7.3 to +1.0 to -25.1. With &epsilon; = "
        "0.5 the robot never once took the perfect route during training (its best was "
        "36 to 39 moves), and it still failed some episodes late in training (as late as "
        "episode 1696). This makes sense: with &epsilon; = 0.5, each move is the planned "
        "one only 62.5% of the time, so making all 31 moves right in a row almost never "
        "happens (about 5 in 10 million). My map was also much harder than the campus "
        "map: in the first 100 episodes only 20-28% of episodes reached W (vs 83-91% on "
        "the campus map), because of the long route and the big dead end next to S."))
    story.append(para(
        "<b>What I learned.</b> The surprise was that &epsilon; hardly changed how fast "
        "the robot learned. I think there are two reasons. First, all Q-values start at "
        "0 and each move costs -1, so actions the robot has not tried look better than "
        "ones it has. Even without random moves, it keeps trying new paths, which is how "
        "it explored the dead end and the loops early on. Once the dead-end cells had "
        "negative values, it stopped going there. Second, Q-learning learns the best "
        "route even while it is moving randomly (it is \"off-policy\"), so extra random "
        "moves make training cost more but do not change what it learns. On this small "
        "map with no slipping, &epsilon; mostly controls how costly training is, and the "
        "smallest value, 0.05, worked best. I think &epsilon; would matter more with "
        "slipping, a much bigger map, or an &epsilon; that shrinks over time. Limits: "
        "only 3 seeds per setting, one task, and one map."))

    output = ROOT / "report" / "EagleDelivery_Report.pdf"
    document = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=0.8 * inch,
                                 rightMargin=0.8 * inch, topMargin=0.65 * inch,
                                 bottomMargin=0.65 * inch, title="Eagle Delivery Report",
                                 author=AUTHOR)
    document.build(story)
    print("Wrote", output)


if __name__ == "__main__":
    main()
