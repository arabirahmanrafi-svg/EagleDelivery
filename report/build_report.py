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
        "<b>Formulation.</b> A <i>state</i> is the robot's cell (row, column) on any "
        "traversable square (walkway or building entrance). The <i>initial state</i> is "
        "the start entrance's cell, e.g. S = (1, 1). The <i>actions</i> are UP, DOWN, "
        "LEFT, RIGHT. The <i>transition model</i> is deterministic: a move shifts the "
        "robot one cell if the target is inside the map and not a wall (#), and "
        "otherwise leaves it in place. The <i>goal test</i> is position = destination "
        "entrance; other entrances are ordinary walkways. The <i>path cost</i> is the "
        "number of actions, each costing 1. Search only uses moves that change the "
        "position, because a blocked move costs 1 and returns to the same state, so it "
        "is never part of a cheapest route."))
    story.append(para(
        "<b>Why BFS is optimal.</b> BFS removes states in order of depth, and it records "
        "each state the first time it reaches it, which is through a shallowest path. "
        "Every action costs 1, so depth equals path cost, and the first time the goal is "
        "removed it lies on a minimum-cost route. <b>Why Manhattan distance is "
        "admissible.</b> Each action changes the row or the column by at most 1, so any "
        "route from s to g needs at least |r<sub>s</sub> - r<sub>g</sub>| + "
        "|c<sub>s</sub> - c<sub>g</sub>| actions. Walls only remove moves and there are "
        "no diagonals, so h(s) never overestimates the true remaining cost. It is also "
        "consistent: one unit-cost move changes h by at most 1."))
    story.append(para(
        "<b>Implementation.</b> The heap holds (f, insertion number, g, state), so "
        "equal-f ties pop in insertion order. The <font face='Courier'>best_g</font> and "
        "<font face='Courier'>parents</font> dictionaries are updated whenever a strictly "
        "cheaper g is found, and a new entry is pushed. A popped entry whose g no longer "
        "matches <font face='Courier'>best_g</font> is stale and skipped. The goal test "
        "runs when an entry is removed, and a node counts as expanded when its successors "
        "are generated, the same convention as BFS."))
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
        "<b>Discussion.</b> Both algorithms find routes of the same length in every case, "
        "as expected, since both are optimal here. A* expanded fewer nodes in all six "
        "cases, but how many fewer depends on how closely h matches the real distance. "
        "When the walls bend the route only a little, A* expands barely more than the "
        "route itself: baseline S-W has h(S) = 14 and cost 18 (22 vs 67 expansions), and "
        "H-P has h = 12 and cost 16 (16 vs 41). For L-D, h(L) = 14 but the cost is 22, "
        "because the route has to swing far away from the straight line. A* must then "
        "expand every state with f &lt; 22, which covers much of the map (62 vs 79). The "
        "closure at (3, 6) in the detour map turns the corridor that h points toward into "
        "a dead end. On detour S-W, A* follows that corridor first and then backs out, so "
        "its expansions rise from 22 to 51 while BFS barely changes (67 to 70). Detour L-D "
        "is the closest case (60 vs 65). A* is not guaranteed to expand fewer nodes; a "
        "misleading heuristic and ties among states with f equal to the optimal cost can "
        "make it explore states BFS would not need."))

    # Part B
    story.append(Paragraph("Part B: Learning a delivery policy with Q-learning", heading))
    story.append(para(
        "<b>Implementation.</b> <font face='Courier'>epsilon_greedy</font> draws "
        "<font face='Courier'>rng.random()</font>; below &epsilon; it returns "
        "<font face='Courier'>rng.choice</font> over all four actions. Otherwise it finds "
        "the largest Q-value (unseen entries count as 0.0) and uses "
        "<font face='Courier'>rng.choice</font> to break ties among the actions that reach "
        "it. <font face='Courier'>q_learning_update</font> uses the target y = r for a "
        "terminal transition and y = r + &gamma; max<sub>a'</sub> Q(s', a') otherwise "
        "(unseen actions count as 0.0), then updates only Q(s, a) &larr; Q(s, a) + "
        "&alpha;(y - Q(s, a)). All 14 public tests pass. Baseline settings: 2,000 "
        "training episodes, 100 greedy evaluation episodes, a 200-action cap, &alpha; = "
        "0.3, &gamma; = 0.95, constant &epsilon; = 0.2, no slip, and seeds 0, 1, 2 with a "
        "fresh Q-table each."))
    rows = [["Task", "Optimal length", "Eval success", "Eval steps (return)",
             "First success (ep.)", "Success rate, eps 1-100", "Truncated train eps",
             "Train steps, last 500"]]
    for task, folder in (("S-W", "baseline_SW"), ("L-D", "baseline_LD"),
                         ("H-P", "baseline_HP")):
        s = training_stats(folder)
        rows.append([task, search[("emu_campus", task, "bfs")][0],
                     f"{s['eval_success']:.0%}", f"{s['eval_steps']} ({s['eval_return']})",
                     s["first"], s["success100"], s["truncated"], s["steps500"]])
    story.append(table(rows, [0.5 * inch, 0.65 * inch, 0.65 * inch, 0.85 * inch,
                              0.85 * inch, 0.95 * inch, 0.9 * inch, 0.9 * inch]))
    story.append(Paragraph(
        "Table 2. Baseline Q-learning on emu_campus, mean &plusmn; sample SD over seeds "
        "0, 1, 2 (from the training, evaluation, and summary CSVs).", caption))
    story.append(para(
        "<b>Observations.</b> All nine runs reach 100% evaluation success, and every "
        "evaluation episode follows a shortest route (18, 22, and 16 actions), so the "
        "returns equal 40 - L (22, 18, 24) with zero variance across seeds. Training goes "
        "through three phases. (1) <i>Exploration:</i> in the first 100 episodes many "
        "episodes hit the 200-action limit (8-18 truncated episodes per run, all within "
        "the first 150), yet the first delivery comes within the first 6 episodes. The "
        "Q-values start at 0 and every step costs -1, so an action that has been tried "
        "quickly drops below 0 and the untried actions win the greedy choice; the agent "
        "sweeps the map systematically. (2) <i>Convergence:</i> the training curves "
        "(Figure 1) drop to a plateau by about episode 60 for H-P, 120-140 for S-W, and "
        "140 for L-D, matching task difficulty: L-D has the longest route, and the +39 "
        "reward must propagate back through 22 states. After episode 150 "
        "every training episode succeeds for every task. (3) <i>Plateau:</i> training "
        "steps level off above the optimum (22.5 vs 18, 27.7 vs 22, 20.1 vs 16) because "
        "&epsilon; = 0.2 keeps adding random detours. Longer routes give exploration more "
        "chances to stray, so the gap is largest for L-D."))
    story.append(figure("figures/baseline_training.png", 4.6 * inch))
    story.append(Paragraph(
        "Figure 1. Mean training steps per episode in 20-episode blocks (averaged over "
        "3 seeds, log scale) for the first 600 of 2,000 episodes.", caption))

    # Part C
    story.append(Paragraph("Part C: Custom map and exploration experiment", heading))
    map_table = Table([[Preformatted(my_map, mono), Paragraph(
        "<b>Map (maps/my_campus.txt, 13 &times; 19).</b> S = (1, 1), W = (10, 17); "
        "110 open cells and 112 connections, so 3 independent loops. <b>Dead ends:</b> "
        "the largest is the 20-cell branch that starts right next to S: it runs east "
        "along row 1, dips through (3, 7)-(3, 9), and continues east to (1, 17), right "
        "above W's column, where it stops. Smaller dead ends lead to (3, 17), to H, and "
        "to D. <b>Loops:</b> a middle loop through L, and two loops on the east side "
        "around P and around the (7, 13)-(7, 17) corridor next to W. <b>Route "
        "choices:</b> every route goes down column 1. The shortest then follows row 11 "
        "east and steps up into W (31 actions). Going up through (10, 9) past P costs 35, "
        "and taking the corridor above W costs 37. <b>Search:</b> BFS "
        f"{custom_bfs[0]} actions / {custom_bfs[1]} expanded; A* {custom_astar[0]} / "
        f"{custom_astar[1]}. h(S) = 25: along that branch f = g + h stays between 25 "
        "and 27, below the true cost of 31, so A* explores the whole dead-end branch "
        "before taking the real route.", body)]],
        colWidths=[1.55 * inch, 5.15 * inch])
    map_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(map_table)
    story.append(para(
        "<b>Question and prediction.</b> How does the exploration rate &epsilon; affect "
        "learning speed, the cost of training, and the final policy on this map? "
        "<i>Prediction:</i> higher &epsilon; would find the goal faster, but it would "
        "give lower training returns, because random moves keep pulling the agent off "
        "the route."))
    story.append(para(
        "<b>Method.</b> I varied only &epsilon; (0.05, 0.2, 0.5) on the S-W task. "
        "Everything else stayed at the Part B defaults: &alpha; = 0.3, &gamma; = 0.95, "
        "2,000 training episodes, a 200-action cap, no slip, and seeds 0, 1, 2 with a "
        "fresh Q-table for every (setting, seed) pair, giving 9 training runs. Each run "
        "was evaluated for 100 episodes with &epsilon; = 0 and frozen Q-values. I compared "
        "the settings using the evaluation summaries and the raw training CSVs."))
    rows = [["&epsilon;", "Eval success / steps", "First success (ep.)",
             "Success rate, eps 1-100", "Truncated train eps", "Total train actions",
             "Train steps, last 500", "Train return, last 500", "Best train episode"]]
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
        "Table 3. my_campus S-W by exploration rate (mean &plusmn; SD over seeds 0-2). "
        "Optimal route = 31 actions (return 9). \"Best train episode\" lists each seed's "
        "shortest training episode.", caption))
    story.append(figure("figures/custom_epsilon_training.png", 4.6 * inch))
    story.append(Paragraph("Figure 2. Mean training steps per episode (20-episode "
                           "blocks, 3 seeds, log scale) for each &epsilon;.", caption))
    story.append(para(
        "<b>Findings.</b> (1) <i>Final policy: no difference.</i> All nine runs evaluate "
        "at 100% success along the 31-action route (return 9). (2) <i>Learning speed: "
        "little difference, so the first half of my prediction was wrong.</i> Higher "
        "&epsilon; did not find the goal sooner: the first delivery came at episode "
        "27, 21, and 32 on average, and the seed-to-seed spread is larger than the gaps between "
        "settings. All three curves reach their plateau at about episode 180-220. (3) "
        "<i>Training cost: a large difference, as predicted.</i> As &epsilon; rises, total "
        "training actions grow from 85k to 97k to 147k, and late-training returns fall "
        "from +7.3 to +1.0 to -25.1. With &epsilon; = 0.5 the agent never walked the "
        "optimal route during training (its best episodes took 36-39 actions), and it "
        "still had occasional failed episodes late in training (as late as episode 1696). "
        "Each step picks the greedy action with probability 0.5 + 0.5/4 = 0.625, so a "
        "perfect 31-step episode has probability of only about 0.625<sup>31</sup> &asymp; "
        "5 &times; 10<sup>-7</sup>. This map was also much harder than the campus map: in "
        "the first 100 episodes only 20-28% of episodes succeeded (vs 83-91% for the "
        "baseline tasks), because the 20-cell dead end starts right beside S and the "
        "route is 31 steps long."))
    story.append(para(
        "<b>Discussion.</b> The surprise is that &epsilon; barely changed how fast the "
        "agent learned. Two properties of this setup explain it. First, with Q-values "
        "starting at 0 and a -1 reward per step, every tried action looks worse than an "
        "untried one, so even the greedy choice explores systematically. This optimism, "
        "not random &epsilon; moves, drives the agent through the dead end and around "
        "the loops early on; once the dead-end cells hold negative values they stop "
        "attracting it. Second, Q-learning is off-policy: it learns the values of the "
        "greedy policy no matter how the data was collected, so heavy exploration lowers "
        "training returns without changing what is learned. On this small deterministic "
        "map, &epsilon; therefore acts mainly as a cost setting, and the smallest value "
        "was best. I would expect &epsilon; to matter more with slip noise, a much "
        "larger map, or a decaying &epsilon;. Limitations: only three seeds per setting, "
        "one task, and one map."))

    output = ROOT / "report" / "EagleDelivery_Report.pdf"
    document = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=0.8 * inch,
                                 rightMargin=0.8 * inch, topMargin=0.65 * inch,
                                 bottomMargin=0.65 * inch, title="Eagle Delivery Report",
                                 author=AUTHOR)
    document.build(story)
    print("Wrote", output)


if __name__ == "__main__":
    main()
