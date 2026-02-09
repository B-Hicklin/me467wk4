from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from UCS_pathfinder import SearchResult as UcsResult, ucs_path
from ASTAR_pathfinder import SearchResult as AstarResult, astar_path
from warehouse_env import WarehouseEnv
from warehouse_vis import replay_animation

@dataclass
class TrialStats:
    path_len: int
    nodes_expanded: int
    frontier_max: int
    time_sec: float
SearchResult = UcsResult | AstarResult

def _run_segment(
    grid: List[str],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    algo: str,
) -> SearchResult:
    if algo == "ucs":
        return ucs_path(grid, start, goal)
    if algo == "astar":
        return astar_path(grid, start, goal)
    raise ValueError(f"Unknown algo: {algo}")

def _aggregate(results: List[SearchResult]) -> TrialStats:
    path_len = sum(max(0, len(r.path) - 1) for r in results)
    nodes_expanded = sum(r.nodes_expanded for r in results)
    frontier_max = max((r.frontier_max for r in results), default=0)
    time_sec = sum(r.time_sec for r in results)
    return TrialStats(path_len, nodes_expanded, frontier_max, time_sec)

def run_trials(num_trials: int = 10) -> Dict[str, List[TrialStats]]:
    env = WarehouseEnv()
    stats: Dict[str, List[TrialStats]] = {"ucs": [], "astar": []}
    for _ in range(num_trials):
        obs = env.reset(randomize=True)
        start = obs["robot_pos"]
        pickup = obs["pickup_pos"]
        dropoff = obs["dropoff_pos"]
        if pickup is None or dropoff is None:
            continue
        for algo in ["ucs", "astar"]:
            seg1 = _run_segment(env.grid, start, pickup, algo)
            seg2 = _run_segment(env.grid, pickup, dropoff, algo)
            stats[algo].append(_aggregate([seg1, seg2]))
    return stats

def _frame_from_grid(grid: List[str], pos: Tuple[int, int], loaded: bool) -> List[List[str]]:
    rows = [list(r) for r in grid]
    r, c = pos
    rows[r][c] = "r" if loaded else "R"
    return rows

def visualize_sample() -> None:
    """Replay a single UCS run as an animation (start → pickup → dropoff)."""
    env = WarehouseEnv()
    obs = env.reset(randomize=True)
    start = obs["robot_pos"]
    pickup = obs["pickup_pos"]
    dropoff = obs["dropoff_pos"]
    if pickup is None or dropoff is None:
        return
    seg1 = ucs_path(env.grid, start, pickup)
    seg2 = ucs_path(env.grid, pickup, dropoff)
    frames: List[List[List[str]]] = [env.render_grid()]
    battery = [env.state.battery]
    rewards = [0.0]
    dist_pickup = [abs(start[0] - pickup[0]) + abs(start[1] - pickup[1])]
    dist_dropoff = [abs(start[0] - dropoff[0]) + abs(start[1] - dropoff[1])]
    def step_to(next_pos: Tuple[int, int]) -> None:
        r0, c0 = env.state.robot_pos
        r1, c1 = next_pos
        if (r1, c1) == (r0 - 1, c0):
            action = "N"
        elif (r1, c1) == (r0 + 1, c0):
            action = "S"
        elif (r1, c1) == (r0, c0 - 1):
            action = "W"
        elif (r1, c1) == (r0, c0 + 1):
            action = "E"
        else:
            raise ValueError(f"Non-adjacent step: {(r0, c0)} -> {(r1, c1)}")
        obs_step, reward, _, _, _ = env.step(action)
        frames.append(env.render_grid())
        battery.append(obs_step["battery"])
        rewards.append(reward)
        dist_pickup.append(abs(obs_step["robot_pos"][0] - pickup[0]) + abs(obs_step["robot_pos"][1] - pickup[1]))
        dist_dropoff.append(abs(obs_step["robot_pos"][0] - dropoff[0]) + abs(obs_step["robot_pos"][1] - dropoff[1]))
    for pos in seg1.path[1:]:
        step_to(pos)
    obs_step, reward, _, _, _ = env.step("PICK")
    frames.append(env.render_grid())
    battery.append(obs_step["battery"])
    rewards.append(reward)
    dist_pickup.append(abs(obs_step["robot_pos"][0] - pickup[0]) + abs(obs_step["robot_pos"][1] - pickup[1]))
    dist_dropoff.append(abs(obs_step["robot_pos"][0] - dropoff[0]) + abs(obs_step["robot_pos"][1] - dropoff[1]))
    for pos in seg2.path[1:]:
        step_to(pos)
    obs_step, reward, _, _, _ = env.step("DROP")
    frames.append(env.render_grid())
    battery.append(obs_step["battery"])
    rewards.append(reward)
    dist_pickup.append(abs(obs_step["robot_pos"][0] - pickup[0]) + abs(obs_step["robot_pos"][1] - pickup[1]))
    dist_dropoff.append(abs(obs_step["robot_pos"][0] - dropoff[0]) + abs(obs_step["robot_pos"][1] - dropoff[1]))
    metrics = {
        "battery": battery,
        "rewards": rewards,
        "dist_pickup": dist_pickup,
        "dist_dropoff": dist_dropoff,
    }
    anim = replay_animation(frames, metrics=metrics)
    
    
def _build_animation(env: WarehouseEnv, path_a: List[Tuple[int, int]], path_b: List[Tuple[int, int]]) -> None:
    frames: List[List[List[str]]] = [env.render_grid()]
    battery = [env.state.battery]
    rewards = [0.0]
    pickup = path_a[-1]
    dropoff = path_b[-1]
    dist_pickup = [abs(env.state.robot_pos[0] - pickup[0]) + abs(env.state.robot_pos[1] - pickup[1])]
    dist_dropoff = [abs(env.state.robot_pos[0] - dropoff[0]) + abs(env.state.robot_pos[1] - dropoff[1])]
    def step_to(next_pos: Tuple[int, int]) -> None:
        r0, c0 = env.state.robot_pos
        r1, c1 = next_pos
        if (r1, c1) == (r0 - 1, c0):
            action = "N"
        elif (r1, c1) == (r0 + 1, c0):
            action = "S"
        elif (r1, c1) == (r0, c0 - 1):
            action = "W"
        elif (r1, c1) == (r0, c0 + 1):
            action = "E"
        else:
            raise ValueError(f"Non-adjacent step: {(r0, c0)} -> {(r1, c1)}")
        obs_step, reward, _, _, _ = env.step(action)
        frames.append(env.render_grid())
        battery.append(obs_step["battery"])
        rewards.append(reward)
        dist_pickup.append(abs(obs_step["robot_pos"][0] - pickup[0]) + abs(obs_step["robot_pos"][1] - pickup[1]))
        dist_dropoff.append(abs(obs_step["robot_pos"][0] - dropoff[0]) + abs(obs_step["robot_pos"][1] - dropoff[1]))
    for pos in path_a[1:]:
        step_to(pos)
    obs_step, reward, _, _, _ = env.step("PICK")
    frames.append(env.render_grid())
    battery.append(obs_step["battery"])
    rewards.append(reward)
    dist_pickup.append(abs(obs_step["robot_pos"][0] - pickup[0]) + abs(obs_step["robot_pos"][1] - pickup[1]))
    dist_dropoff.append(abs(obs_step["robot_pos"][0] - dropoff[0]) + abs(obs_step["robot_pos"][1] - dropoff[1]))
    for pos in path_b[1:]:
        step_to(pos)
    obs_step, reward, _, _, _ = env.step("DROP")
    frames.append(env.render_grid())
    battery.append(obs_step["battery"])
    rewards.append(reward)
    dist_pickup.append(abs(obs_step["robot_pos"][0] - pickup[0]) + abs(obs_step["robot_pos"][1] - pickup[1]))
    dist_dropoff.append(abs(obs_step["robot_pos"][0] - dropoff[0]) + abs(obs_step["robot_pos"][1] - dropoff[1]))
    metrics = {
        "battery": battery,
        "rewards": rewards,
        "dist_pickup": dist_pickup,
        "dist_dropoff": dist_dropoff,
    }
    anim = replay_animation(frames, metrics=metrics)
    
def example_ucs_then_astar() -> None:
    """Run a single UCS example, then a single A* example, on the same layout."""
    env = WarehouseEnv()
    obs = env.reset(randomize=True)
    start = obs["robot_pos"]
    pickup = obs["pickup_pos"]
    dropoff = obs["dropoff_pos"]
    if pickup is None or dropoff is None:
        return
    ucs_seg1 = ucs_path(env.grid, start, pickup, record_expansions=True)
    ucs_seg2 = ucs_path(env.grid, pickup, dropoff, record_expansions=True)
    print("\nUCS example path length:", (len(ucs_seg1.path) - 1) + (len(ucs_seg2.path) - 1))
    _build_animation(WarehouseEnv(grid=env.grid, start_pos=start), ucs_seg1.path, ucs_seg2.path)
    astar_seg1 = astar_path(env.grid, start, pickup, record_expansions=True)
    astar_seg2 = astar_path(env.grid, pickup, dropoff, record_expansions=True)
    print("A* example path length:", (len(astar_seg1.path) - 1) + (len(astar_seg2.path) - 1))
    _build_animation(WarehouseEnv(grid=env.grid, start_pos=start), astar_seg1.path, astar_seg2.path)
    plot_expansion_maps(
        env.grid,
        ucs_seg1.expanded_order + ucs_seg2.expanded_order,
        astar_seg1.expanded_order + astar_seg2.expanded_order,
        start,
        pickup,
        dropoff,
        ucs_seg1.path + ucs_seg2.path[1:],
        astar_seg1.path + astar_seg2.path[1:],
    )

def plot_expansion_maps(
    grid: List[str],
    ucs_expanded: List[Tuple[int, int]],
    astar_expanded: List[Tuple[int, int]],
    start: Tuple[int, int],
    pickup: Tuple[int, int],
    dropoff: Tuple[int, int],
    ucs_path: List[Tuple[int, int]],
    astar_path: List[Tuple[int, int]],
) -> None:
    """Show where UCS vs A* expanded nodes on the same layout."""
    height = len(grid)
    width = len(grid[0]) if grid else 0
    def counts(expanded: List[Tuple[int, int]]) -> np.ndarray:
        heat = np.zeros((height, width), dtype=float)
        for r, c in expanded:
            heat[r, c] += 1.0
        return heat
    wall_mask = np.array([[1 if ch == "#" else 0 for ch in row] for row in grid])
    ucs_heat = counts(ucs_expanded)
    astar_heat = counts(astar_expanded)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    legend_handles = None
    last_im = None
    for ax, heat, title, path in zip(
        axes,
        [ucs_heat, astar_heat],
        ["UCS Expansions", "A* Expansions"],
        [ucs_path, astar_path],
    ):
        ax.imshow(wall_mask, cmap="gray", alpha=0.25)
        im = ax.imshow(heat, cmap="magma", alpha=0.85)
        last_im = im
        if path:
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            line = ax.plot(xs, ys, color="cyan", linewidth=2.0, alpha=0.9, label="Path")[0]
        else:
            line = ax.plot([], [], color="cyan", linewidth=2.0, alpha=0.9, label="Path")[0]
        start_handle = ax.scatter(start[1], start[0], s=60, c="#2ca02c", marker="o", label="Start")
        pickup_handle = ax.scatter(pickup[1], pickup[0], s=70, c="#1f77b4", marker="s", label="Pickup")
        dropoff_handle = ax.scatter(dropoff[1], dropoff[0], s=70, c="#d62728", marker="X", label="Dropoff")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if legend_handles is None:
            legend_handles = [line, start_handle, pickup_handle, dropoff_handle]
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=["Path", "Start", "Pickup", "Dropoff"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            fontsize=8,
            ncol=4,
        )
    fig.subplots_adjust(top=0.85, right=0.88)
    plt.show()

def summary_table(stats: Dict[str, List[TrialStats]]) -> None:
    def arr(algo: str, attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in stats[algo]])
    rows = []
    for algo in ["ucs", "astar"]:
        mean_time_ms = arr(algo, "time_sec").mean() * 1000.0
        rows.append(
            [
                algo.upper(),
                arr(algo, "path_len").mean(),
                arr(algo, "nodes_expanded").mean(),
                arr(algo, "frontier_max").mean(),
                mean_time_ms,
            ]
        )
    headers = ["Algorithm", "Mean Path Len", "Mean Nodes", "Mean Frontier", "Mean Time (ms)"]
    print("\nSummary (10 trials):")
    print("{:<10} {:>14} {:>12} {:>14} {:>14}".format(*headers))
    for row in rows:
        print(
            "{:<10} {:>14.2f} {:>12.2f} {:>14.2f} {:>14.2f}".format(
                row[0], row[1], row[2], row[3], row[4]
            )
        )

def plot_nodes(stats: Dict[str, List[TrialStats]]) -> None:
    means = [
        np.mean([s.nodes_expanded for s in stats["ucs"]]),
        np.mean([s.nodes_expanded for s in stats["astar"]]),
    ]
    labels = ["UCS", "A*"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, means, color=["#4e79a7", "#f28e2b"])
    plt.ylabel("Mean Nodes Expanded")
    plt.title("UCS vs A* (10 Random Trials)")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha="center")
    plt.tight_layout()
    plt.show()

stats = run_trials(num_trials=10)
summary_table(stats)
plot_nodes(stats)
visualize_sample()
example_ucs_then_astar()