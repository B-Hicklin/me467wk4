"""Uniform Cost Search (UCS) pathfinding for the warehouse grid."""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Tuple
Position = Tuple[int, int]
@dataclass(frozen=True)
class SearchResult:
    path: List[Position]
    cost: int
    nodes_expanded: int
    frontier_max: int
    time_sec: float
    expanded_order: List[Position]
def _neighbors(grid: List[str], pos: Position) -> Iterable[Position]:
    r, c = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if nr < 0 or nc < 0 or nr >= len(grid) or nc >= len(grid[0]):
            continue
        if grid[nr][nc] != "#":
            yield (nr, nc)
def _reconstruct(came_from: Dict[Position, Optional[Position]], goal: Position) -> List[Position]:
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
def ucs_path(
    grid: List[str],
    start: Position,
    goal: Position,
    record_expansions: bool = False,
) -> SearchResult:
    """Compute the lowest-cost path on a grid using UCS (Dijkstra).
    Cost per move is 1. Walls are '#'. Returns path + search statistics.
    """
    import time
    start_time = time.perf_counter()
    # frontier is a min-heap of (path_cost, tie_breaker, position)
    frontier: List[Tuple[int, int, Position]] = []
    heappush(frontier, (0, 0, start))
    # came_from lets us reconstruct the path at the end
    came_from: Dict[Position, Optional[Position]] = {start: None}
    # cost_so_far stores the best known cost to each position
    cost_so_far: Dict[Position, int] = {start: 0}
    nodes_expanded = 0
    frontier_max = 1
    tie = 0
    expanded_order: List[Position] = []
    while frontier:
        # Always expand the cheapest path so far
        cost, _, current = heappop(frontier)
        nodes_expanded += 1
        if record_expansions:
            expanded_order.append(current)
        # Goal test: stop once we pop the goal from the frontier
        if current == goal:
            path = _reconstruct(came_from, goal)
            return SearchResult(
                path=path,
                cost=cost,
                nodes_expanded=nodes_expanded,
                frontier_max=frontier_max,
                time_sec=time.perf_counter() - start_time,
                expanded_order=expanded_order,
            )
        # Explore neighbors (up, down, left, right) and relax edges
        for nxt in _neighbors(grid, current):
            new_cost = cost_so_far[current] + 1  # each move costs 1
            # If this is a better path to nxt, record it and push to frontier
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                tie += 1  # ensures heap order is stable when costs tie
                heappush(frontier, (new_cost, tie, nxt))
                if len(frontier) > frontier_max:
                    frontier_max = len(frontier)
    return SearchResult(
        path=[],
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        frontier_max=frontier_max,
        time_sec=time.perf_counter() - start_time,
        expanded_order=expanded_order,
    )
if __name__ == "__main__":
    # Simple smoke test on the default warehouse grid.
    from warehouse_env import WarehouseEnv
    env = WarehouseEnv()
    obs = env.reset(randomize=False)
    start = obs["robot_pos"]
    pickup = obs["pickup_pos"]
    if pickup is None:
        raise RuntimeError("No pickup tile found in grid.")
    result = ucs_path(env.grid, start, pickup)
    print("UCS path length:", len(result.path) - 1)
    print("Nodes expanded:", result.nodes_expanded)