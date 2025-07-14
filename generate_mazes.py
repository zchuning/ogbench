import argparse
import json
import os
import random
import sys


def generate_maze_prim(m):
    """
    Generate a perfect maze on an m×m grid using Randomized Prim’s,
    with a solid wall border (1) and passages only inside.
    Returns a list-of-lists where 1=wall, 0=passage.
    """
    # start with all walls
    grid = [[1] * m for _ in range(m)]

    # pick a random starting cell inside the border
    r = random.randrange(1, m - 1)
    c = random.randrange(1, m - 1)
    grid[r][c] = 0

    # frontier set: neighbors of the carved region, but only interior cells
    frontier = set()
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nr, nc = r + dr, c + dc
        if 1 <= nr < m - 1 and 1 <= nc < m - 1:
            frontier.add((nr, nc))

    # carve until no frontier remains
    while frontier:
        fr, fc = random.choice(tuple(frontier))
        frontier.remove((fr, fc))

        # count adjacent passages (only direct neighbors)
        neigh_passages = 0
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = fr + dr, fc + dc
            if grid[nr][nc] == 0:
                neigh_passages += 1

        # carve if it connects to exactly one existing passage
        if neigh_passages == 1:
            grid[fr][fc] = 0
            # add its interior wall-neighbors to frontier
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = fr + dr, fc + dc
                if 1 <= nr < m - 1 and 1 <= nc < m - 1 and grid[nr][nc] == 1:
                    frontier.add((nr, nc))

    # enforce outer border as walls (just in case)
    for i in range(m):
        grid[0][i] = grid[m - 1][i] = 1
        grid[i][0] = grid[i][m - 1] = 1

    return grid


def save_maze(maze, filename):
    with open(filename, "w") as f:
        for row in maze:
            # make row string like "[1, 0, 1, ...]"
            row_str = ", ".join(map(str, row))
            f.write(f"{row_str}\n")


def main(m, n, output_dir="mazes", prefix="maze"):
    """
    Generate n mazes of size m×m with solid borders and save each
    to prefix_i.txt as a human-readable array of arrays.
    """
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        maze = generate_maze_prim(m)
        filename = os.path.join(output_dir, f"{prefix}_{i}.txt")
        save_maze(maze, filename)
        print(f"Saved maze #{i} → {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate N Prim's-algorithm mazes on an M×M grid"
    )
    parser.add_argument(
        "-m",
        "--size",
        type=int,
        default=8,
        help="Grid size (M). Maze will be M×M. (default: 8).",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=5,
        help="Number of mazes to generate (default: 5).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="mazes",
        help="Directory to save maze files (default: 'mazes').",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="maze",
        help="Filename prefix (default: 'maze').",
    )
    args = parser.parse_args()

    main(args.size, args.count, args.output_dir, args.prefix)
