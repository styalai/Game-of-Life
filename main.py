import random, pyxel, time, torch
from kernel import step_kernel
import tilelang
import tilelang.language as T
from tqdm import tqdm
import matplotlib.pyplot as plt

def make_grid(random_fill=True):
    if random_fill:
        return torch.tensor([[int(random.random() < 0.3) for _ in range(COLS)] for _ in range(ROWS)])
    return torch.tensor([[0] * COLS for _ in range(ROWS)])


def step(grid):
    new = [[0] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            live_neighbors = sum(
                grid[max(0, min(ROWS-1, r + dr))][max(0, min(COLS-1, c + dc))]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            )
            if grid[r][c]:
                new[r][c] = live_neighbors in (2, 3)
            else:
                new[r][c] = live_neighbors == 3
    return new

def visualize_grids(grid_a, grid_b, title_a="Grid A", title_b="Grid B"):
    """
    Open a Pyxel window and display two Game-of-Life grids side by side.

    Parameters
    ----------
    grid_a  : list[list[bool]]   Left grid  (ROWS x COLS)
    grid_b  : list[list[bool]]   Right grid (ROWS x COLS)
    title_a : str                Label shown above the left grid
    title_b : str                Label shown above the right grid
    """

    offset_b = COLS * CELL + PADDING   # x-origin of the right grid

    def draw():
        pyxel.cls(0)

        for grid, ox in ((grid_a, 0), (grid_b, offset_b)):
            for r in range(ROWS):
                for c in range(COLS):
                    color = 11 if grid[r][c] else 1
                    pyxel.rect(ox + c * CELL + 1, r * CELL + 1,
                               CELL - 1, CELL - 1, color)

            # grid lines
            for c in range(COLS + 1):
                pyxel.line(ox + c * CELL, 0,
                           ox + c * CELL, ROWS * CELL, 5)
            for r in range(ROWS + 1):
                pyxel.line(ox, r * CELL,
                           ox + COLS * CELL, r * CELL, 5)

        # separator
        sx = COLS * CELL + PADDING // 2
        pyxel.line(sx, 0, sx, ROWS * CELL, 13)

        # labels
        ty = ROWS * CELL + 4
        pyxel.text(2,          ty, title_a[:20], 7)
        pyxel.text(offset_b,   ty, title_b[:20], 7)
        pyxel.text(WIDTH - 38, ty, "Q:quit", 6)

    def update():
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    pyxel.init(WIDTH, HEIGHT, title=f"{title_a} vs {title_b}")
    pyxel.mouse(True)
    pyxel.run(update, draw)


if __name__ == "__main__":

    CELL    = 8
    COLS    = 10000
    ROWS    = 10000
    PADDING = 8   # gap between the two grids

    WIDTH  = COLS * CELL * 2 + PADDING
    HEIGHT = ROWS * CELL + 16   # 16px for labels

    program = step_kernel(ROWS, COLS, BLOCK_N=16, BLOCK_M=16, dtype=T.float16, threads=256)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")

    a = make_grid()
    n_steps = 1000

    # without kernel
    #start = time.time()
    #bcpu = a
    #for i in tqdm(range(n_steps)):
    #    bcpu = step(bcpu)
    #elapsed = time.time() - start
    #print(f"step() {n_steps} steps: {elapsed:.6f}s  ({elapsed/n_steps*1000:.3f} ms/step)")

    # with kernel
    start = time.time()
    b = a.clone().cuda().half()
    for i in range(n_steps):
        b = kernel(b)
    elapsed = time.time() - start
    print(f"kernel() {n_steps} steps: {elapsed:.6f}s  ({elapsed/n_steps*1000:.3f} ms/step)")

    plt.imshow(a, cmap="binary")
    plt.title(f"original")
    plt.axis("off")

    #plt.imshow(bcpu, cmap="binary")
    #plt.title(f"cpu Step {i}")
    #plt.axis("off")

    plt.imshow(b.cpu(), cmap="binary")
    plt.title(f"gpu Step {i}")
    plt.axis("off")
    plt.show()
    
    #visualize_grids(a, b,
    #                title_a="original", title_b=f"After {n_steps}")

# 100x100: kernel() 10000 steps: 0.335843s  (0.034 ms/step)
# 1000x1000: kernel() 10000 steps: 0.837147s  (0.084 ms/step