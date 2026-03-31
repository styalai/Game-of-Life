import pyxel
import random

CELL = 8          # pixel size of each cell
COLS = 20         # grid width  (160 / 8)
ROWS = 20         # grid height (160 / 8)
WIDTH  = COLS * CELL
HEIGHT = ROWS * CELL + 16  # extra 16px for toolbar
N_STEPS = 1


def make_grid(random_fill=True):
    if random_fill:
        return [[random.random() < 0.3 for _ in range(COLS)] for _ in range(ROWS)]
    return [[False] * COLS for _ in range(ROWS)]


def step(grid):
    new = [[False] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            live_neighbors = sum(
                grid[(r + dr) % ROWS][(c + dc) % COLS]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            )
            if grid[r][c]:
                new[r][c] = live_neighbors in (2, 3)
            else:
                new[r][c] = live_neighbors == 3
    return new


class GameOfLife:
    def __init__(self):
        pyxel.init(WIDTH, HEIGHT, title="Conway's Game of Life", fps=2)
        pyxel.mouse(True)
        self.grid = make_grid()
        self.running = False
        self.generation = 0
        pyxel.run(self.update, self.draw)

    def update(self):
        # Space  → play / pause
        if pyxel.btnp(pyxel.KEY_SPACE):
            self.running = not self.running

        # R      → random reset
        if pyxel.btnp(pyxel.KEY_R):
            self.grid = make_grid()
            self.generation = 0
            self.running = True

        # C      → clear
        if pyxel.btnp(pyxel.KEY_C):
            self.grid = make_grid(random_fill=False)
            self.generation = 0
            self.running = False

        # Left mouse → toggle cell
        if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            mx, my = pyxel.mouse_x, pyxel.mouse_y
            c = mx // CELL
            r = my // CELL
            if 0 <= r < ROWS and 0 <= c < COLS:
                self.grid[r][c] = not self.grid[r][c]

        # Advance simulation when running
        if self.running:
            for i in range(N_STEPS):
                self.grid = step(self.grid)
                self.generation += 1

    def draw(self):
        pyxel.cls(0)

        # Draw cells
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL
                y = r * CELL
                if self.grid[r][c]:
                    pyxel.rect(x + 1, y + 1, CELL - 1, CELL - 1, 11)  # green
                else:
                    pyxel.rect(x + 1, y + 1, CELL - 1, CELL - 1, 1)   # dark blue

        # Grid lines
        for c in range(COLS + 1):
            pyxel.line(c * CELL, 0, c * CELL, ROWS * CELL, 5)
        for r in range(ROWS + 1):
            pyxel.line(0, r * CELL, WIDTH, r * CELL, 5)

        # Toolbar
        toolbar_y = ROWS * CELL + 1
        state = "RUNNING" if self.running else "PAUSED "
        pyxel.text(2,  toolbar_y + 1, f"{state}  Gen:{self.generation:04d}", 7)
        pyxel.text(2,  toolbar_y + 8, "SPC:play/pause  R:reset  C:clear", 6)


GameOfLife()