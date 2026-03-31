import tilelang
import tilelang.language as T
from tilelang import jit
import torch


def step_kernel(N, M, BLOCK_N, BLOCK_M, dtype, threads):

    @T.prim_func
    def main(A: T.Tensor((N, M), dtype), B: T.Tensor((N, M), dtype)): # A: input grid, B: output grid
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (bn, bm):
            rowstart = bn * BLOCK_N
            colstart = bm * BLOCK_M
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                x = rowstart + i
                y = colstart + j

                live_neighbors = [A[x+dx, y+dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
                
    
    return main

if __name__ == "__main__":
    program = step_kernel(200, 200, BLOCK_N=20, BLOCK_M=20, dtype=T.bfloat16, threads=256)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
