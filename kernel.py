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

                count: T.int32 = 0
                # manually unrolled 8 neighbors
                nx = T.clamp(x - 1, 0, N - 1)
                ny = T.clamp(y - 1, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x - 1, 0, N - 1)
                ny = T.clamp(y, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x - 1, 0, N - 1)
                ny = T.clamp(y + 1, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x, 0, N - 1)
                ny = T.clamp(y - 1, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x, 0, N - 1)
                ny = T.clamp(y + 1, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x + 1, 0, N - 1)
                ny = T.clamp(y - 1, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x + 1, 0, N - 1)
                ny = T.clamp(y, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)

                nx = T.clamp(x + 1, 0, N - 1)
                ny = T.clamp(y + 1, 0, M - 1)
                count += T.cast(A[nx, ny], T.int32)
                                                                    
                cell = T.cast(A[x, y], dtype)
                alive = T.cast(
                    (count == 3) or (count == 2 and cell > 0),
                    dtype
                )
                B[x, y] = alive
    
    return main

if __name__ == "__main__":
    program = step_kernel(200, 200, BLOCK_N=16, BLOCK_M=16, dtype=T.float16, threads=256)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")

    
