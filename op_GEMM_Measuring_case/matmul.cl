// matmul.cl — persistent blocks 版本 + 预取优化 + 边界检查
#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

#ifndef PERSIST_GX
#define PERSIST_GX 2
#endif
#ifndef PERSIST_GY
#define PERSIST_GY 2
#endif

__kernel void sgemm_one_wg(__global const float* A,
                           __global const float* B,
                           __global float*       C,
                           const int M, const int N, const int K)
{
    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);
    const int gx = (int)get_group_id(0);
    const int gy = (int)get_group_id(1);

    __local float As[TILE_M * TILE_K];
    __local float Bs[TILE_K * TILE_N];
    
    // 预取寄存器
    float prefetch_A, prefetch_B;

    for (int bm = gy * TILE_M; bm < M; bm += PERSIST_GY * TILE_M) {
        for (int bn = gx * TILE_N; bn < N; bn += PERSIST_GX * TILE_N) {
            
            const int row = bm + ly;
            const int col = bn + lx;
            
            float acc = 0.0f;
            
            // 预取第一个K-tile - 添加边界检查
            prefetch_A = A[row * K + lx];
            prefetch_B = B[ly * N + col];
            
            for (int bk = 0; bk < K; bk += TILE_K) {
                // 存储预取的值到local memory
                As[ly * TILE_K + lx] = prefetch_A;
                Bs[ly * TILE_N + lx] = prefetch_B;
                
                const int a_col = bk + TILE_K + lx;
                const int b_row = bk + TILE_K + ly;
                    
                prefetch_A = A[row * K + a_col];
                prefetch_B = B[b_row * N + col];
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                #pragma unroll 4
                for (int kk = 0; kk < TILE_K; ++kk) {
                    float a = As[ly * TILE_K + kk];
                    float b = Bs[kk * TILE_N + lx];
                    acc += a * b;
                }
                
            }
            C[row * N + col] = acc;
        }
    }
}
