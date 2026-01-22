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
            prefetch_A = (row < M && lx < K) ? A[row * K + lx] : 0.0f;
            prefetch_B = (ly < K && col < N) ? B[ly * N + col] : 0.0f;
            
            for (int bk = 0; bk < K; bk += TILE_K) {
                // 存储预取的值到local memory
                As[ly * TILE_K + lx] = prefetch_A;
                Bs[ly * TILE_N + lx] = prefetch_B;
                
                // 为下一个K-tile预取（如果不是最后一个）- 添加边界检查
                if (bk + TILE_K < K) {
                    const int a_col = bk + TILE_K + lx;
                    const int b_row = bk + TILE_K + ly;
                    
                    prefetch_A = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
                    prefetch_B = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
                } else {
                    // 如果是最后一个K-tile，将预取寄存器置零
                    prefetch_A = 0.0f;
                    prefetch_B = 0.0f;
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                #pragma unroll 4
                for (int kk = 0; kk < TILE_K; ++kk) {
                    float a = As[ly * TILE_K + kk];
                    float b = Bs[kk * TILE_N + lx];
                    acc += a * b;
                }
                
            }
            
            // 写入结果时添加边界检查
            if (row < M && col < N) {
                C[row * N + col] = acc;
            }
        }
    }
}
