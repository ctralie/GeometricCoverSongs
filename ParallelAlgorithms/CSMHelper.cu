__global__ void getSumSquares(float* X, float* XSqr, int dim, int dimpow2) {
    extern __shared__ float x[];
    int i, k;
    int offset = blockIdx.x*dim;
    int jump = dimpow2;
    int sumjump = jump >> 1;
    //Step 0: Figure out K (number of batches per block)
    int K = dimpow2 >> 9;
    if (K == 0) {
        K = 1;
    }
    if (jump > 512) {
        jump = 512;
    }
    //Step 1: Copy over each row to shared memory
    //and square in the process
    for (k = 0; k < K; k++) {
        i = k*jump + threadIdx.x;
        if (i < dim) {
            x[i] = X[offset + i]*X[offset + i];
        }
        else if (i < dimpow2) {
            x[i] = 0.0;
        }
    }
    __syncthreads();
    //Step 2: Perform sums
    while (sumjump > 0) {
        if (threadIdx.x < sumjump) {
            K = sumjump >> 9;
            if (K == 0) {
                K = 1;
            }
            jump = sumjump;
            if (jump > 512) {
                jump = 512;
            }
            for (k = 0; k < K; k++) {
                i = k*jump + threadIdx.x;
                x[i] += x[i + sumjump];
            }
        }
        sumjump = sumjump >> 1;
        __syncthreads();
    }

    //Step 3: Copy back results
    XSqr[blockIdx.x] = x[0];
}

//CSM is N x M
__global__ void finishCSM(float* CSM, float* XSqr, float* YSqr, int N, int M, int MPow2) {
    int offset = blockIdx.x*M;
    int K = MPow2 >> 9;
    int i;
    int k, jump = MPow2;
    float val = 0.0;
    if (K == 0) {
        K = 1;
    }
    if (jump > 512) {
        jump = 512;
    }
    for (k = 0; k < K; k++) {
        i = k*jump + threadIdx.x;
        if (i < M) {
            val = XSqr[i] + YSqr[blockIdx.x];
            val = val - 2*CSM[offset + i];
            if (val < 0) {
                val = 0;
            }
            CSM[offset + i] = sqrt(val);
        }
    }
}
