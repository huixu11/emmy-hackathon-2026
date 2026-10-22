extern "C" __global__
__launch_bounds__(256) void k_matmul_2db247(const float* gelu_inv_sqrt2, const float* gelu_half, const float* gelu_one, const float* b, const float* x, const float* w, float* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1376) {
        int a0_b = _gid / 86;
        int a1_b = (_gid) % 86;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c2_2 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c3_2 = 0.0f;
        for (int a2 = 0; a2 < 128; a2++) {
            float in4__ar0 = x[(a0_b * 4 + a0_u * 4) * 128 + a2];
            float in3__bc0 = w[a2 * 256 + (a1_b * 3 + a1_u * 3) % 256];
            float v0__c0_0 = in3__bc0 * in4__ar0;
            acc0__c0_0 += v0__c0_0;
            float in3__bc1 = w[a2 * 256 + (a1_b * 3 + a1_u * 3 + 1) % 256];
            float v0__c0_1 = in3__bc1 * in4__ar0;
            acc0__c0_1 += v0__c0_1;
            float in3__bc2 = w[a2 * 256 + (a1_b * 3 + a1_u * 3 + 2) % 256];
            float v0__c0_2 = in3__bc2 * in4__ar0;
            acc0__c0_2 += v0__c0_2;
            float in4__ar1 = x[(a0_b * 4 + a0_u * 4 + 1) * 128 + a2];
            float v0__c1_0 = in3__bc0 * in4__ar1;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in3__bc1 * in4__ar1;
            acc0__c1_1 += v0__c1_1;
            float v0__c1_2 = in3__bc2 * in4__ar1;
            acc0__c1_2 += v0__c1_2;
            float in4__ar2 = x[(a0_b * 4 + a0_u * 4 + 2) * 128 + a2];
            float v0__c2_0 = in3__bc0 * in4__ar2;
            acc0__c2_0 += v0__c2_0;
            float v0__c2_1 = in3__bc1 * in4__ar2;
            acc0__c2_1 += v0__c2_1;
            float v0__c2_2 = in3__bc2 * in4__ar2;
            acc0__c2_2 += v0__c2_2;
            float in4__ar3 = x[(a0_b * 4 + a0_u * 4 + 3) * 128 + a2];
            float v0__c3_0 = in3__bc0 * in4__ar3;
            acc0__c3_0 += v0__c3_0;
            float v0__c3_1 = in3__bc1 * in4__ar3;
            acc0__c3_1 += v0__c3_1;
            float v0__c3_2 = in3__bc2 * in4__ar3;
            acc0__c3_2 += v0__c3_2;
        }
        float in5__c0_0 = b[(a1_b * 3 + a1_u * 3) % 256];
        float v1__c0_0 = acc0__c0_0 + in5__c0_0;
        float in0__c0_0 = gelu_half[0];
        float in1__c0_0 = gelu_inv_sqrt2[0];
        float in2__c0_0 = gelu_one[0];
        float v6__c0_0 = (in2__c0_0 + erff((in1__c0_0 * v1__c0_0))) * (in0__c0_0 * v1__c0_0);
        if (a1_b * 3 + a1_u * 3 + 0 < 256) {
            gelu[(a0_b * 4 + a0_u * 4) * 256 + (a1_b * 3 + a1_u * 3) % 256] = v6__c0_0;
        }
        float in5__c0_1 = b[(a1_b * 3 + a1_u * 3 + 1) % 256];
        float v1__c0_1 = acc0__c0_1 + in5__c0_1;
        float in0__c0_1 = gelu_half[0];
        float in1__c0_1 = gelu_inv_sqrt2[0];
        float in2__c0_1 = gelu_one[0];
        float v6__c0_1 = (in2__c0_1 + erff((in1__c0_1 * v1__c0_1))) * (in0__c0_1 * v1__c0_1);
        if (a1_b * 3 + a1_u * 3 + 1 < 256) {
            gelu[(a0_b * 4 + a0_u * 4) * 256 + (a1_b * 3 + a1_u * 3 + 1) % 256] = v6__c0_1;
        }
        float in5__c0_2 = b[(a1_b * 3 + a1_u * 3 + 2) % 256];
        float v1__c0_2 = acc0__c0_2 + in5__c0_2;
        float in0__c0_2 = gelu_half[0];
        float in1__c0_2 = gelu_inv_sqrt2[0];
        float in2__c0_2 = gelu_one[0];
        float v6__c0_2 = (in2__c0_2 + erff((in1__c0_2 * v1__c0_2))) * (in0__c0_2 * v1__c0_2);
        if (a1_b * 3 + a1_u * 3 + 2 < 256) {
            gelu[(a0_b * 4 + a0_u * 4) * 256 + (a1_b * 3 + a1_u * 3 + 2) % 256] = v6__c0_2;
        }
        float in5__c1_0 = b[(a1_b * 3 + a1_u * 3) % 256];
        float v1__c1_0 = acc0__c1_0 + in5__c1_0;
        float in0__c1_0 = gelu_half[0];
        float in1__c1_0 = gelu_inv_sqrt2[0];
        float in2__c1_0 = gelu_one[0];
        float v6__c1_0 = (in2__c1_0 + erff((in1__c1_0 * v1__c1_0))) * (in0__c1_0 * v1__c1_0);
        if (a1_b * 3 + a1_u * 3 + 0 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 1) * 256 + (a1_b * 3 + a1_u * 3) % 256] = v6__c1_0;
        }
        float in5__c1_1 = b[(a1_b * 3 + a1_u * 3 + 1) % 256];
        float v1__c1_1 = acc0__c1_1 + in5__c1_1;
        float in0__c1_1 = gelu_half[0];
        float in1__c1_1 = gelu_inv_sqrt2[0];
        float in2__c1_1 = gelu_one[0];
        float v6__c1_1 = (in2__c1_1 + erff((in1__c1_1 * v1__c1_1))) * (in0__c1_1 * v1__c1_1);
        if (a1_b * 3 + a1_u * 3 + 1 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 1) * 256 + (a1_b * 3 + a1_u * 3 + 1) % 256] = v6__c1_1;
        }
        float in5__c1_2 = b[(a1_b * 3 + a1_u * 3 + 2) % 256];
        float v1__c1_2 = acc0__c1_2 + in5__c1_2;
        float in0__c1_2 = gelu_half[0];
        float in1__c1_2 = gelu_inv_sqrt2[0];
        float in2__c1_2 = gelu_one[0];
        float v6__c1_2 = (in2__c1_2 + erff((in1__c1_2 * v1__c1_2))) * (in0__c1_2 * v1__c1_2);
        if (a1_b * 3 + a1_u * 3 + 2 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 1) * 256 + (a1_b * 3 + a1_u * 3 + 2) % 256] = v6__c1_2;
        }
        float in5__c2_0 = b[(a1_b * 3 + a1_u * 3) % 256];
        float v1__c2_0 = acc0__c2_0 + in5__c2_0;
        float in0__c2_0 = gelu_half[0];
        float in1__c2_0 = gelu_inv_sqrt2[0];
        float in2__c2_0 = gelu_one[0];
        float v6__c2_0 = (in2__c2_0 + erff((in1__c2_0 * v1__c2_0))) * (in0__c2_0 * v1__c2_0);
        if (a1_b * 3 + a1_u * 3 + 0 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 2) * 256 + (a1_b * 3 + a1_u * 3) % 256] = v6__c2_0;
        }
        float in5__c2_1 = b[(a1_b * 3 + a1_u * 3 + 1) % 256];
        float v1__c2_1 = acc0__c2_1 + in5__c2_1;
        float in0__c2_1 = gelu_half[0];
        float in1__c2_1 = gelu_inv_sqrt2[0];
        float in2__c2_1 = gelu_one[0];
        float v6__c2_1 = (in2__c2_1 + erff((in1__c2_1 * v1__c2_1))) * (in0__c2_1 * v1__c2_1);
        if (a1_b * 3 + a1_u * 3 + 1 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 2) * 256 + (a1_b * 3 + a1_u * 3 + 1) % 256] = v6__c2_1;
        }
        float in5__c2_2 = b[(a1_b * 3 + a1_u * 3 + 2) % 256];
        float v1__c2_2 = acc0__c2_2 + in5__c2_2;
        float in0__c2_2 = gelu_half[0];
        float in1__c2_2 = gelu_inv_sqrt2[0];
        float in2__c2_2 = gelu_one[0];
        float v6__c2_2 = (in2__c2_2 + erff((in1__c2_2 * v1__c2_2))) * (in0__c2_2 * v1__c2_2);
        if (a1_b * 3 + a1_u * 3 + 2 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 2) * 256 + (a1_b * 3 + a1_u * 3 + 2) % 256] = v6__c2_2;
        }
        float in5__c3_0 = b[(a1_b * 3 + a1_u * 3) % 256];
        float v1__c3_0 = acc0__c3_0 + in5__c3_0;
        float in0__c3_0 = gelu_half[0];
        float in1__c3_0 = gelu_inv_sqrt2[0];
        float in2__c3_0 = gelu_one[0];
        float v6__c3_0 = (in2__c3_0 + erff((in1__c3_0 * v1__c3_0))) * (in0__c3_0 * v1__c3_0);
        if (a1_b * 3 + a1_u * 3 + 0 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 3) * 256 + (a1_b * 3 + a1_u * 3) % 256] = v6__c3_0;
        }
        float in5__c3_1 = b[(a1_b * 3 + a1_u * 3 + 1) % 256];
        float v1__c3_1 = acc0__c3_1 + in5__c3_1;
        float in0__c3_1 = gelu_half[0];
        float in1__c3_1 = gelu_inv_sqrt2[0];
        float in2__c3_1 = gelu_one[0];
        float v6__c3_1 = (in2__c3_1 + erff((in1__c3_1 * v1__c3_1))) * (in0__c3_1 * v1__c3_1);
        if (a1_b * 3 + a1_u * 3 + 1 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 3) * 256 + (a1_b * 3 + a1_u * 3 + 1) % 256] = v6__c3_1;
        }
        float in5__c3_2 = b[(a1_b * 3 + a1_u * 3 + 2) % 256];
        float v1__c3_2 = acc0__c3_2 + in5__c3_2;
        float in0__c3_2 = gelu_half[0];
        float in1__c3_2 = gelu_inv_sqrt2[0];
        float in2__c3_2 = gelu_one[0];
        float v6__c3_2 = (in2__c3_2 + erff((in1__c3_2 * v1__c3_2))) * (in0__c3_2 * v1__c3_2);
        if (a1_b * 3 + a1_u * 3 + 2 < 256) {
            gelu[(a0_b * 4 + a0_u * 4 + 3) * 256 + (a1_b * 3 + a1_u * 3 + 2) % 256] = v6__c3_2;
        }
    }
}
