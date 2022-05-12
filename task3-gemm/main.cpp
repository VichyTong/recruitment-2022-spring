#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>

#define PRINT_TIME(code) do { \
    auto start = system_clock::now(); \
    code \
    auto end   = system_clock::now(); \
    auto duration = duration_cast<microseconds>(end - start); \
    cout << "time spent: " << double(duration.count()) << "us" << endl; \
} while(0)

using namespace std;

using namespace chrono;

using vec = vector<int>;

const int scale[] = {256, 512, 1024, 2048};
const string data_path("./data/");

int A[2060 * 2060], B[2060 * 2060], C[2060 * 2060];

#define mc 256
#define kc 128


void PackA(int k, int *a, int lda, int *a_to){
    int *a_i0_ptr = &a[0], *a_i1_ptr = &a[lda], *a_i2_ptr = &a[lda * 2], *a_i3_ptr = &a[lda * 3];
    for(int j = 0; j < k; j++){
        *a_to ++ = *a_i0_ptr ++;
        *a_to ++ = *a_i1_ptr ++;
        *a_to ++ = *a_i2_ptr ++;
        *a_to ++ = *a_i3_ptr ++;
    }
}
void PackB(int k, int *b, int ldb, int *b_to){
    for(int i = 0; i < k; i++){
        int *b_ij_ptr = &b[i * ldb];
        *(b_to + 0) = *b_ij_ptr;
        *(b_to + 1) = *(b_ij_ptr + 1);
        *(b_to + 2) = *(b_ij_ptr + 2);
        *(b_to + 3) = *(b_ij_ptr + 3);
        b_to += 4;
    }
}

void AddDot4x4(int K, int *a, int lda, int *b, int ldb, int *c, int ldc){
    int k;
    __m128i
            c_00_c_01_c_02_c_03_reg,
            c_10_c_11_c_12_c_13_reg,
            c_20_c_21_c_22_c_23_reg,
            c_30_c_31_c_32_c_33_reg,
            b_p0_b_p1_b_p2_b_p3_reg,
            a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    int *a0p_ptr, *a1p_ptr, *a2p_ptr, *a3p_ptr;
    a0p_ptr = &a[0];
    a1p_ptr = &a[lda];
    a2p_ptr = &a[lda * 2];
    a3p_ptr = &a[lda * 3];

    c_00_c_01_c_02_c_03_reg = _mm_setzero_si128();
    c_10_c_11_c_12_c_13_reg = _mm_setzero_si128();
    c_20_c_21_c_22_c_23_reg = _mm_setzero_si128();
    c_30_c_31_c_32_c_33_reg = _mm_setzero_si128();

    for(k = 0; k < K; k++){
        b_p0_b_p1_b_p2_b_p3_reg = _mm_load_si128((const __m128i*) &b[k * ldb]);
        register int ap = *a0p_ptr;
        a_0p_reg = _mm_set_epi32(ap, ap, ap, ap);
        ++ a0p_ptr;

        ap = *a1p_ptr;
        a_1p_reg = _mm_set_epi32(ap, ap, ap, ap);
        ++ a1p_ptr;

        ap = *a2p_ptr;
        a_2p_reg = _mm_set_epi32(ap, ap, ap, ap);
        ++ a2p_ptr;

        ap = *a3p_ptr;
        a_3p_reg = _mm_set_epi32(ap, ap, ap, ap);
        ++ a3p_ptr;

        c_00_c_01_c_02_c_03_reg = _mm_add_epi32( c_00_c_01_c_02_c_03_reg, _mm_mullo_epi32(a_0p_reg, b_p0_b_p1_b_p2_b_p3_reg));
        c_10_c_11_c_12_c_13_reg = _mm_add_epi32( c_10_c_11_c_12_c_13_reg, _mm_mullo_epi32(a_1p_reg, b_p0_b_p1_b_p2_b_p3_reg));
        c_20_c_21_c_22_c_23_reg = _mm_add_epi32( c_20_c_21_c_22_c_23_reg, _mm_mullo_epi32(a_2p_reg, b_p0_b_p1_b_p2_b_p3_reg));
        c_30_c_31_c_32_c_33_reg = _mm_add_epi32( c_30_c_31_c_32_c_33_reg, _mm_mullo_epi32(a_3p_reg, b_p0_b_p1_b_p2_b_p3_reg));

    }

    __m128i tmp = _mm_load_si128((const __m128i*) &c[0]);
    _mm_store_si128((__m128i *) &c[0], _mm_add_epi32(c_00_c_01_c_02_c_03_reg, tmp));
    tmp = _mm_load_si128((const __m128i*) &c[ldc]);
    _mm_store_si128((__m128i *) &c[ldc], _mm_add_epi32(c_10_c_11_c_12_c_13_reg, tmp));
    tmp = _mm_load_si128((const __m128i*) &c[2 * ldc]);
    _mm_store_si128((__m128i *) &c[2 * ldc], _mm_add_epi32(c_20_c_21_c_22_c_23_reg, tmp));
    tmp = _mm_load_si128((const __m128i*) &c[3 * ldc]);
    _mm_store_si128((__m128i *) &c[3 * ldc], _mm_add_epi32(c_30_c_31_c_32_c_33_reg, tmp));
}

void AddDot1x4(const int &size, int *a, int *b, int *c){ // no SIMD
    int k;
    register int
            c_00_reg = 0,   c_01_reg = 0,   c_02_reg = 0,   c_03_reg = 0,
            c_10_reg = 0,   c_11_reg = 0,   c_12_reg = 0,   c_13_reg = 0,
            c_20_reg = 0,   c_21_reg = 0,   c_22_reg = 0,   c_23_reg = 0,
            c_30_reg = 0,   c_31_reg = 0,   c_32_reg = 0,   c_33_reg = 0,
            b_0p_reg = 0,   b_1p_reg = 0,   b_2p_reg = 0,   b_3p_reg = 0,
            a_p0_reg = 0,   a_p1_reg = 0,   a_p2_reg = 0,   a_p3_reg = 0;

    int *ap0_ptr, *ap1_ptr, *ap2_ptr, *ap3_ptr;
    ap0_ptr = &a[0];
    ap1_ptr = &a[size];
    ap2_ptr = &a[size * 2];
    ap3_ptr = &a[size * 3];

    for(k = 0; k < size; k++){
        b_0p_reg = b[k * size];
        b_1p_reg = b[k * size + 1];
        b_2p_reg = b[k * size + 2];
        b_3p_reg = b[k * size + 3];

        a_p0_reg = *ap0_ptr ++;
        a_p1_reg = *ap1_ptr ++;
        a_p2_reg = *ap2_ptr ++;
        a_p3_reg = *ap3_ptr ++;

        c_00_reg += a_p0_reg * b_0p_reg;
        c_10_reg += a_p1_reg * b_0p_reg;

        c_01_reg += a_p0_reg * b_1p_reg;
        c_11_reg += a_p1_reg * b_1p_reg;

        c_02_reg += a_p0_reg * b_2p_reg;
        c_12_reg += a_p1_reg * b_2p_reg;

        c_03_reg += a_p0_reg * b_3p_reg;
        c_13_reg += a_p1_reg * b_3p_reg;

        c_20_reg += a_p2_reg * b_0p_reg;
        c_30_reg += a_p3_reg * b_0p_reg;

        c_21_reg += a_p2_reg * b_1p_reg;
        c_31_reg += a_p3_reg * b_1p_reg;

        c_22_reg += a_p2_reg * b_2p_reg;
        c_32_reg += a_p3_reg * b_2p_reg;

        c_23_reg += a_p2_reg * b_3p_reg;
        c_33_reg += a_p3_reg * b_3p_reg;
    }

    c[0 * size + 0] += c_00_reg; c[0 * size + 1] += c_01_reg; c[0 * size + 2] += c_02_reg; c[0 * size + 3] += c_03_reg;
    c[1 * size + 0] += c_10_reg; c[1 * size + 1] += c_11_reg; c[1 * size + 2] += c_12_reg; c[1 * size + 3] += c_13_reg;
    c[2 * size + 0] += c_20_reg; c[2 * size + 1] += c_21_reg; c[2 * size + 2] += c_22_reg; c[2 * size + 3] += c_23_reg;
    c[3 * size + 0] += c_30_reg; c[3 * size + 1] += c_31_reg; c[3 * size + 2] += c_32_reg; c[3 * size + 3] += c_33_reg;
}

void InnerKernel(int n, int m, int k, int *a, int lda, int *b, int ldb, int *c, int ldc){
#pragma omp parallel for
    for (int i = 0; i < n ; i += 4 ) {
        for (int j = 0; j < m; j += 4 ) {
            AddDot4x4(k, &a[i * lda], lda, &b[j], ldb, &c[i * ldc + j], ldc);
        }
    }
}
void Gemm(const int &size, vec &a, vec &b, vec &c) {
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            A[i * size + j] = a[i * size + j];
            B[i * size + j] = b[i * size + j];
            C[i * size + j] = 0;
        }
    }
    const int N = size;
    int kb, jb;
    for (int k = 0; k < size; k += kc ){
        kb = min(N - k, kc);
        for (int j = 0; j < size; j += mc ){
            jb = min(N - j, mc);
            InnerKernel(N, jb, kb, &A[k], N, &B[k * N + j], N, &C[j], N);
        }
    }

//#pragma omp parallel for
//    for(int i = 0; i < size; i+=4){
//        for(int j = 0; j < size; j+=4){
//            AddDot1x4(size, &A[i * size], &B[j], &C[i * size + j]);
//        }
//    }

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            c[i * size + j] = C[i * size + j];
        }
    }
}

void CheckResult(const vec &c, const string &result_path) {
    ifstream file_result(result_path);
    int nelems = c.size();
    float res_i;
    for(int i = 0; i < nelems; i++) {
        file_result >> res_i;
        assert(c[i] == res_i);
    }
    file_result.close();
}

// c = a * b
void Benchmark(const int &size) {
    const int nelems = size * size;
    const string a_path(data_path+to_string(size)+"/a");
    const string b_path(data_path+to_string(size)+"/b");
    const string result_path(data_path+to_string(size)+"/result");
    ifstream file_a(a_path);
    ifstream file_b(b_path);

    vec a(nelems, 0);
    vec b(nelems, 0);
    vec c(nelems, 0);

    for(int i = 0; i < nelems; i++) {
        file_a >> a[i];
    }
    for(int i = 0; i < nelems; i++) {
        file_b >> b[i];
    }

    PRINT_TIME(
            Gemm(size, a, b, c);
    );

    CheckResult(c, result_path);

    file_a.close();
    file_b.close();
}

int main() {
    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}