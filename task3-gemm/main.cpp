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

typedef union{
    __m128i v;
    int d[4];
} v4i;

void AddDot4x4(const int &size, int *a, int *b, int *c){
    int k;
    v4i
        c_00_c_01_c_02_c_03_reg,
        c_10_c_11_c_12_c_13_reg,
        c_20_c_21_c_22_c_23_reg,
        c_30_c_31_c_32_c_33_reg,
        b_p0_b_p1_b_p2_b_p3_reg,
        a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    int *a0p_ptr, *a1p_ptr, *a2p_ptr, *a3p_ptr;
    a0p_ptr = &a[0];
    a1p_ptr = &a[size];
    a2p_ptr = &a[size * 2];
    a3p_ptr = &a[size * 3];

    c_00_c_01_c_02_c_03_reg.v = _mm_setzero_si128();
    c_10_c_11_c_12_c_13_reg.v = _mm_setzero_si128();
    c_20_c_21_c_22_c_23_reg.v = _mm_setzero_si128();
    c_30_c_31_c_32_c_33_reg.v = _mm_setzero_si128();

    for(k = 0; k < size; k++){
        b_p0_b_p1_b_p2_b_p3_reg.v = _mm_set_epi32(b[k * size + 3], b[k * size + 2], b[k * size + 1], b[k * size]);
        int ap = *a0p_ptr;
//        printf("*** %d ***\n",ap);
        a_0p_reg.v = _mm_set_epi32(ap, ap, ap, ap);
        ++ a0p_ptr;

        ap = *a1p_ptr;
//        printf("*** %d ***\n",ap);
        a_1p_reg.v = _mm_set_epi32(ap, ap, ap, ap);
        ++ a1p_ptr;

        ap = *a2p_ptr;
//        printf("*** %d ***\n",ap);
        a_2p_reg.v = _mm_set_epi32(ap, ap, ap, ap);
        ++ a2p_ptr;

        ap = *a3p_ptr;
//        printf("*** %d ***\n",ap);
        a_3p_reg.v = _mm_set_epi32(ap, ap, ap, ap);
        ++ a3p_ptr;

//        getchar();
        c_00_c_01_c_02_c_03_reg.v = _mm_add_epi32( c_00_c_01_c_02_c_03_reg.v, _mm_mullo_epi32(a_0p_reg.v, b_p0_b_p1_b_p2_b_p3_reg.v));
        c_10_c_11_c_12_c_13_reg.v = _mm_add_epi32( c_10_c_11_c_12_c_13_reg.v, _mm_mullo_epi32(a_1p_reg.v, b_p0_b_p1_b_p2_b_p3_reg.v));
        c_20_c_21_c_22_c_23_reg.v = _mm_add_epi32( c_20_c_21_c_22_c_23_reg.v, _mm_mullo_epi32(a_2p_reg.v, b_p0_b_p1_b_p2_b_p3_reg.v));
        c_30_c_31_c_32_c_33_reg.v = _mm_add_epi32( c_30_c_31_c_32_c_33_reg.v, _mm_mullo_epi32(a_3p_reg.v, b_p0_b_p1_b_p2_b_p3_reg.v));
//        printf("%d %d %d %d\n",c_00_c_01_c_02_c_03_reg.d[0],c_00_c_01_c_02_c_03_reg.d[1],c_00_c_01_c_02_c_03_reg.d[2],c_00_c_01_c_02_c_03_reg.d[3]);
//        printf("%d %d %d %d\n",c_10_c_11_c_12_c_13_reg.d[0],c_10_c_11_c_12_c_13_reg.d[1],c_10_c_11_c_12_c_13_reg.d[2],c_10_c_11_c_12_c_13_reg.d[3]);
//        printf("%d %d %d %d\n",c_20_c_21_c_22_c_23_reg.d[0],c_20_c_21_c_22_c_23_reg.d[1],c_20_c_21_c_22_c_23_reg.d[2],c_20_c_21_c_22_c_23_reg.d[3]);
//        printf("%d %d %d %d\n",c_30_c_31_c_32_c_33_reg.d[0],c_30_c_31_c_32_c_33_reg.d[1],c_30_c_31_c_32_c_33_reg.d[2],c_30_c_31_c_32_c_33_reg.d[3]);
//        getchar();
    }

    c[0 * size + 0] += c_00_c_01_c_02_c_03_reg.d[0];
    c[0 * size + 1] += c_00_c_01_c_02_c_03_reg.d[1];
    c[0 * size + 2] += c_00_c_01_c_02_c_03_reg.d[2];
    c[0 * size + 3] += c_00_c_01_c_02_c_03_reg.d[3];
    c[1 * size + 0] += c_10_c_11_c_12_c_13_reg.d[0];
    c[1 * size + 1] += c_10_c_11_c_12_c_13_reg.d[1];
    c[1 * size + 2] += c_10_c_11_c_12_c_13_reg.d[2];
    c[1 * size + 3] += c_10_c_11_c_12_c_13_reg.d[3];
    c[2 * size + 0] += c_20_c_21_c_22_c_23_reg.d[0];
    c[2 * size + 1] += c_20_c_21_c_22_c_23_reg.d[1];
    c[2 * size + 2] += c_20_c_21_c_22_c_23_reg.d[2];
    c[2 * size + 3] += c_20_c_21_c_22_c_23_reg.d[3];
    c[3 * size + 0] += c_30_c_31_c_32_c_33_reg.d[0];
    c[3 * size + 1] += c_30_c_31_c_32_c_33_reg.d[1];
    c[3 * size + 2] += c_30_c_31_c_32_c_33_reg.d[2];
    c[3 * size + 3] += c_30_c_31_c_32_c_33_reg.d[3];
}
void Gemm(const int &size, vec &a, vec &b, vec &c) {
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            A[i * size + j] = a[i * size + j];
            B[i * size + j] = b[i * size + j];
            C[i * size + j] = 0;
        }
    }
    for(int i = 0; i < size; i+=4){
        for(int j = 0; j < size; j+=4){
            AddDot4x4(size, &A[i * size], &B[j], &C[i * size + j]);
        }
    }
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
//    for(int i = 0; i < nelems; i++){
//        printf("%d ", c[i]);
//    }
//    getchar();
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