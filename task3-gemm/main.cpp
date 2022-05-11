#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <emmintrin.h>

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
void AddDot1x4(const int &size, int *a, int *b, int *c){
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
            AddDot1x4(size, &A[i * size], &B[j], &C[i * size + j]);
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