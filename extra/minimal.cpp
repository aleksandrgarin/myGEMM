#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <CL/cl.h>

#define NUM_RUNS 10
#define SIZE 4096

#define TS 16
#define WPT 8

const char *kernelstring =
    "#define RTS (TS/WPT)\n"
    "__kernel void myGEMM4(const int M, const int N, const int K,"
    "                      const __global float* A,"
    "                      const __global float* B,"
    "                      __global float* C) {"
    "    const int row = get_local_id(0);"
    "    const int col = get_local_id(1);"
    "    const int globalRow = TS*get_group_id(0) + row;"
    "    const int globalCol = TS*get_group_id(1) + col;"
    "    __local float Asub[TS][TS];"
    "    __local float Bsub[TS][TS];"
    "    float acc[WPT];"
    "    for (int w=0; w<WPT; w++) {"
    "        acc[w] = 0.0f;"
    "    }"
    "    const int numTiles = K/TS;"
    "    for (int t=0; t<numTiles; t++) {"
    "        for (int w=0; w<WPT; w++) {"
    "            const int tiledRow = TS*t + row;"
    "            const int tiledCol = TS*t + col + w*RTS;"
    "            Asub[col + w*RTS][row] = A[(tiledCol)*M + globalRow];"
    "            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];"
    "        }"
    "        barrier(CLK_LOCAL_MEM_FENCE);"
    "        for (int k=0; k<TS; k++) {"
    "            for (int w=0; w<WPT; w++) {"
    "                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];"
    "            }"
    "        }"
    "        barrier(CLK_LOCAL_MEM_FENCE);"
    "    }"
    "    for (int w=0; w<WPT; w++) {"
    "        C[(globalCol + w*RTS)*M + globalRow] = acc[w];"
    "    }"
    "}";

int main(int argc, char* argv[]) {
    int K = SIZE;
    int M = SIZE;
    int N = SIZE;

    float* A = (float*)malloc((size_t)M*(size_t)K*sizeof(float));
    float* B = (float*)malloc((size_t)K*(size_t)N*sizeof(float));
    float* C = (float*)malloc((size_t)M*(size_t)N*sizeof(float));
    for (int i=0; i<M*K; i++) { A[i] = 3.6*i + i*i + 3.1; }
    for (int i=0; i<K*N; i++) { B[i] = 1.2*i + 0.01*i*i + 13.9; }
    for (int i=0; i<M*N; i++) { C[i] = 0.0; }

    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id device = 0;
    cl_platform_id selected_platform = 0;

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_int err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            selected_platform = platforms[i];
            break;
        }
    }

    free(platforms);
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)selected_platform, 0 };
    cl_int err;
    cl_context context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_event event = NULL;

    char compile_options[128];
    sprintf(compile_options, "-DTS=%d -DWPT=%d", TS, WPT);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelstring, NULL, NULL);
    cl_int build_err = clBuildProgram(program, 1, &device, compile_options, NULL, NULL);

    if (build_err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* messages = (char*)malloc((1+logSize)*sizeof(char));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
        printf(">>> OPENCL BUILD ERROR LOG:\n%s\n", messages);
        free(messages);
        return 1;
    }

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  (size_t)M*(size_t)K*sizeof(float), NULL, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  (size_t)K*(size_t)N*sizeof(float), NULL, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)M*(size_t)N*sizeof(float), NULL, &err);

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, (size_t)M*(size_t)K*sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, (size_t)K*(size_t)N*sizeof(float), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, (size_t)M*(size_t)N*sizeof(float), C, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "myGEMM4", &err);
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&M);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);

    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;

    for (int r=0; r<NUM_RUNS; r++) {
        auto run_start = std::chrono::high_resolution_clock::now();

        const size_t local[2] = { TS, (size_t)(TS / WPT) };
        const size_t global[2] = { (size_t)M, (size_t)(N / WPT) };
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
        if (err != CL_SUCCESS) return 1;

        clWaitForEvents(1, &event);
        auto run_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = run_end - run_start;
        double t = diff.count();

        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }

    double avg_time = total_time / (double)NUM_RUNS;
    double operations = (double)K * (double)M * (double)N * 2.0;

    double avg_gflop = operations / (1000.0*1000.0*1000.0 * avg_time);
    double min_gflop = operations / (1000.0*1000.0*1000.0 * max_time);
    double max_gflop = operations / (1000.0*1000.0*1000.0 * min_time);
    double spread = (max_gflop - min_gflop) / 2.0;

    printf(">>> Performance: %.1lf GFLOPS (±%.1lf GFLOPS)\n", avg_gflop, spread);

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    clReleaseProgram(program); clReleaseKernel(kernel);
    free(A); free(B); free(C);

    return 0;
}