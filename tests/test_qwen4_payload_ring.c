#include "../ds4.c"
#include <assert.h>

int main(void) {
    assert(ds4_gpu_init());
    enum { ROWS = 23, CAP = 8 };
    const uint64_t width = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t rb = width * sizeof(float);
    float *logical = xmalloc(ROWS * rb), *physical = xmalloc(CAP * rb), *got = xmalloc(ROWS * rb);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint64_t col = 0; col < width; col++) {
            logical[row * width + col] = (float)(row * 1000 + col);
            physical[(row % CAP) * width + col] = logical[row * width + col];
        }
    }
    ds4_gpu_tensor *ring = ds4_gpu_tensor_alloc(CAP * rb);
    ds4_gpu_tensor *restored = ds4_gpu_tensor_alloc(CAP * rb);
    ds4_gpu_tensor *full = ds4_gpu_tensor_alloc(ROWS * rb);
    assert(ring && restored && full);
    assert(ds4_gpu_tensor_write(ring, 0, physical, CAP * rb));
    uint8_t buffer[4096]; char error[256] = {0};
    FILE *file = tmpfile(); assert(file);
    assert(!qwen4_payload_write_ik(file, ring, ROWS, buffer, sizeof(buffer), error, sizeof(error)));
    assert(ftell(file) == (long)(ROWS * rb));
    rewind(file); assert(fread(got, rb, ROWS, file) == ROWS);
    for (uint64_t i = 0; i < (ROWS-CAP)*width; i++) assert(got[i] == 0.0f);
    assert(!memcmp(got + (ROWS-CAP)*width, logical + (ROWS-CAP)*width, CAP * rb));
    rewind(file); uint64_t remaining = ROWS * rb;
    assert(!qwen4_payload_read_ik(file, restored, ROWS, buffer, sizeof(buffer), &remaining, error, sizeof(error)));
    assert(!remaining && ds4_gpu_tensor_read(restored, 0, got, CAP * rb));
    assert(!memcmp(got, physical, CAP * rb));
    rewind(file); remaining = ROWS * rb;
    assert(!qwen4_payload_read_ik(file, full, ROWS, buffer, sizeof(buffer), &remaining, error, sizeof(error)));
    assert(ds4_gpu_tensor_read(full, (ROWS-CAP)*rb, got, CAP*rb));
    assert(!memcmp(got, logical + (ROWS-CAP)*width, CAP*rb));
    fclose(file);
    file = tmpfile(); assert(file);
    assert(fwrite(logical, rb, ROWS, file) == ROWS); rewind(file); remaining = ROWS * rb;
    assert(!qwen4_payload_read_ik(file, restored, ROWS, buffer, sizeof(buffer), &remaining, error, sizeof(error)));
    assert(!remaining && ds4_gpu_tensor_read(restored, 0, got, CAP * rb));
    assert(!memcmp(got, physical, CAP * rb));
    fclose(file);
    ds4_gpu_tensor_free(ring); ds4_gpu_tensor_free(restored); ds4_gpu_tensor_free(full);
    free(logical); free(physical); free(got);
    puts("Qwen FP32 indexer ring: wrapped payload roundtrip, full/ring compatibility and exact tail OK");
    return 0;
}
