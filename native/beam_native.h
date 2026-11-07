#ifndef BEAM_NATIVE_H
#define BEAM_NATIVE_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define BEAM_API __declspec(dllexport)
#else
#define BEAM_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define BEAM_NATIVE_ABI_VERSION 1u
typedef struct beam_context beam_context;
typedef enum beam_result {
    BEAM_OK = 0, BEAM_INVALID = 1, BEAM_BUSY = 2, BEAM_FAILED = 3,
    BEAM_CANCELLED = 4, BEAM_TIMEOUT = 5, BEAM_BUFFER_TOO_SMALL = 6
} beam_result;
typedef enum beam_phase {
    BEAM_CREATED = 0, BEAM_PREPARING = 1, BEAM_READY = 2,
    BEAM_RUNNING = 3, BEAM_STOPPED = 4, BEAM_ERROR = 5
} beam_phase;

/* Views are borrowed only for the duration of submit. No GPU address crosses
 * this interface. Native context owns its engine thread and GPU resources.
 * One context per worker PROCESS: the legacy engine uses process environment,
 * current directory and stdout. Never load this into a process using torch CUDA.
 * Library must remain loaded until all contexts have been destroyed. */
typedef struct beam_job_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t sequence;
    const uint8_t* state;
    uint64_t state_len;
    const char* output_directory;
    uint64_t output_directory_len;
} beam_job_v1;

typedef struct beam_status_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t phase;
    uint32_t result;
    uint64_t completed_jobs;
} beam_status_v1;

typedef struct beam_layout_v1 {
    uint32_t abi_version, struct_size;
    uint32_t state_len, storage_len, state_alignment, move_count;
    uint32_t candidate_bytes, candidate_alignment, hash_bytes, hash_alignment;
} beam_layout_v1;

BEAM_API uint32_t beam_native_abi_version(void);
BEAM_API beam_result beam_native_layout(beam_layout_v1*);
BEAM_API beam_result beam_context_create(uint32_t abi_version, beam_context** out);
/* argv is copied before returning; prepare starts asynchronous GPU setup. */
BEAM_API beam_result beam_context_prepare(beam_context*, int argc, const char* const* argv);
BEAM_API beam_result beam_context_submit(beam_context*, const beam_job_v1*);
/* Wait for preparation or current job; no polling of GPU counters here. */
BEAM_API beam_result beam_context_wait(beam_context*, uint64_t timeout_ms);
BEAM_API beam_result beam_context_status(beam_context*, beam_status_v1*);
/* Cooperative cancellation is terminal. Supervisor must kill ALL rank workers
 * after its grace period if a rank is blocked in an external collective. */
BEAM_API beam_result beam_context_cancel(beam_context*);
BEAM_API beam_result beam_context_error(beam_context*, char* dst, uint64_t capacity,
                                      uint64_t* required);
BEAM_API beam_result beam_context_destroy(beam_context*);
/* Coarse CLI compatibility entry point; catches every C++ exception. */
BEAM_API int beam_native_main(int argc, const char* const* argv);
#ifdef __cplusplus
}
#endif
#endif
