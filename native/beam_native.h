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

typedef struct beam_history_entry_v1 {
    uint64_t parent_idx;
    uint32_t route, pad;
} beam_history_entry_v1;
typedef struct beam_history_stats_v1 {
    uint64_t bytes_received, bytes_stored, bytes_stored_ram, bytes_stored_disk, bytes_pruned;
    uint64_t bytes_pinned_slots, bytes_slot_staging, bytes_static_ram_arena, bytes_static_disk_arena;
    uint64_t required_entries, ram_reserved, disk_reserved, states_before_target_beam;
    uint32_t effective_depth, target_beam_depth;
} beam_history_stats_v1;
/* Optional ABI-1 host-history extension. The context copies this table before
 * prepare. user_data and callbacks live until destroy joins the engine AND its
 * pinned-slot readers. Each write borrows count aligned CandidateMeta records
 * (32 bytes each) only until returning; CUDA completion precedes that call.
 * Callbacks may run on native writer threads and must contain Rust panics. */
typedef struct beam_history_api_v1 {
    uint32_t abi_version, struct_size;
    void* user_data;
    int32_t (*initialize)(void*, const char* json, uint64_t bytes);
    int32_t (*write)(void*, uint32_t depth, const void* candidates, uint64_t count);
    int32_t (*read)(void*, uint32_t depth, uint64_t index, beam_history_entry_v1*);
    int32_t (*stats)(void*, beam_history_stats_v1*);
    int32_t (*close)(void*);
    int32_t (*error)(void*, char*, uint64_t capacity, uint64_t* required);
} beam_history_api_v1;

typedef enum beam_depth_checkpoint_v1 {
    BEAM_DEPTH_BEFORE_FRONTIER_COLLECTIVE = 1,
    BEAM_DEPTH_AFTER_FRONTIER_COLLECTIVE = 2
} beam_depth_checkpoint_v1;
typedef enum beam_depth_checkpoint_v2 {
    BEAM_DEPTH_V2_BEFORE_FRONTIER_COLLECTIVE = 1,
    BEAM_DEPTH_V2_AFTER_FRONTIER_COLLECTIVE = 2,
    BEAM_DEPTH_V2_SOLVE_BUCKET_STOP = 3
} beam_depth_checkpoint_v2;
typedef enum beam_depth_stop_reason_v2 {
    BEAM_DEPTH_STOP_REASON_NONE = 0,
    BEAM_DEPTH_STOP_REASON_DEPTH = 1,
    BEAM_DEPTH_STOP_REASON_CAPACITY = 2
} beam_depth_stop_reason_v2;
typedef struct beam_depth_control_input_v1 {
    uint32_t abi_version, struct_size;
    uint32_t checkpoint;
    uint32_t calibration_limit_reached;
    uint32_t frontier_live;
    uint32_t tracked_path_stop;
} beam_depth_control_input_v1;
typedef struct beam_depth_control_output_v1 {
    uint32_t abi_version, struct_size;
    uint32_t continue_search;
} beam_depth_control_output_v1;
/* ABI-1 retains the original two-checkpoint scalar callback. */
typedef struct beam_depth_control_api_v1 {
    uint32_t abi_version, struct_size;
    void* user_data;
    int32_t (*decide)(void*, const beam_depth_control_input_v1*, beam_depth_control_output_v1*);
    int32_t (*error)(void*, char*, uint64_t capacity, uint64_t* required);
} beam_depth_control_api_v1;

typedef struct beam_depth_control_input_v2 {
    uint32_t abi_version, struct_size;
    uint32_t checkpoint;
    uint32_t calibration_limit_reached;
    uint32_t frontier_live;
    uint32_t tracked_path_stop;
    uint64_t max_solutions;
    uint64_t accepted_solutions;
    uint32_t completed_depths;
    uint32_t current_depth_index;
    uint32_t stop_depth;
    uint32_t first_found_depth_index;
    uint32_t extra_depths;
    uint32_t solve_bucket_found_any;
} beam_depth_control_input_v2;
typedef struct beam_depth_control_output_v2 {
    uint32_t abi_version, struct_size;
    uint32_t continue_search;
    uint32_t stop_reason;
} beam_depth_control_output_v2;
/* ABI-2 adds scalar solve-bucket continuation data. No CUDA address crosses
 * this boundary. */
typedef struct beam_depth_control_api_v2 {
    uint32_t abi_version, struct_size;
    void* user_data;
    int32_t (*decide)(void*, const beam_depth_control_input_v2*, beam_depth_control_output_v2*);
    int32_t (*error)(void*, char*, uint64_t capacity, uint64_t* required);
} beam_depth_control_api_v2;

BEAM_API uint32_t beam_native_abi_version(void);
BEAM_API beam_result beam_native_layout(beam_layout_v1*);
BEAM_API beam_result beam_context_create(uint32_t abi_version, beam_context** out);
BEAM_API beam_result beam_context_set_history(beam_context*, const beam_history_api_v1*);
/* Optional ABI-1 extension. Must be set before prepare; native copies the
 * original two-checkpoint table. */
BEAM_API beam_result beam_context_set_depth_control(beam_context*, const beam_depth_control_api_v1*);
/* Optional ABI-2 extension for Rust continuation and solve-bucket policy. */
BEAM_API beam_result beam_context_set_depth_control_v2(beam_context*, const beam_depth_control_api_v2*);
/* Optional ABI-1 extension. Exactly 64 lowercase hexadecimal bytes are copied
 * during this call. Must be set while the context is still CREATED. */
BEAM_API beam_result beam_context_set_graph_hash(beam_context*, const char* hash, uint64_t length);
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
/* Blocking CLI/probe entry, with the same host-history lifetime as a context. */
BEAM_API int beam_native_main_with_history(int argc, const char* const* argv,
                                          const beam_history_api_v1*);
/* Optional ABI-1 entry point for a Rust-owned depth continuation policy. */
BEAM_API int beam_native_main_with_history_and_depth_control(
    int argc, const char* const* argv, const beam_history_api_v1*,
    const beam_depth_control_api_v1*);
/* ABI-2 blocking entry, with coarse scalar depth control through solve-bucket. */
BEAM_API int beam_native_main_with_history_and_depth_control_v2(
    int argc, const char* const* argv, const beam_history_api_v1*,
    const beam_depth_control_api_v2*);
#ifdef __cplusplus
}
#endif
#endif
