#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(BEAM_CNOT_DOMAIN_BUILD)
#    define BEAM_CNOT_DOMAIN_API __declspec(dllexport)
#  else
#    define BEAM_CNOT_DOMAIN_API __declspec(dllimport)
#  endif
#else
#  define BEAM_CNOT_DOMAIN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Version 1: all arrays are borrowed host memory, never CUDA pointers. */
typedef struct beam_cnot_domain beam_cnot_domain;
typedef struct beam_cnot_domain_config {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t device;
    int32_t n;
    int32_t world;
    uint32_t reserved;
    uint64_t parent_capacity;
    uint64_t child_capacity;
    uint64_t closed_capacity;
} beam_cnot_domain_config;

enum beam_cnot_domain_status {
    BEAM_CNOT_DOMAIN_OK = 0,
    BEAM_CNOT_DOMAIN_INVALID = 1,
    BEAM_CNOT_DOMAIN_CAPACITY = 2,
    BEAM_CNOT_DOMAIN_CUDA_FAILURE = 3,
    BEAM_CNOT_DOMAIN_WRONG_THREAD = 4,
    BEAM_CNOT_DOMAIN_INTERNAL = 5
};

BEAM_CNOT_DOMAIN_API uint32_t beam_cnot_domain_abi_version(void);
/* Thread-local message remains valid until the next ABI call on this thread. */
BEAM_CNOT_DOMAIN_API const char* beam_cnot_domain_error(void);
/* Context must only be used and destroyed by its creating thread. */
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_create(const beam_cnot_domain_config*, beam_cnot_domain**);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_destroy(beam_cnot_domain*);
/* Lengths below count int64 elements, not bytes. A zero length accepts NULL. */
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_set_parents(beam_cnot_domain*, const int64_t*, uint64_t words, uint64_t states);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_set_closed(beam_cnot_domain*, const int64_t*, uint64_t words, uint64_t states);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_set_descriptors(beam_cnot_domain*, const int64_t* indices, uint64_t indices_len,
                                     const int64_t* moves, uint64_t moves_len);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_expand(beam_cnot_domain*);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_owners(beam_cnot_domain*);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_first_unseen(beam_cnot_domain*, uint64_t* selected_count);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_download_children(beam_cnot_domain*, int64_t*, uint64_t capacity_words);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_download_owners(beam_cnot_domain*, int64_t*, uint64_t capacity);
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_download_selected(beam_cnot_domain*, int64_t*, uint64_t capacity);

#ifdef __cplusplus
}
#endif
