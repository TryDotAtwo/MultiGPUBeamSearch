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

/* Base version 1 uses borrowed host memory. The optional device extension
 * below has a separate version and explicitly borrows CUDA addresses. */
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

/* Optional extension ABI 1. A view describes borrowed device allocation
 * memory; elements count lanes of the type specified by the operation, and
 * reserved must be zero. The caller guarantees physical allocation bounds,
 * accessibility from the domain CUDA context, and lifetime through the call. */
typedef struct beam_cnot_device_view {
    void* data;
    uint64_t elements;
    int32_t device;
    uint32_t reserved;
} beam_cnot_device_view;

BEAM_CNOT_DOMAIN_API uint32_t beam_cnot_domain_device_abi_version(void);
/* states and owners contain int64 lanes. stream must be a valid CUDA stream of
 * this context (NULL means the default stream). The caller makes inputs ready
 * on that stream before the call and
 * must not supply overlapping input/output regions. This operation completes
 * the supplied stream before returning; no host data transfer is performed. */
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_owners_device(
    beam_cnot_domain*, const beam_cnot_device_view* states, uint64_t state_count,
    const beam_cnot_device_view* owners, void* stream);

/* Optional additive device operations. The device-only context retains its
 * configured capacities but does not allocate host-ABI buffers or CUB scratch;
 * host-buffer operations return BEAM_CNOT_DOMAIN_INVALID on that context. */
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_create_device(
    const beam_cnot_domain_config*, beam_cnot_domain**);

/* Parents, indices and children contain int64 lanes. Moves contain int64
 * lanes when moves_are_u16=0 and uint16 lanes when it is 1. Each parent index
 * and physical move must be valid. Output must not overlap an input. The same
 * stream, lifetime, ordering and synchronous-return rules as owners_device
 * apply; invalid CUDA descriptor values retain the kernel's device-error
 * behavior. No host data transfer is performed. */
BEAM_CNOT_DOMAIN_API int beam_cnot_domain_children_device(
    beam_cnot_domain*, const beam_cnot_device_view* parents, uint64_t parent_count,
    const beam_cnot_device_view* indices, const beam_cnot_device_view* moves,
    uint32_t moves_are_u16, uint64_t child_count, const beam_cnot_device_view* children,
    void* stream);

#ifdef __cplusplus
}
#endif
