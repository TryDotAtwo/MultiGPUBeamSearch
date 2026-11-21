#define BEAM_CNOT_DOMAIN_BUILD
#include "native/cnot_owned_domain.h"

#ifndef MGBS_TEST_DOMAIN_ABI
#define MGBS_TEST_DOMAIN_ABI 1
#endif
#ifndef MGBS_TEST_DEVICE_ABI
#define MGBS_TEST_DEVICE_ABI 1
#endif

struct beam_cnot_domain { int unused; };
static const char* message = "admission fixture";

uint32_t beam_cnot_domain_abi_version(void) { return MGBS_TEST_DOMAIN_ABI; }
const char* beam_cnot_domain_error(void) { return message; }
int beam_cnot_domain_create(const beam_cnot_domain_config* config, beam_cnot_domain** out) {
    (void)config; (void)out; return 0;
}
int beam_cnot_domain_create_device(const beam_cnot_domain_config* config, beam_cnot_domain** out) {
    (void)config; (void)out; return 0;
}
int beam_cnot_domain_destroy(beam_cnot_domain* domain) { (void)domain; return 0; }
int beam_cnot_domain_set_parents(beam_cnot_domain* d, const int64_t* p, uint64_t w, uint64_t n) {
    (void)d; (void)p; (void)w; (void)n; return 0;
}
int beam_cnot_domain_set_closed(beam_cnot_domain* d, const int64_t* p, uint64_t w, uint64_t n) {
    (void)d; (void)p; (void)w; (void)n; return 0;
}
int beam_cnot_domain_set_descriptors(beam_cnot_domain* d, const int64_t* i, uint64_t il,
                                     const int64_t* m, uint64_t ml) {
    (void)d; (void)i; (void)il; (void)m; (void)ml; return 0;
}
int beam_cnot_domain_expand(beam_cnot_domain* d) { (void)d; return 0; }
int beam_cnot_domain_owners(beam_cnot_domain* d) { (void)d; return 0; }
int beam_cnot_domain_first_unseen(beam_cnot_domain* d, uint64_t* n) { (void)d; (void)n; return 0; }
int beam_cnot_domain_download_children(beam_cnot_domain* d, int64_t* p, uint64_t n) {
    (void)d; (void)p; (void)n; return 0;
}
int beam_cnot_domain_download_owners(beam_cnot_domain* d, int64_t* p, uint64_t n) {
    (void)d; (void)p; (void)n; return 0;
}
int beam_cnot_domain_download_selected(beam_cnot_domain* d, int64_t* p, uint64_t n) {
    (void)d; (void)p; (void)n; return 0;
}

uint32_t beam_cnot_domain_device_abi_version(void) { return MGBS_TEST_DEVICE_ABI; }
int beam_cnot_domain_owners_device(beam_cnot_domain* d, const beam_cnot_device_view* s,
                                  uint64_t n, const beam_cnot_device_view* o, void* stream) {
    (void)d; (void)s; (void)n; (void)o; (void)stream; return 0;
}
int beam_cnot_domain_children_device(beam_cnot_domain* d, const beam_cnot_device_view* p,
                                     uint64_t pn, const beam_cnot_device_view* i,
                                     const beam_cnot_device_view* m, uint32_t u16,
                                     uint64_t cn, const beam_cnot_device_view* c, void* stream) {
    (void)d; (void)p; (void)pn; (void)i; (void)m; (void)u16; (void)cn; (void)c; (void)stream;
    return 0;
}
#ifndef MGBS_TEST_MISSING_UNSEEN
int beam_cnot_domain_unseen_scratch_bytes(beam_cnot_domain* d, uint64_t c, uint64_t n, uint64_t* b) {
    (void)d; (void)c; (void)n; (void)b; return 0;
}
int beam_cnot_domain_first_unseen_device(beam_cnot_domain* d, const beam_cnot_device_view* c,
                                        uint64_t cn, const beam_cnot_device_view* s, uint64_t sn,
                                        const beam_cnot_unseen_workspace* w, void* stream) {
    (void)d; (void)c; (void)cn; (void)s; (void)sn; (void)w; (void)stream; return 0;
}
#endif
