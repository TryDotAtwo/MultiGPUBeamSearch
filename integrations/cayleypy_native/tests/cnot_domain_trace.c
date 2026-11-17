/* Linux test-only C ABI forwarding library. It records which device stages the
 * search calls and delegates their behavior to the real CNOT domain library. */
#include "native/cnot_owned_domain.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void *real_library(void) {
    static void *library;
    if (!library) {
        const char *path = getenv("BEAM_TEST_CNOT_REAL_LIBRARY");
        if (!path || !(library = dlopen(path, RTLD_NOW | RTLD_LOCAL))) {
            fprintf(stderr, "test CNOT forwarding library could not load its real target\n");
            abort();
        }
    }
    return library;
}

static void *symbol(const char *name) {
    void *value = dlsym(real_library(), name);
    if (!value) {
        fprintf(stderr, "test CNOT forwarding library is missing %s\n", name);
        abort();
    }
    return value;
}

static void trace(const char *name) {
    const char *path = getenv("BEAM_TEST_CNOT_TRACE");
    FILE *file = path ? fopen(path, "a") : NULL;
    if (!file) abort();
    fprintf(file, "%s\n", name);
    if (fclose(file)) abort();
}

#define FORWARD(type, name, args, values) \
    type name args { return ((__typeof__(&name))symbol(#name)) values; }

FORWARD(uint32_t, beam_cnot_domain_abi_version, (void), ())
FORWARD(const char *, beam_cnot_domain_error, (void), ())
FORWARD(int, beam_cnot_domain_create, (const beam_cnot_domain_config *c, beam_cnot_domain **d), (c, d))
FORWARD(int, beam_cnot_domain_create_device, (const beam_cnot_domain_config *c, beam_cnot_domain **d), (c, d))
FORWARD(int, beam_cnot_domain_destroy, (beam_cnot_domain *d), (d))
FORWARD(int, beam_cnot_domain_set_parents, (beam_cnot_domain *d, const int64_t *p, uint64_t a, uint64_t b), (d, p, a, b))
FORWARD(int, beam_cnot_domain_set_closed, (beam_cnot_domain *d, const int64_t *p, uint64_t a, uint64_t b), (d, p, a, b))
FORWARD(int, beam_cnot_domain_set_descriptors, (beam_cnot_domain *d, const int64_t *i, uint64_t a, const int64_t *m, uint64_t b), (d, i, a, m, b))
FORWARD(int, beam_cnot_domain_expand, (beam_cnot_domain *d), (d))
FORWARD(int, beam_cnot_domain_owners, (beam_cnot_domain *d), (d))
FORWARD(int, beam_cnot_domain_first_unseen, (beam_cnot_domain *d, uint64_t *n), (d, n))
FORWARD(int, beam_cnot_domain_download_children, (beam_cnot_domain *d, int64_t *p, uint64_t n), (d, p, n))
FORWARD(int, beam_cnot_domain_download_owners, (beam_cnot_domain *d, int64_t *p, uint64_t n), (d, p, n))
FORWARD(int, beam_cnot_domain_download_selected, (beam_cnot_domain *d, int64_t *p, uint64_t n), (d, p, n))
FORWARD(uint32_t, beam_cnot_domain_device_abi_version, (void), ())
FORWARD(int, beam_cnot_domain_unseen_scratch_bytes, (beam_cnot_domain *d, uint64_t c, uint64_t n, uint64_t *b), (d, c, n, b))

int beam_cnot_domain_owners_device(beam_cnot_domain *d, const beam_cnot_device_view *s,
                                  uint64_t n, const beam_cnot_device_view *o, void *stream) {
    trace("owners");
    return ((__typeof__(&beam_cnot_domain_owners_device))symbol("beam_cnot_domain_owners_device"))(d, s, n, o, stream);
}
int beam_cnot_domain_children_device(beam_cnot_domain *d, const beam_cnot_device_view *p,
                                    uint64_t pn, const beam_cnot_device_view *i,
                                    const beam_cnot_device_view *m, uint32_t t, uint64_t n,
                                    const beam_cnot_device_view *c, void *stream) {
    trace("children");
    return ((__typeof__(&beam_cnot_domain_children_device))symbol("beam_cnot_domain_children_device"))(d, p, pn, i, m, t, n, c, stream);
}
int beam_cnot_domain_first_unseen_device(beam_cnot_domain *d, const beam_cnot_device_view *c,
                                        uint64_t cn, const beam_cnot_device_view *s, uint64_t sn,
                                        const beam_cnot_unseen_workspace *w, void *stream) {
    trace("unseen");
    return ((__typeof__(&beam_cnot_domain_first_unseen_device))symbol("beam_cnot_domain_first_unseen_device"))(d, c, cn, s, sn, w, stream);
}

