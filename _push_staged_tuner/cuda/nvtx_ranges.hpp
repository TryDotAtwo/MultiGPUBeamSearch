#pragma once

#include <nvtx3/nvToolsExt.h>

namespace beam {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~NvtxRange() {
        nvtxRangePop();
    }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

} // namespace beam
