#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cxxabi.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <typeinfo>

namespace {
template <class T>
void print_type(const char* label) {
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> name(
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
        std::free);
    std::cout << label << '='
              << (status == 0 && name ? name.get() : typeid(T).name()) << '\n';
}
}

int main() {
    using namespace stream1_sm120_nvfp4_cutlass_dsm;
    print_type<Ff1Epilogue>("ff1_epilogue");
    print_type<typename Ff1Epilogue::Params>("ff1_epilogue_params");
    print_type<typename Ff1Epilogue::Arguments>("ff1_epilogue_arguments");
    print_type<Ff1Mainloop>("ff1_mainloop");
    print_type<typename Ff1Mainloop::Params>("ff1_mainloop_params");
    std::cout << "sizes epilogue=" << sizeof(Ff1Epilogue)
              << " epilogue_params=" << sizeof(typename Ff1Epilogue::Params)
              << " mainloop_params=" << sizeof(typename Ff1Mainloop::Params)
              << '\n';
}
