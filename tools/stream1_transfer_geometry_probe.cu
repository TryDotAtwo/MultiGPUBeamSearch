// Independent public CUDA 1D API reference. No producer or collector headers.
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "../third_party/nlohmann/json.hpp"
static void check(cudaError_t code) {
    if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
static nlohmann::json geometry(cudaGraphNode_t node) {
    cudaMemcpy3DParms p{};check(cudaGraphMemcpyNodeGetParams(node,&p));
    if(p.srcArray || p.dstArray || p.kind!=cudaMemcpyDeviceToDevice ||
       p.extent.height!=1 || p.extent.depth!=1 || p.srcPos.x || p.srcPos.y ||
       p.srcPos.z || p.dstPos.x || p.dstPos.y || p.dstPos.z)
        throw std::runtime_error("reference 1D transfer normalized unexpectedly");
    return {{"bytes",p.extent.width},
        {"copy_geometry",{{"source_array_null",p.srcArray==nullptr},{"destination_array_null",p.dstArray==nullptr},
            {"source_position",{{"x",p.srcPos.x},{"y",p.srcPos.y},{"z",p.srcPos.z}}},
            {"destination_position",{{"x",p.dstPos.x},{"y",p.dstPos.y},{"z",p.dstPos.z}}},
            {"extent",{{"width",p.extent.width},{"height",p.extent.height},{"depth",p.extent.depth}}}}},
        {"source_pitched",{{"pitch",p.srcPtr.pitch},{"xsize",p.srcPtr.xsize},{"ysize",p.srcPtr.ysize}}},
        {"destination_pitched",{{"pitch",p.dstPtr.pitch},{"xsize",p.dstPtr.xsize},{"ysize",p.dstPtr.ysize}}}};
}
static nlohmann::json memset_geometry(cudaGraphNode_t node) {
    cudaMemsetParams p{};check(cudaGraphMemsetNodeGetParams(node,&p));
    return {{"pitch",p.pitch},{"width",p.width},{"height",p.height},
        {"element_size",p.elementSize},{"value",p.value}};
}
int main(int argc,char** argv) {
    try {
        if(argc<3 || argc>5)throw std::runtime_error("device and 1-3 byte counts required");
        const int device=std::stoi(argv[1]);check(cudaSetDevice(device));
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,device));
        std::ostringstream uuid;uuid<<std::hex<<std::setfill('0');
        for(unsigned char byte:prop.uuid.bytes)uuid<<std::setw(2)<<static_cast<unsigned>(byte);
        int runtime=0;check(cudaRuntimeGetVersion(&runtime));
        auto cases=nlohmann::json::array();
        for(int i=2;i<argc;++i) {
            const auto bytes=std::stoull(argv[i]);
            if(!bytes || bytes>8*1024*1024)throw std::runtime_error("reference transfer size bound");
            void *src=nullptr,*dst=nullptr;check(cudaMalloc(&src,bytes));check(cudaMalloc(&dst,bytes));
            cudaGraph_t explicit_graph{};check(cudaGraphCreate(&explicit_graph,0));
            cudaGraphNode_t explicit_node{};
            check(cudaGraphAddMemcpyNode1D(&explicit_node,explicit_graph,nullptr,0,dst,src,bytes,cudaMemcpyDeviceToDevice));
            auto explicit_geometry=geometry(explicit_node);
            cudaStream_t stream{};check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));
            cudaGraph_t capture{};check(cudaStreamEndCapture(stream,&capture));
            std::size_t count=0;cudaGraphNode_t capture_node{};
            check(cudaGraphGetNodes(capture,nullptr,&count));
            if(count!=1)throw std::runtime_error("reference capture not a single node");
            check(cudaGraphGetNodes(capture,&capture_node,&count));
            if(count!=1)throw std::runtime_error("reference capture changed during enumeration");
            cases.push_back({{"bytes",bytes},{"add_node",explicit_geometry},{"capture_async",geometry(capture_node)}});
            check(cudaGraphDestroy(capture));check(cudaGraphDestroy(explicit_graph));
            check(cudaStreamDestroy(stream));check(cudaFree(dst));check(cudaFree(src));
        }
        void* reset=nullptr;check(cudaMalloc(&reset,4096));
        cudaGraph_t reset_graph{};check(cudaGraphCreate(&reset_graph,0));
        cudaMemsetParams params{};params.dst=reset;params.width=4096;params.height=1;params.elementSize=1;
        cudaGraphNode_t reset_node{};check(cudaGraphAddMemsetNode(&reset_node,reset_graph,nullptr,0,&params));
        auto explicit_reset=memset_geometry(reset_node);
        cudaStream_t reset_stream{};check(cudaStreamCreateWithFlags(&reset_stream,cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(reset_stream,cudaStreamCaptureModeThreadLocal));
        check(cudaMemsetAsync(reset,0,4096,reset_stream));
        cudaGraph_t reset_capture{};check(cudaStreamEndCapture(reset_stream,&reset_capture));
        std::size_t reset_count=0;check(cudaGraphGetNodes(reset_capture,nullptr,&reset_count));
        if(reset_count!=1)throw std::runtime_error("reset reference capture not single node");
        check(cudaGraphGetNodes(reset_capture,&reset_node,&reset_count));
        auto reset_reference=nlohmann::json{{"add_node",explicit_reset},{"capture_async",memset_geometry(reset_node)}};
        check(cudaGraphDestroy(reset_capture));check(cudaGraphDestroy(reset_graph));
        check(cudaStreamDestroy(reset_stream));check(cudaFree(reset));
        std::cout<<nlohmann::json{{"schema_version",1},
            {"scope","independent_cuda_1d_transfer_geometry_not_admission"},
            {"production_admitted",false},{"runtime_version",runtime},
            {"device",{{"device",device},{"sm",prop.major*10+prop.minor},{"uuid_hex",uuid.str()}}},
            {"cases",cases},{"memset_reference",reset_reference}}.dump()<<'\n';
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
