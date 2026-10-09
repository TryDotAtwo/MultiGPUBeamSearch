// Diagnostic only: identical restored physical input for both operations.
// Baseline is per-physical Stream4, NOT a correct cross-buffer union.
#define BEAM_TEST_UNION_CAPACITY 262144
#define BEAM_TEST_UNION_TEMP_BYTES (256ULL << 20)
#define main bounded_union_fixture_main
#include "stream4_logical_union_cuda_tests.cu"
#undef main
#include <iomanip>

int main() {
    try {
        int devices=0;check(cudaGetDeviceCount(&devices));
        if(devices!=2)throw std::runtime_error("requires owned two-GPU instance");
        std::cout << "scope=isolated_stream4_not_pipeline capacity=" << Fixture::cap << '\n';
        for(int gpu=0;gpu<devices;++gpu) {
            check(cudaSetDevice(gpu));Fixture f;
            cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
            for(unsigned percent:{25U,100U})for(unsigned overlap:{0U,50U}) {
                const unsigned live=Fixture::cap*percent/100;
                std::vector<CandidateMeta> input(2ULL*Fixture::cap);
                for(unsigned side=0;side<2;++side)for(unsigned i=0;i<live;++i) {
                    // Each side unique; a controlled fraction crosses A/B.
                    const std::uint64_t id=side && i>=live*overlap/100 ? live+i : i;
                    input[side*Fixture::cap+i]=CandidateMeta{
                        {id*11400714819323198485ULL,id^0xabcdefULL},unsigned(id%1000),i,side};
                }
#ifdef BEAM_BENCH_OLD_SOURCE
                for(unsigned mode:{0U}) {
#else
                for(unsigned mode:{0U,1U}) {
#endif
                    std::vector<float> timings;
                    unsigned observed=0;
                    for(unsigned repeat=0;repeat<13;++repeat) {
                        const unsigned counts[2]={live,live};
                        check(cudaMemcpyAsync(f.states,input.data(),input.size()*sizeof(CandidateMeta),cudaMemcpyHostToDevice,f.stream));
                        check(cudaMemcpyAsync(f.clean,counts,sizeof(counts),cudaMemcpyHostToDevice,f.stream));
                        check(cudaMemsetAsync(f.dirty,0,2*sizeof(unsigned),f.stream));
                        check(cudaStreamSynchronize(f.stream));
                        check(cudaEventRecord(start,f.stream));
#ifndef BEAM_BENCH_OLD_SOURCE
                        if(mode)stream4_finalize_logical_shard_union_cuda(f.states,f.clean,f.dirty,f.processing,
                            SCORE_MAX_KEY,Fixture::cap,f.key,f.reduced_key,f.value,f.reduced_value,
                            f.score_a,f.score_b,f.count_a,f.count_b,f.flags,f.blocks,f.offsets,f.count,
                            f.hist_a,f.hist_b,f.active,f.temp,Fixture::temp_bytes,f.stream,2*live);
                        else
#endif
                        for(unsigned p=0;p<2;++p)stream4_shard_job_cuda(f.states+p*Fixture::cap,
                            f.clean+p,f.dirty+p,f.processing+p,SCORE_MAX_KEY,Fixture::cap,
                            f.key,f.reduced_key,f.value,f.reduced_value,f.score_a,f.score_b,
                            f.count_a,f.count_b,f.flags,f.blocks,f.offsets,f.count,
                            f.hist_a+p*SCORE_BIN_COUNT,f.hist_b+p*SCORE_BIN_COUNT,f.active+p,
                            f.temp,Fixture::temp_bytes,f.stream);
                        check(cudaEventRecord(stop,f.stream));check(cudaEventSynchronize(stop));
                        check(cudaGetLastError());float ms=0;check(cudaEventElapsedTime(&ms,start,stop));
                        unsigned clean[2];check(cudaMemcpy(clean,f.clean,sizeof(clean),cudaMemcpyDeviceToHost));
                        observed=clean[0]+clean[1];
                        const unsigned want=mode?2*live-live*overlap/100:2*live;
                        if(observed!=want)throw std::runtime_error("output count mismatch");
                        if(repeat>=3)timings.push_back(ms);
                    }
                    std::sort(timings.begin(),timings.end());
                    std::cout << "gpu=" << gpu << " occupancy_pct=" << percent << " overlap_pct=" << overlap
                        << " mode=" << (mode?"logical_union":"old_physical")
                        << " median_ms=" << std::setprecision(8) << (timings[4]+timings[5])/2
                        << " min_ms=" << timings.front() << " max_ms=" << timings.back()
                        << " output_count=" << observed << " repeats=10\n" << std::flush;
                }
            }
            check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
