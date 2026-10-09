#pragma once
// Isolated service probes at the exact admitted capacities. Synthetic metadata
// is NOT a legal graph frontier or evidence of a completed search depth.
#include "../cuda/static_memory.hpp"
#include "../cuda/stream3.hpp"
#include "../cuda/stream4.hpp"
#include "../third_party/nlohmann/json.hpp"
#include "cuda_check.hpp"
#include <nccl.h>
#include <chrono>
#include <vector>

namespace beam::component_probe {
__device__ inline std::uint64_t mix(std::uint64_t x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
__global__ void metadata(CandidateMeta* out,std::uint32_t n,std::uint32_t rank) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]={Hash128{(i+1ULL)*0x9e3779b97f4a7c15ULL,mix(i+31ULL)>>1},
                   (static_cast<std::uint64_t>(rank)<<48)|i,(i%16)*SCORE_SCALE,0};
}
__global__ void rings(LayoutStreamsView v,std::uint32_t n,std::uint32_t parents) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {v.score_ring[i]=(i%16)*SCORE_SCALE;
        v.hash_ring[i]={(i+1ULL)*0x9e3779b97f4a7c15ULL,mix(i+31ULL)>>1};}
    if(i<(n/(parents*MOVE_COUNT))) {v.parent_base[i]=static_cast<std::uint64_t>(i)*parents;v.count[i]=parents;}
}
__global__ void validate_sorted(const CandidateMeta* rows,std::uint32_t n,std::uint32_t* error) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    if(rows[i].score_key>SCORE_MAX_KEY) atomicAdd(error,1U);
    if(i && (rows[i-1].hash.hi>rows[i].hash.hi ||
       (rows[i-1].hash.hi==rows[i].hash.hi && rows[i-1].hash.lo>=rows[i].hash.lo))) atomicAdd(error,1U);
}
__global__ void validate_transport(const CandidateMeta* rows,std::uint32_t per_peer,
                                  std::uint32_t world,std::uint32_t rank,std::uint32_t* error) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=per_peer*world) return;
    const auto peer=i/per_peer,local=i%per_peer;
    const auto expected=(static_cast<std::uint64_t>(peer)<<48)|(rank*per_peer+local);
    if(rows[i].parent_idx!=expected) atomicAdd(error,1U);
}
template<class Reset,class Run>
nlohmann::json timings(cudaStream_t stream,Reset reset,Run run,bool collective=false) {
    cudaEvent_t begin,end;BEAM_CUDA_CHECK(cudaEventCreate(&begin));BEAM_CUDA_CHECK(cudaEventCreate(&end));
    const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(80);
    unsigned warmups=0;
    do {reset();run();BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));++warmups;}
    while(collective ? warmups<5 : (warmups<3 || std::chrono::steady_clock::now()<until));
    nlohmann::json samples=nlohmann::json::array();
    for(unsigned repeat=0;repeat<5;++repeat) {
        reset();BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        BEAM_CUDA_CHECK(cudaEventRecord(begin,stream));run();BEAM_CUDA_CHECK(cudaEventRecord(end,stream));
        BEAM_CUDA_CHECK(cudaEventSynchronize(end));float ms;
        BEAM_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms*.001);
    }
    BEAM_CUDA_CHECK(cudaEventDestroy(begin));BEAM_CUDA_CHECK(cudaEventDestroy(end));return samples;
}
inline void nccl_check(ncclResult_t result) {
    if(result!=ncclSuccess) throw std::runtime_error(ncclGetErrorString(result));
}
inline nlohmann::json run(const StaticMemoryPlan& p,StaticDeviceMemory& m,ncclComm_t comm) {
    const auto& c=p.config;auto& v=m.streams;const auto q=c.stream3_batch_candidates,capacity=c.shard_capacity_candidates;
    cudaStream_t s3,s4,s5;BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&s3,cudaStreamNonBlocking));
    BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&s4,cudaStreamNonBlocking));
    BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&s5,cudaStreamNonBlocking));
    std::uint32_t* error=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&error,sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(error,0,sizeof(std::uint32_t)));
    auto reset3=[&](){rings<<<(q+255)/256,256,0,s3>>>(v,q,c.b_micro);};
    auto work3=[&](){stream3_pack_threshold_cuda(v.score_ring,v.hash_ring,v.parent_base,v.count,
        v.stream3_key_a,v.stream3_val_a,v.stream3_key_b,v.stream3_val_b,v.unique_key,v.unique_val,
        v.stream3_keep_flags,v.stream3_block_counts,v.stream3_block_offsets,v.unique_count,
        v.stream3_cub_temp,p.stream3_cub_temp_bytes,SCORE_MAX_KEY,c.b_micro,q,s3);};
    const auto t3=timings(s3,reset3,work3);
    std::uint32_t unique=0;BEAM_CUDA_CHECK(cudaMemcpy(&unique,v.unique_count,sizeof(unique),cudaMemcpyDeviceToHost));
    if(unique!=q) throw std::runtime_error("component Stream3 lost synthetic unique candidates");
    const auto jobs=std::min(c.stream4_active_sort_slots,p.storage_shard_count);
    std::vector<cudaStream_t> job_streams(jobs);std::vector<cudaEvent_t> done(jobs);
    cudaEvent_t ready;BEAM_CUDA_CHECK(cudaEventCreate(&ready));
    for(unsigned j=0;j<jobs;++j) {BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&job_streams[j],cudaStreamNonBlocking));BEAM_CUDA_CHECK(cudaEventCreate(&done[j]));}
    auto reset4=[&](){
        const std::uint32_t zero=0,one=1;
        BEAM_CUDA_CHECK(cudaMemsetAsync(v.shard_score_hist_active_index,0,jobs*4,s4));
        BEAM_CUDA_CHECK(cudaMemsetAsync(v.shard_score_hist_a,0,static_cast<std::size_t>(jobs)*SCORE_BIN_COUNT*4,s4));
        BEAM_CUDA_CHECK(cudaMemsetAsync(v.shard_score_hist_b,0,static_cast<std::size_t>(jobs)*SCORE_BIN_COUNT*4,s4));
        for(unsigned j=0;j<jobs;++j) {
            metadata<<<(capacity+255)/256,256,0,s4>>>(v.survivor_shard+static_cast<std::uint64_t>(j)*capacity,capacity,c.local_rank);
            BEAM_CUDA_CHECK(cudaMemcpyAsync(v.clean_count+j,&zero,4,cudaMemcpyHostToDevice,s4));
            BEAM_CUDA_CHECK(cudaMemcpyAsync(v.dirty_count+j,&capacity,4,cudaMemcpyHostToDevice,s4));
            BEAM_CUDA_CHECK(cudaMemcpyAsync(v.processing_flag+j,&one,4,cudaMemcpyHostToDevice,s4));
        }
        BEAM_CUDA_CHECK(cudaStreamSynchronize(s4));
    };
    auto work4=[&](){
        BEAM_CUDA_CHECK(cudaEventRecord(ready,s4));
        for(unsigned j=0;j<jobs;++j) {
            const auto offset=static_cast<std::uint64_t>(j)*capacity;
            const auto blocks=static_cast<std::uint64_t>(j)*((capacity+255)/256);
            BEAM_CUDA_CHECK(cudaStreamWaitEvent(job_streams[j],ready,0));
            stream4_shard_job_cuda(v.survivor_shard+offset,v.clean_count+j,v.dirty_count+j,v.processing_flag+j,
                SCORE_MAX_KEY,capacity,v.stream4_key_a+offset,v.stream4_key_b+offset,
                v.stream4_val_a+offset,v.stream4_val_b+offset,v.stream4_score_key_a+offset,v.stream4_score_key_b+offset,
                v.stream4_score_count_a+offset,v.stream4_score_count_b+offset,v.stream4_keep_flags+offset,
                v.stream4_block_counts+blocks,v.stream4_block_offsets+blocks,v.stream4_count+j,
                v.shard_score_hist_a+static_cast<std::uint64_t>(j)*SCORE_BIN_COUNT,
                v.shard_score_hist_b+static_cast<std::uint64_t>(j)*SCORE_BIN_COUNT,v.shard_score_hist_active_index+j,
                static_cast<char*>(v.stream4_cub_temp)+j*p.stream4_cub_temp_bytes,p.stream4_cub_temp_bytes,job_streams[j]);
            BEAM_CUDA_CHECK(cudaEventRecord(done[j],job_streams[j]));
        }
        for(auto event:done) BEAM_CUDA_CHECK(cudaStreamWaitEvent(s4,event,0));
    };
    const auto t4=timings(s4,reset4,work4);
    for(unsigned j=0;j<jobs;++j) {
        std::uint32_t clean=0;BEAM_CUDA_CHECK(cudaMemcpy(&clean,v.clean_count+j,4,cudaMemcpyDeviceToHost));
        if(clean!=capacity) throw std::runtime_error("component Stream4 lost unique candidates");
        validate_sorted<<<(capacity+255)/256,256,0,s4>>>(v.survivor_shard+static_cast<std::uint64_t>(j)*capacity,capacity,error);
    }
    BEAM_CUDA_CHECK(cudaStreamSynchronize(s4));
    // Final union is sequential and reuses its actual 2*C scratch overlay.
    auto& u=m.final_union;
    auto reset_union=[&](){
        BEAM_CUDA_CHECK(cudaMemsetAsync(v.shard_score_hist_active_index,0,8,s4));
        metadata<<<(2ULL*capacity+255)/256,256,0,s4>>>(v.survivor_shard,2*capacity,c.local_rank);
        const std::uint32_t zero=0;
        for(unsigned j=0;j<2;++j) {
            BEAM_CUDA_CHECK(cudaMemcpyAsync(v.clean_count+j,&capacity,4,cudaMemcpyHostToDevice,s4));
            BEAM_CUDA_CHECK(cudaMemcpyAsync(u.dirty+j,&zero,4,cudaMemcpyHostToDevice,s4));
            BEAM_CUDA_CHECK(cudaMemcpyAsync(u.processing+j,&zero,4,cudaMemcpyHostToDevice,s4));
        }
        BEAM_CUDA_CHECK(cudaStreamSynchronize(s4));
    };
    auto work_union=[&](){stream4_finalize_logical_shard_union_cuda(v.survivor_shard,v.clean_count,
        u.dirty,u.processing,SCORE_MAX_KEY,capacity,u.key_a,u.key_b,u.value_a,u.value_b,
        u.score_a,u.score_b,u.score_count_a,u.score_count_b,u.keep,u.blocks,u.offsets,u.count,
        v.shard_score_hist_a,v.shard_score_hist_b,v.shard_score_hist_active_index,u.cub_temp,p.union_cub_temp_bytes,s4);};
    const auto tu=timings(s4,reset_union,work_union);
    std::uint32_t clean_pair[2];BEAM_CUDA_CHECK(cudaMemcpy(clean_pair,v.clean_count,8,cudaMemcpyDeviceToHost));
    if(static_cast<std::uint64_t>(clean_pair[0])+clean_pair[1]!=2ULL*capacity) throw std::runtime_error("component union lost synthetic unique candidates");
    nlohmann::json transport=nlohmann::json::array();
    if(c.world_size>1) for(unsigned divisor:{2U,1U}) {
        const auto per_peer=std::max(1U,q/(divisor*c.world_size));const auto items=per_peer*c.world_size;
        if(items>p.stream5_send_slot_capacity || items>p.stream5_recv_slot_capacity) throw std::runtime_error("component transport exceeds admitted slot");
        auto reset5=[&](){metadata<<<(items+255)/256,256,0,s5>>>(v.remote_send_buffer,items,c.local_rank);};
        auto work5=[&](){nccl_check(ncclGroupStart());for(unsigned peer=0;peer<c.world_size;++peer) {
            nccl_check(ncclSend(v.remote_send_buffer+peer*per_peer,per_peer*sizeof(CandidateMeta),ncclUint8,peer,comm,s5));
            nccl_check(ncclRecv(v.remote_recv_buffer+peer*per_peer,per_peer*sizeof(CandidateMeta),ncclUint8,peer,comm,s5));
        }nccl_check(ncclGroupEnd());};
        const auto t5=timings(s5,reset5,work5,true);
        validate_transport<<<(items+255)/256,256,0,s5>>>(v.remote_recv_buffer,per_peer,c.world_size,c.local_rank,error);
        BEAM_CUDA_CHECK(cudaStreamSynchronize(s5));transport.push_back({{"items",items},{"seconds",t5}});
    }
    std::uint32_t errors=0;BEAM_CUDA_CHECK(cudaMemcpy(&errors,error,4,cudaMemcpyDeviceToHost));
    if(errors) throw std::runtime_error("component sorted/transport correctness gate failed");
    BEAM_CUDA_CHECK(cudaFree(error));
    for(unsigned j=0;j<jobs;++j) {BEAM_CUDA_CHECK(cudaStreamDestroy(job_streams[j]));BEAM_CUDA_CHECK(cudaEventDestroy(done[j]));}
    BEAM_CUDA_CHECK(cudaEventDestroy(ready));BEAM_CUDA_CHECK(cudaStreamDestroy(s3));
    BEAM_CUDA_CHECK(cudaStreamDestroy(s4));BEAM_CUDA_CHECK(cudaStreamDestroy(s5));
    return {{"component_calibration",true},{"rank",c.local_rank},{"correctness_passed",true},
        {"scope","synthetic_unique_metadata_at_exact_admitted_capacities"},{"outer_candidates",q},
        {"shard_capacity",capacity},{"sort_jobs_concurrent",jobs},{"stream3_seconds",t3},
        {"stream4_group_seconds",t4},{"union_seconds",tu},{"transport",transport},
        {"full_step_verified",false}};
}
} // namespace beam::component_probe
