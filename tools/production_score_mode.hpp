#pragma once
#include "production_score_request.hpp"
#include "stream1_weight_io.hpp"
#include "stream1_graph_inventory.hpp"
#include "graph_transfer_inventory.hpp"
#include "graph_pointer_inventory.hpp"
#include "../cuda/stream1_execution_contract.hpp"
#include "../cuda/stream1_transformer_chunk_launch.hpp"
#include "../cuda/stream1_transformer_policy_snapshot.hpp"
#include "../cuda/stream1_kernel_description.hpp"
#include "../cuda/stream1_layout_inventory.hpp"
#include "../cuda/stream1_execution_shape.hpp"
#include "../third_party/picosha2/picosha2.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <array>
#include <cstring>
#include <sstream>
namespace beam::score_mode {
struct Resources {
    stream1_weights::DeviceWeights weights;
    stream1_weights::ScratchAllocation scratch;
    State128* frontier=nullptr;
    std::uint64_t* base=nullptr;
    std::uint32_t *count=nullptr,*job=nullptr,*keys=nullptr;
    half* raw=nullptr;
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    ~Resources() {
        if(exec)cudaGraphExecDestroy(exec);
        if(graph)cudaGraphDestroy(graph);
        if(stream)cudaStreamDestroy(stream);
        cudaFree(raw);cudaFree(keys);cudaFree(job);cudaFree(count);cudaFree(base);cudaFree(frontier);
        stream1_weights::free_stream1_scratch(scratch);
        stream1_weights::free_weights(weights);
    }
    Resources()=default;
    Resources(const Resources&)=delete;
    Resources& operator=(const Resources&)=delete;
};
inline float two_byte_float(std::uint16_t bits,std::uint32_t dtype) {
    if(dtype==STREAM1_DTYPE_FP16) {half value;std::memcpy(&value,&bits,2);return __half2float(value);}
    if(dtype==STREAM1_DTYPE_BF16) {__nv_bfloat16 value;std::memcpy(&value,&bits,2);return __bfloat162float(value);}
    throw std::invalid_argument("unsupported raw-score dtype");
}
inline void publish_json(const std::filesystem::path& dir,const char* name,const nlohmann::json& value) {
    const auto target=dir/name;
    const auto temporary=dir/(std::string(name)+".tmp");
    std::ofstream file(temporary,std::ios::binary);
    if(!file)throw std::runtime_error("cannot open production-score output");
    file<<value.dump()<<'\n';file.close();
    if(!file)throw std::runtime_error("cannot finish production-score output");
    std::filesystem::rename(temporary,target);
}
inline nlohmann::json resolved_host_choices(const Stream1ExecutionContract& contract) {
    const auto& launch=contract.launch_choices();
    nlohmann::json flags=nlohmann::json::object(),families=nlohmann::json::array();
    for(std::size_t i=0;i<stream1_launch_flag_keys.size();++i)
        flags[stream1_launch_flag_keys[i]]=launch.flags[i];
    constexpr std::array<const char*,4> names={"qkv","attention_out","ff1","ff2"};
    for(std::size_t i=0;i<names.size();++i) {
        const auto& choice=contract.gemm_choices().families[i];
        families.push_back({{"family",names[i]},{"policy_code",static_cast<int>(choice.policy)},
            {"stage_code",static_cast<int>(choice.stage)},{"swizzle_code",static_cast<int>(choice.swizzle)},
            {"epilogue_code",static_cast<int>(choice.epilogue)}});
    }
    // Codes are pinned to the source enum definitions, not a portable kernel ABI.
    return {{"schema_version",1},{"scope","resolved_host_choices_not_kernel_admission"},
        {"production_admitted",false},{"kernel_coverage_complete",false},
        {"sm",contract.sm()},{"outer_microbatch",contract.outer_microbatch()},
        {"transformer_microbatch",contract.transformer_microbatch()},{"lanes",contract.lanes()},
        {"graph_executor",contract.graph_executor()},{"fp16",contract.gemm_choices().fp16},
        {"gemm_families",std::move(families)},
        {"launch",{{"attention_tile_code",static_cast<int>(launch.attention_tile)},
            {"attention_max_k_code",static_cast<int>(launch.attention_max_k)},
            {"cls_attention_code",static_cast<int>(launch.cls_attention)},
            {"layernorm_rows_code",static_cast<int>(launch.layernorm_rows)},
            {"layernorm_copy_grid",launch.layernorm_resident.copy_grid},
            {"layernorm_bias_round_grid",launch.layernorm_resident.bias_round_grid},
            {"hopper_ff1_epilogue_128x64",launch.hopper_ff1_epilogue_128x64},
            {"stage_profile_skip_calls",launch.stage_profile_skip_calls},{"flags",std::move(flags)}}}};
}
inline int run_production_score_mode(int argc,char** argv) {
    const auto r=parse_request(argc,argv);
    const auto policies=capture_stream1_transformer_launch_policies();
    const auto manifest=bounded_text(r.weight_dir/"manifest.json");
    const auto host=stream1_weights::load_stream1_weights(r.weight_dir);
    if(host.model.backend!=STREAM1_BACKEND_PIECE_TRANSFORMER || host.model.state_len!=96 ||
       host.model.output_dim!=24)
        throw std::invalid_argument("production-score mode requires Cube4 piece_transformer");
    if(host.verified_transformer_manifest!=manifest ||
       bounded_text(r.weight_dir/"manifest.json")!=manifest)
        throw std::runtime_error("weight manifest changed during load");
    const auto local_rank=profile_u32("LOCAL_RANK",0);
    if(local_rank>static_cast<unsigned>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid production-score device ordinal");
    BEAM_CUDA_CHECK(cudaSetDevice(static_cast<int>(local_rank)));
    cudaDeviceProp props{};BEAM_CUDA_CHECK(cudaGetDeviceProperties(&props,static_cast<int>(local_rank)));
    std::size_t free=0,total=0;BEAM_CUDA_CHECK(cudaMemGetInfo(&free,&total));
    const auto required=stream1_weights::stream1_scratch_bytes(host.model,r.inner,r.lanes)+
        stream1_weights::total_device_weight_bytes(host)+
        (static_cast<std::uint64_t>(r.outer)+1)*sizeof(State128)+
        static_cast<std::uint64_t>(r.outer)*24*6+(16ULL<<20);
    if(required>free || free-required<(256ULL<<20))
        throw std::invalid_argument("production-score profile does not fit device headroom");
    Resources resource;
    resource.weights=stream1_weights::upload_weights(host);
    auto holder=stream1_weights::transformer_network_view(resource.weights.transformer,host.model);
    const auto ln_occupancy=observe_stream1_layernorm_occupancy_cuda();
    const Stream1ExecutionContract contract(holder.view,props.major*10+props.minor,
        stream1_transformer_has_hopper_launch_path(),r.outer,r.inner,r.lanes,policies,ln_occupancy,
        r.executor==Stream1Executor::NativeGraph);
    const auto& network=contract.network();
    resource.scratch=stream1_weights::alloc_stream1_scratch(host.model,r.inner,r.lanes);
    std::vector<State128> frontier(static_cast<std::size_t>(r.outer)+1);
    for(std::size_t i=0;i<frontier.size();++i)frontier[i]=r.states[i%r.states.size()];
    BEAM_CUDA_CHECK(cudaMalloc(&resource.frontier,frontier.size()*sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMemcpy(resource.frontier,frontier.data(),frontier.size()*sizeof(State128),cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMalloc(&resource.base,sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&resource.count,sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&resource.job,sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(resource.job,0,sizeof(std::uint32_t)));
    const std::size_t raw_bytes=static_cast<std::size_t>(r.outer)*24*2;
    BEAM_CUDA_CHECK(cudaMalloc(&resource.raw,raw_bytes));
    BEAM_CUDA_CHECK(cudaMalloc(&resource.keys,static_cast<std::size_t>(r.outer)*24*4));
    BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&resource.stream,cudaStreamNonBlocking));
    std::array<std::uint16_t,24> bias{};
    BEAM_CUDA_CHECK(cudaMemcpy(bias.data(),network.output_bias,48,cudaMemcpyDeviceToHost));
    struct Job {const char* name;std::uint64_t base;std::uint32_t count;};
    const std::array<Job,4> cases{{{"full",0,r.outer},{"partial",1,r.outer-1},
        {"singleton",r.outer,1},{"zero",0,0}}};
    nlohmann::json jobs=nlohmann::json::array();
    Stream1KernelObservation observed(4096);
    Stream1KernelObservationScope observation_scope(observed);
    const bool graph_mode=r.executor==Stream1Executor::NativeGraph;
    GraphKernelInventory graph_inventory;
    GraphTransferInventory graph_transfers;
    GraphPointerInventory graph_pointers;
    for(std::uint32_t lane=0;lane<r.lanes;++lane) {
        const auto scratch=stream1_weights::transformer_scratch_view(resource.scratch,host.model,r.inner,lane);
        const auto issue=[&] {
            BEAM_CUDA_CHECK(cudaMemsetAsync(resource.raw,0,raw_bytes,resource.stream));
            BEAM_CUDA_CHECK(cudaMemsetAsync(scratch.numeric_error,0,sizeof(std::uint32_t),resource.stream));
            launch_stream1_transformer_chunks_cuda(resource.frontier,resource.base,resource.count,
                graph_mode?resource.job:nullptr,graph_mode,network,scratch,resource.keys,
                r.outer,r.inner,resource.stream,[&](std::uint32_t offset,std::uint32_t count) {
                    // Capture-safe device copy preserves each chunk before the
                    // next launch reuses its logits scratch. No hot-path change.
                    BEAM_CUDA_CHECK(cudaMemcpyAsync(resource.raw+static_cast<std::size_t>(offset)*24,
                        scratch.logits,static_cast<std::size_t>(count)*24*2,
                        cudaMemcpyDeviceToDevice,resource.stream));
                },&contract.policies(),&contract);
        };
        if(graph_mode) {
            const std::uint64_t base=0;
            BEAM_CUDA_CHECK(cudaMemcpy(resource.base,&base,sizeof(base),cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaMemcpy(resource.count,&r.outer,sizeof(r.outer),cudaMemcpyHostToDevice));
            issue();BEAM_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
            BEAM_CUDA_CHECK(cudaStreamBeginCapture(resource.stream,cudaStreamCaptureModeThreadLocal));
            issue();BEAM_CUDA_CHECK(cudaStreamEndCapture(resource.stream,&resource.graph));
            BEAM_CUDA_CHECK(cudaGraphInstantiate(&resource.exec,resource.graph,nullptr,nullptr,0));
            // Register actual per-lane allocation slices, not guessed address
            // tags. This diagnostic collection does not attest kernel extents.
            const auto bytes=stream1_weights::transformer_scratch_byte_plan(host.model,
                stream1_inference_rows(r.inner,host.model));
            GraphPointerRegistry pointers;
            pointers.add("fast_slot_projected",network.fast_slot_projected,host.transformer.fast_slot_projected.size());
            pointers.add("fast_piece_static",network.fast_piece_static,host.transformer.fast_piece_static.size());
            pointers.add("cls_token",network.cls_token,host.transformer.cls_token.size());
            pointers.add("piece_positions",network.piece_positions,host.transformer.piece_positions.size());
            pointers.add("piece_mask",network.piece_mask,host.transformer.piece_mask.size());
            pointers.add("frontier_states",resource.frontier,frontier.size()*sizeof(State128));
            pointers.add("parent_base",resource.base,sizeof(std::uint64_t));
            pointers.add("lane_tokens",scratch.tokens,bytes.token_bytes);
            pointers.add("lane_qkv",scratch.qkv,bytes.qkv_bytes);
            pointers.add("lane_attention",scratch.attention_scores_probs,bytes.attention_bytes);
            pointers.add("lane_context",scratch.attention_context,bytes.context_bytes);
            pointers.add("lane_ff_hidden",scratch.ff_hidden,bytes.ff_hidden_bytes);
            pointers.add("lane_logits",scratch.logits,bytes.logits_bytes);
            pointers.add("output_bias",network.output_bias,static_cast<std::size_t>(host.model.output_dim)*sizeof(half));
            pointers.add("output_weight",network.output_weight,host.transformer.output_weight.size());
            pointers.add("active_count",resource.count,sizeof(std::uint32_t));
            pointers.add("graph_job_index",resource.job,sizeof(std::uint32_t));
            pointers.add("score_keys",resource.keys,static_cast<std::size_t>(r.outer)*24*4);
            pointers.add("numeric_error",scratch.numeric_error,sizeof(std::uint32_t));
            const auto norm_bytes=static_cast<std::size_t>(host.model.d_model)*sizeof(half);
            pointers.add("input_ln_gamma",network.input_ln_gamma,norm_bytes);
            pointers.add("input_ln_beta",network.input_ln_beta,norm_bytes);
            pointers.add("output_ln_gamma",network.output_ln_gamma,norm_bytes);
            pointers.add("output_ln_beta",network.output_ln_beta,norm_bytes);
            for(std::uint32_t block=0;block<host.model.transformer_layers;++block) {
                const auto prefix="block"+std::to_string(block)+"_";
                const auto& b=network.blocks[block];
                const auto& hb=host.transformer.blocks[block];
                pointers.add(prefix+"attn_qkv_weight",b.attn_qkv_weight,hb.attn_qkv_weight.size());
                pointers.add(prefix+"attn_qkv_bias",b.attn_qkv_bias,hb.attn_qkv_bias.size());
                pointers.add(prefix+"attn_out_weight",b.attn_out_weight,hb.attn_out_weight.size());
                pointers.add(prefix+"ff1_weight",b.ff1_weight,hb.ff1_weight.size());
                pointers.add(prefix+"ff1_bias",b.ff1_bias,hb.ff1_bias.size());
                pointers.add(prefix+"ff2_weight",b.ff2_weight,hb.ff2_weight.size());
                pointers.add(prefix+"ln1_gamma",b.ln1_gamma,norm_bytes);
                pointers.add(prefix+"ln1_beta",b.ln1_beta,norm_bytes);
                pointers.add(prefix+"ln2_gamma",b.ln2_gamma,norm_bytes);
                pointers.add(prefix+"ln2_beta",b.ln2_beta,norm_bytes);
                pointers.add(prefix+"attn_out_bias",b.attn_out_bias,norm_bytes);
                pointers.add(prefix+"ff2_bias",b.ff2_bias,norm_bytes);
            }
            graph_inventory.observe(resource.graph,lane,&pointers);
            graph_pointers.observe(resource.graph,lane,pointers);
            graph_transfers.observe(resource.graph,lane,resource.raw,raw_bytes,
                scratch.logits,static_cast<std::size_t>(r.inner)*24*2,
                scratch.numeric_error,sizeof(std::uint32_t));
        }
        for(const auto& job:cases) {
            BEAM_CUDA_CHECK(cudaMemcpyAsync(resource.base,&job.base,sizeof(job.base),cudaMemcpyHostToDevice,resource.stream));
            BEAM_CUDA_CHECK(cudaMemcpyAsync(resource.count,&job.count,sizeof(job.count),cudaMemcpyHostToDevice,resource.stream));
            if(graph_mode)BEAM_CUDA_CHECK(cudaGraphLaunch(resource.exec,resource.stream));
            else issue();
            BEAM_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
            std::uint32_t numeric_error=0;
            BEAM_CUDA_CHECK(cudaMemcpy(&numeric_error,scratch.numeric_error,sizeof(numeric_error),cudaMemcpyDeviceToHost));
            if(numeric_error)throw std::runtime_error("production-score job raised numeric error");
            std::vector<std::uint16_t> raw(static_cast<std::size_t>(job.count)*24);
            if(!raw.empty())BEAM_CUDA_CHECK(cudaMemcpy(raw.data(),resource.raw,raw.size()*2,cudaMemcpyDeviceToHost));
            nlohmann::json rows=nlohmann::json::array(),mapping=nlohmann::json::array();
            for(std::uint32_t row=0;row<job.count;++row) {
                mapping.push_back((job.base+row)%r.states.size());
                nlohmann::json values=nlohmann::json::array();
                for(unsigned move=0;move<24;++move) {
                    const float value=two_byte_float(raw[static_cast<std::size_t>(row)*24+move],host.model.dtype)+
                        two_byte_float(bias[move],host.model.dtype);
                    if(!std::isfinite(value))throw std::runtime_error("nonfinite production raw score");
                    values.push_back(value);
                }
                rows.push_back(std::move(values));
            }
            jobs.push_back({{"lane",lane},{"case",job.name},{"parent_base",job.base},
                {"active_rows",job.count},{"reference_rows",std::move(mapping)},
                {"raw_scores",std::move(rows)}});
        }
        if(resource.exec) {BEAM_CUDA_CHECK(cudaGraphExecDestroy(resource.exec));resource.exec=nullptr;}
        if(resource.graph) {BEAM_CUDA_CHECK(cudaGraphDestroy(resource.graph));resource.graph=nullptr;}
    }
    if(bounded_text(r.reference_dir/"reference.json")!=r.reference_text ||
       bounded_text(r.weight_dir/"manifest.json")!=manifest)
        throw std::runtime_error("production-score input changed during invocation");
    std::ostringstream device,loaded;
    write_stream1_device_identity(device,static_cast<int>(local_rank),props.major*10+props.minor,props.uuid.bytes);
    write_stream1_loaded_view_inventory(loaded,network);
    nlohmann::json requested=nlohmann::json::object(),kernels=nlohmann::json::array();
    for(const auto& entry:policies.values())requested[entry.first]=entry.second?nlohmann::json(*entry.second):nlohmann::json(nullptr);
    for(const auto& d:observed.records())kernels.push_back({{"family",d.family},{"implementation",d.implementation},
        {"dtype",d.dtype},{"epilogue",d.epilogue},{"architecture",d.architecture},
        {"rows",d.rows},{"input_cols",d.input_cols},{"output_cols",d.output_cols},
        {"tile",{d.tile_m,d.tile_n,d.tile_k}},{"warp",{d.warp_m,d.warp_n,d.warp_k}},
        {"instruction",{d.instruction_m,d.instruction_n,d.instruction_k}},
        {"stages",d.stages},{"swizzle",d.swizzle},{"packed_weight",d.packed_weight}});
    nlohmann::json execution={{"schema_version",1},{"scope","actual_runner_raw_score_candidate_not_admission"},
        {"production_quality_accepted",false},{"resolved_execution_contract_complete",false},
        {"loaded_tensor_content_attestation_complete",true},
        {"loaded_tensor_bindings",nlohmann::json::parse(host.verified_transformer_manifest).at("tensor_files")},
        {"device_upload_bytes_verified",true},
        {"communicator_initialized",false},{"search_buffers_allocated",false},
        {"executor",graph_mode?"native_cuda_graph":"native_eager"},{"outer_microbatch",r.outer},
        {"transformer_microbatch",r.inner},{"lanes",r.lanes},{"lanes_exercised_serially",true},
        {"reference_sha256",picosha2::hash256_hex_string(r.reference_text)},
        {"manifest_sha256",picosha2::hash256_hex_string(manifest)},
        {"distinct_reference_rows",r.states.size()},{"device",nlohmann::json::parse(device.str())},
        {"loaded_view",nlohmann::json::parse(loaded.str())},{"requested_launch_policies",std::move(requested)},
        {"host_kernel_observations",std::move(kernels)},{"kernel_coverage_complete",false},
        {"cohort_generalization_proven",false},{"scores_before_quantization",true}};
    if(!std::filesystem::create_directories(r.output_dir))
        throw std::runtime_error("production-score output appeared during invocation");
    const nlohmann::json raw_payload={{"schema_version",1},{"jobs",std::move(jobs)}};
    publish_json(r.output_dir,"raw_scores.json",raw_payload);
    publish_json(r.output_dir,"execution.json",execution);
    if(graph_mode) {
        auto inventory=graph_inventory.payload();
        inventory["device"]=execution.at("device");
        inventory["execution_sha256"]=picosha2::hash256_hex_string(execution.dump()+"\n");
        publish_json(r.output_dir,"graph_kernel_inventory.json",inventory);
        auto parameter_layout=graph_inventory.parameter_layout_payload();
        parameter_layout["device"]=execution.at("device");
        parameter_layout["execution_sha256"]=inventory.at("execution_sha256");
        parameter_layout["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_parameter_layout.json",parameter_layout);
        auto scalar_values=graph_inventory.u32_values_payload();
        scalar_values["device"]=execution.at("device");
        scalar_values["execution_sha256"]=inventory.at("execution_sha256");
        scalar_values["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_u32_values.json",scalar_values);
        auto dims_values=graph_inventory.dims_values_payload();
        dims_values["device"]=execution.at("device");
        dims_values["execution_sha256"]=inventory.at("execution_sha256");
        dims_values["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_embedded_dims.json",dims_values);
        auto network_members=graph_inventory.network_members_payload();
        network_members["device"]=execution.at("device");
        network_members["execution_sha256"]=inventory.at("execution_sha256");
        network_members["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_network_members.json",network_members);
        auto attention_members=graph_inventory.attention_members_payload();
        attention_members["device"]=execution.at("device");
        attention_members["execution_sha256"]=inventory.at("execution_sha256");
        attention_members["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_attention_members.json",attention_members);
        auto gemm_members=graph_inventory.gemm_members_payload();
        gemm_members["device"]=execution.at("device");
        gemm_members["execution_sha256"]=inventory.at("execution_sha256");
        gemm_members["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_gemm_members.json",gemm_members);
        auto pointer_values=graph_pointers.payload();
        pointer_values["device"]=execution.at("device");
        pointer_values["execution_sha256"]=inventory.at("execution_sha256");
        pointer_values["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_pointer_roles.json",pointer_values);
        auto transfers=graph_transfers.payload();
        transfers["device"]=execution.at("device");
        transfers["execution_sha256"]=inventory.at("execution_sha256");
        transfers["inventory_sha256"]=picosha2::hash256_hex_string(inventory.dump()+"\n");
        publish_json(r.output_dir,"graph_transfers.json",transfers);
    }
    auto resolved=resolved_host_choices(contract);
    // These launch-site observations describe fixed specialized routes that
    // bypass the ordinary family resolver. They do not prove graph replay or
    // compiled kernel coverage, and must not promote this candidate to admission.
    resolved["specialized_gemm_observations"]=nlohmann::json::array();
    for(const auto& description:execution.at("host_kernel_observations"))
        if(description.at("family")=="linear_bias_strided")
            resolved["specialized_gemm_observations"].push_back(description);
    for(const auto* binding:{"device","manifest_sha256","reference_sha256",
                            "loaded_tensor_bindings","requested_launch_policies"})
        resolved[binding]=execution.at(binding);
    resolved["raw_scores_sha256"]=picosha2::hash256_hex_string(raw_payload.dump()+"\n");
    publish_json(r.output_dir,"resolved_execution.json",resolved);
    return 0;
}
}
