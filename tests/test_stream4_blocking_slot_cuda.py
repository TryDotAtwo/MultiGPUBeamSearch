"""Real CUDA event skew against the dispatcher's actual slot-acquire lambdas.

Break caught: blocking on the oldest unfinished slot when a later slot completes
first. This checks event waiting/queue reuse, not full shard/NCCL callbacks.
"""
import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENABLED = os.environ.get('BEAM_TEST_CUDA_EVENT_SKEW') == '1'


@pytest.mark.skipif(not ENABLED or os.name != 'posix' or not shutil.which('nvcc'),
                    reason='explicit authorized CUDA event-skew run required')
@pytest.mark.parametrize('device', [0, 1])
def test_blocking_acquire_reuses_finished_later_slot_before_slow_oldest(device, tmp_path):
    source = (ROOT / 'cuda/dispatcher.cu').read_text()
    start = source.index('    const auto release_completed_stream4_slots_nonblocking =')
    end = source.index('    const auto read_current_threshold_host =', start)
    actual = source[start:end]
    program = r'''
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
struct NvtxRange { explicit NvtxRange(const char*) {} };
void check_cuda(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(operation);
}
__global__ void delayed_event(unsigned long long cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {}
}
struct Resources {
    cudaStream_t slow{}, fast{};
    cudaEvent_t done[2]{};
    ~Resources() {
        if (slow) cudaStreamSynchronize(slow);
        if (fast) cudaStreamSynchronize(fast);
        for (auto event : done) if (event) cudaEventDestroy(event);
        if (slow) cudaStreamDestroy(slow);
        if (fast) cudaStreamDestroy(fast);
    }
};
int main(int argc, char** argv) {
    try {
        check_cuda(cudaSetDevice(std::atoi(argv[1])), "set device");
        cudaDeviceProp property{};
        check_cuda(cudaGetDeviceProperties(&property, std::atoi(argv[1])), "device properties");
        delayed_event<<<1, 1>>>(0);
        check_cuda(cudaDeviceSynchronize(), "warmup");
        Resources resource;
        check_cuda(cudaStreamCreateWithFlags(&resource.slow, cudaStreamNonBlocking), "slow stream");
        check_cuda(cudaStreamCreateWithFlags(&resource.fast, cudaStreamNonBlocking), "fast stream");
        for (auto& event : resource.done)
            check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event");
        // Warm module/context first; 20ms versus 300ms is diagnostic skew only.
        delayed_event<<<1, 1, 0, resource.slow>>>(static_cast<unsigned long long>(property.clockRate) * 300);
        check_cuda(cudaEventRecord(resource.done[0], resource.slow), "slow record");
        delayed_event<<<1, 1, 0, resource.fast>>>(static_cast<unsigned long long>(property.clockRate) * 20);
        check_cuda(cudaEventRecord(resource.done[1], resource.fast), "fast record");
        check_cuda(cudaGetLastError(), "delay launch");
        if (cudaEventQuery(resource.done[0]) != cudaErrorNotReady ||
            cudaEventQuery(resource.done[1]) != cudaErrorNotReady)
            throw std::runtime_error("fixture did not begin with two unfinished events");
        struct { cudaEvent_t stream4_slot_done[2]; } streams{resource.done[0], resource.done[1]};
        struct { struct { unsigned stream4_active_sort_slots = 2; } config; } plan;
        std::deque<unsigned> stream4_busy_slots{0, 1}, stream4_free_slots;
        std::vector<unsigned> released;
        auto mark_stream4_slot_complete = [&](unsigned slot) {
            released.push_back(slot);
            stream4_free_slots.push_back(slot);
        };
''' + actual + r'''
        const auto started = std::chrono::steady_clock::now();
        const auto slot = acquire_stream4_slot_blocking();
        const auto milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        const auto oldest = cudaEventQuery(resource.done[0]);
        std::cout << "device=" << argv[1] << " acquired=" << slot
                  << " acquire_ms=" << milliseconds << " oldest_not_ready="
                  << (oldest == cudaErrorNotReady) << '\n';
        if (slot != 1 || oldest != cudaErrorNotReady)
            throw std::runtime_error("blocking acquire waited for slow oldest instead of ready later slot");
        if (released != std::vector<unsigned>{1} || stream4_busy_slots != std::deque<unsigned>{0})
            throw std::runtime_error("queue ownership or exactly-once completion changed");
        wait_all_stream4_slots();
        if (released != std::vector<unsigned>({1, 0}) || !stream4_busy_slots.empty())
            throw std::runtime_error("drain failed after out-of-order slot reuse");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
'''
    fixture = tmp_path / 'blocking_slot_probe.cu'
    binary = tmp_path / 'blocking_slot_probe'
    fixture.write_text(program)
    build = subprocess.run(['nvcc', '-std=c++17', '-arch=sm_86', str(fixture), '-o', str(binary)],
                           capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(binary), str(device)], capture_output=True, text=True, timeout=15)
    print(result.stdout, end='')
    assert result.returncode == 0, result.stdout + result.stderr
