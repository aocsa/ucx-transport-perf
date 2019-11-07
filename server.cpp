#include "blazingdb/uc/API.hpp"
#include "blazingsql/api.hpp"
#include "blazingsql/utils/logger.h"
#include "blazingsql/utils/helper_timer.h"

#include "constexpr_header.h"
#include <atomic>
#include <cstring>
#include <gflags/gflags.h>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime_api.h>

DEFINE_int32(port, 5555, "Server port to listen on");
DEFINE_string(context, "tcp", "UCX Context");
DEFINE_int32(device_id, 0, "Device ID Server");
PerformanceTimeStats stats;
std::atomic<int32_t > counter{1};

void example_callback(Message &request) {
  cudaSetDevice(FLAGS_device_id);
  using namespace blazingdb::uc;
  const void *data = CreateGpuBuffer(BUFFER_LENGTH);
  auto context = CreateUCXContext(FLAGS_context);
  auto agent = context->Agent();
  auto buffer = agent->Register(data, BUFFER_LENGTH);

  StopWatchInterface *timer = nullptr;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInMBs = 0.0f;
  cudaEvent_t start, stop;
  sdkCreateTimer(&timer);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  auto memSize = BUFFER_LENGTH;

  sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));
    auto transport = buffer->Link((const uint8_t *)request.data(), request.size());
    auto future = transport->Get();
  checkCudaErrors(cudaEventRecord(stop, 0));
  sdkStopTimer(&timer);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
  sdkResetTimer(&timer);

  bandwidthInMBs = 2.0f * ((float)(1<<10) * memSize) / (elapsedTimeInMs * (float)(1 << 20));
  stats.Update(bandwidthInMBs, memSize);
  if (counter.load() % 5000 == 0) {
    LOG("iter = {} | context = {} |  Bandwidth(MB/s) = {}", counter.load(), FLAGS_context, stats.bandwidth/5000);
    stats.Reset();
  }
  cudaFree((void*)data);

  sdkDeleteTimer(&timer);
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaEventDestroy(start));

  counter.store(counter.load() + 1);
  request.set("OK");
}

int main(int argc, char **argv) {
  cuInit(0);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int SLEEP_TIME{100000};  //! Milliseconds.
  std::cout << "***Creating a server****" << "tcp://*:" +
  std::to_string(FLAGS_port) << "|" << FLAGS_context << "|" <<  FLAGS_device_id << std::endl;

  // A Server listening on port 5555 for requests from any IP address.
  Server server{"tcp://*:" + std::to_string(FLAGS_port), "[string]", example_callback};

  std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));

  return 0;
}
