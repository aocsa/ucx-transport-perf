#include "blazingdb/uc/API.hpp"
#include "blazingsql/api.hpp"
#include "blazingsql/utils/logger.h"

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

//  std::cout << "buffer_size:" << std::dec << context->serializedRecordSize() << std::endl;
//  std::cout << "buffer_descriptors:" << std::endl;
//  size_t checksum = 0;
//  for (size_t i = 0; i < context->serializedRecordSize(); i++)
//  {
//    std::cout << +(unsigned char)request.data()[i] << ", ";
//    checksum += (unsigned char)request.data()[i];
//  }
//  std::cout << "checksum:" << checksum << std::endl;
  StopWatch timer;
  timer.Start();
  auto transport = buffer->Link((const uint8_t *)request.data(), request.size());
  auto future = transport->Get();
//  Print("peer", data, BUFFER_LENGTH);

  uint64_t elapsed_nanos = timer.Stop();
  double time_elapsed =
      static_cast<double>(elapsed_nanos) / static_cast<double>(1000000000);
  stats.Update(time_elapsed, BUFFER_LENGTH);

  if (counter.load() % 10000 == 0) {
    LOG("iter = {} | context = {} | link_time = {} | bytes = {}", counter.load(), FLAGS_context, stats.link_total_time, stats.total_bytes);
    stats.Reset();
  }
  cudaFree((void*)data);

  counter.store(counter.load() + 1);
  request.set("OK");
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int SLEEP_TIME{100000};  //! Milliseconds.
  std::cout << "***Creating a server****" << "tcp://*:" +
  std::to_string(FLAGS_port) << "|" << FLAGS_context << "|" <<  FLAGS_device_id << std::endl;

  // A Server listening on port 5555 for requests from any IP address.
  Server server{"tcp://*:" + std::to_string(FLAGS_port), "[string]", example_callback};

  std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));

  return 0;
}
