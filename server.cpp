#include "blazingdb/uc/API.hpp"
#include "blazingsql/api.hpp"
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

std::atomic<size_t> counter{0};
void example_callback(Message &request) {
  cudaSetDevice(FLAGS_device_id);

  if (counter.load() % 10000 == 0)
    std::cout << counter.load() << std::endl;
  counter.store(counter.load() + 1);
  using namespace blazingdb::uc;
  const void *data = CreateGpuBuffer(BUFFER_LENGTH);

  auto context = CreateUCXContext(FLAGS_context);
  auto agent = context->Agent();
  auto buffer = agent->Register(data, BUFFER_LENGTH);
  auto transport = buffer->Link((const uint8_t *)request.data(), request.size());
  auto future = transport->Get();
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