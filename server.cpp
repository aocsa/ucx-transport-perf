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

std::atomic<size_t> counter{0};
void example_callback(Message &request) {
  std::cout << counter.load() << std::endl;
  counter.store(counter.load() + 1);

  using namespace blazingdb::uc;

  // why const?
  const void *data = CreateGpuBuffer(BUFFER_LENGTH);
  Print("init_peer", data, BUFFER_LENGTH);

  auto context = CreateUCXContext(FLAGS_context);
  auto agent = context->Agent();
  // why const?
  auto buffer = agent->Register(data, BUFFER_LENGTH);

  std::cout << "buffer_size:" << std::dec << request.size()
            << std::endl;
  std::cout << "buffer_descriptors:" << std::endl;
  size_t checksum = 0;
  for (size_t i = 0; i < request.size(); i++) {
    std::cout << (int)request.data()[i] << ", ";
    checksum += (int)request.data()[i];
  }
  std::cout << "checksum:" << checksum << std::endl;

  auto transport = buffer->Link((const uint8_t *)request.data(), request.size());

  auto future = transport->Get();

  Print("peer", data, BUFFER_LENGTH);

  request.set("OK");
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int SLEEP_TIME{10000};  //! Milliseconds.
  std::cout << "***Creating a server****" << "tcp://*:" +
  std::to_string(FLAGS_port) << std::endl;

  // A Server listening on port 5555 for requests from any IP address.
  Server server{"tcp://*:" + std::to_string(FLAGS_port), "[string]", example_callback};

  std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
  return 0;
}