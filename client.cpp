#include "blazingdb/uc/API.hpp"
#include "blazingsql/api.hpp"
#include "constexpr_header.h"

#include <cstring>
#include <iomanip>
#include <gflags/gflags.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

DEFINE_int32(port, 5555, "Server port to listen on");
DEFINE_string(context, "tcp", "UCX Context");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace blazingdb::uc;
  const void *data = CreateData(BUFFER_LENGTH, ownSeed, ownOffset);
  Print("own", data, BUFFER_LENGTH);

  auto context = CreateUCXContext(FLAGS_context);
  auto agent   = context->Agent();
  auto buffer  = agent->Register(data, BUFFER_LENGTH);

  auto buffer_descriptors_serialized = buffer->SerializedRecord();
  const uint8_t *buffer_descriptors = buffer_descriptors_serialized->Data();

  std::cout << "buffer_descriptors->Size(): " << std::dec << buffer_descriptors_serialized->Size() << std::endl;

  size_t checksum = 0;
  std::cout << "buffer_descriptors:" << std::endl;
  for (size_t i = 0; i < buffer_descriptors_serialized->Size(); i++) {
    std::cout << +(unsigned char)buffer_descriptors[i] << ", ";
    checksum += (unsigned char)buffer_descriptors[i];
  }
  std::cout << "checksum:" << std::dec << checksum << std::endl;
  Client client{"tcp://localhost:" + std::to_string(FLAGS_port), "[string]"};
  Message msg((const char*)buffer_descriptors, buffer_descriptors_serialized->Size());
  auto res = client.send(msg);
  cudaFree((void*)data);
  if (res.isOk()) {
    std::cout << "Receiving: \n" << res.unwrap().data() << std::endl;
  } else {
    std::cerr << "Request to the server failed." << std::endl;
  }
  printf ("Received \n");
}
