#include "blazingsql/api.hpp"
#include <gflags/gflags.h>

DEFINE_int32(port, 5555, "Server port to listen on");

void example_callback(String& p) {
  std::cout << "Received a string: " << p.size() << std::endl;
  p.set(p.get() + "*");
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int SLEEP_TIME{10000};  //! Milliseconds.
  std::cout << "***Creating a server****" << "tcp://*:" + std::to_string(FLAGS_port) << std::endl;

  // A Server listening on port 5555 for requests from any IP address.
  Server<String> server{"tcp://*:" + std::to_string(FLAGS_port), example_callback};

  // Wait for 60 seconds. The Service callback is called asynchronously.
  std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));

  std::cout << "Closing the server." << std::endl;
  return 0;
}