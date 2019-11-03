#include "blazingsql/api.hpp"

int main() {
  const int N_RUN{25};
  const int SLEEP_TIME{1000};  //!  Milliseconds.

  // Create a pose message
  String pose("{5.0, 6.0, 7.0}");
  // Create a Client that will send request to a Server on "localhost" and on port "5555".
  Client<String> client{"tcp://localhost:5555"};

  // Send a request every SLEEP_TIME milliseconds for N_RUN times.
  // The request is a Pose message, the reply is the modified Pose message.
  // The message is modified by the Server that listens on localhost:5555 accordingly to its callback function.
  for (auto i = 0; i < N_RUN; ++i) {
    std::cout << "Sending: \n" << pose.data() << std::endl;
    if (client.request(pose)) {
      std::cout << "Receiving: \n" << pose.data() << std::endl;
    } else {
      std::cerr << "Request to the server failed." << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
  }
  std::cout << "Requesting ended." << std::endl;
  return 0;
}