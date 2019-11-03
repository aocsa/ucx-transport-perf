
#pragma once

#include "blazingsql/utils/macros.h"
#include "socket.hpp"
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <zmq.hpp>

template<class T>
class Client {
public:
  explicit Client(const std::string& address, int timeout = 2000, int linger = -1)
      : socket_{zmq_socket_type::req, T::getTopic()},
        address_{address},
        timeout_{timeout},
        linger_{linger}
  {
    initClient();
  }

  /**
  * @brief Sends the request to a server and waits for an answer.
  * @param [in,out] msg - simple_msgs class wrapper for Flatbuffer messages..
  */
  bool request(T& msg) {
    bool success{false};

    // Send the message to the Server and receive back the response.
    if (socket_.send(msg, "[SIMPLE Client] - ")) {
      if (socket_.receive(msg, "[SIMPLE Client] - ")) {
        success = true;
      } else {
        std::cerr << "[SIMPLE Client] - No reply received. Aborting this request." << std::endl;
        // If no message was received back we need to delete the existing socket and create a new one.
        socket_.close();
        initClient();
      }
    }
    return success;
  }

  /**
   * @brief Query the endpoint that this object is bound to.
   *
   * Can be used to find the bound port if binding to ephemeral ports.
   * @return the endpoint in form of a ZMQ DSN string, i.e. "tcp://0.0.0.0:8000"
   */
  const std::string& endpoint() { return socket_.endpoint(); }

private:
  // Initialize the client, setting up the socket and its configuration.
  void initClient() {
    if (!socket_.is_valid()) { socket_.init_socket(zmq_socket_type::req); }
    socket_.set_timeout(timeout_);
    socket_.setLinger(linger_);
    socket_.connect(address_);
  }

  Socket socket_;   //! The internal socket.
  std::string address_{""};  //! The address the Client is connected to.
  int timeout_{30000};       //! Milliseconds the Client should wait for a reply from a Server.
  int linger_{-1};           //! Milliseconds the messages linger in memory after the socket is closed.
};