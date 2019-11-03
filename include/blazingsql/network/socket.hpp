#pragma once

#include "blazingsql/utils/macros.h"
#include "message.hpp"
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <zmq.hpp>

using zmq_socket_type = zmq::socket_type;


class Socket {
public:
  explicit Socket(const zmq_socket_type& type, const std::string& topic);
  void bind(const std::string& address);
  void connect(const std::string& address);
  bool send(GenericMessage& msg, const std::string& custom_error = "[Communication Error] - ") const;
  bool receive(GenericMessage& msg, const std::string& custom_error = "");
  void filter();
  void set_timeout(int timeout);
  void init_socket(const zmq_socket_type& type);
  void close();
  bool is_valid();
  inline const std::string& endpoint() { return endpoint_; }

  void setLinger(int linger) {
    std::lock_guard<std::mutex> lock{mutex_};
    socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
  }

private:
  mutable std::mutex mutex_{};                 //! Mutex for thread-safety.
  std::string topic_{""};                   //! The message topic, internally defined for each SIMPLE message.
  std::unique_ptr<zmq::socket_t> socket_;      //! The internal ZMQ socket.
  std::string endpoint_{""};                //! Stores the used endpoint for connection.

  BZ_INTERFACE(Socket);
};
