#pragma once

#include "blazingsql/utils/macros.h"
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

  bool send(zmq::message_t & msg, const std::string& custom_error = "[TRANSPORT Error] - ") const;

//  bool send_multipart(std::vector<Message> & msg, const std::string& custom_error = "[TRANSPORT Error] - ") const;

  bool receive(zmq::message_t &response_message, const std::string& custom_error = "");
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
  std::string topic_{""};                   //! The message topic, internally defined for each TRANSPORT message.
  std::unique_ptr<zmq::socket_t> socket_;      //! The internal ZMQ socket.
  std::string endpoint_{""};                //! Stores the used endpoint for connection.

  BZ_INTERFACE(Socket);
};


struct ContextManager {
  static zmq::context_t& instance();
  static void destroy();

private:
  static std::mutex context_mutex_;
  static std::shared_ptr<zmq::context_t> context_;  //! zmq::context_t is automatically disposed.
};
//! Static member are here initialized.
std::mutex ContextManager::context_mutex_{};
std::shared_ptr<zmq::context_t> ContextManager::context_{nullptr};

zmq::context_t& ContextManager::instance() {
  std::lock_guard<std::mutex> lock{context_mutex_};
  // Create a new ZMQ context or return the existing one.
  if (context_ == nullptr) {
    context_ = std::make_shared<zmq::context_t>();
  }
  return *context_.get();
}

void ContextManager::destroy() {
  std::lock_guard<std::mutex> lock{context_mutex_};
  context_ = nullptr;
}

Socket::Socket(const zmq_socket_type& type, const std::string& topic)
    : topic_{topic}
{
  init_socket(type);
}

void Socket::bind(const std::string& address) {
  std::lock_guard<std::mutex> lock{mutex_};
  try {
    socket_->bind(address);
  } catch (const zmq::error_t& error) {
    throw std::runtime_error("[TRANSPORT Error] - Cannot bind to the address/port: " + address +
        ". ZMQ Error: " + error.what());
  }

  // Query the bound endpoint from the ZMQ API.
  char last_endpoint[1024];
  size_t size = sizeof(last_endpoint);
  socket_->getsockopt(ZMQ_LAST_ENDPOINT, &last_endpoint, &size);
  endpoint_ = last_endpoint;
}
void Socket::filter() {
  std::lock_guard<std::mutex> lock{mutex_};
  socket_->setsockopt(ZMQ_SUBSCRIBE, topic_.c_str(), topic_.size());
}
void Socket::init_socket(const zmq_socket_type& type){
  std::lock_guard<std::mutex> lock{mutex_};
  if (socket_ == nullptr) {
    socket_ = std::make_unique<zmq::socket_t>(ContextManager::instance(), (int)type);
  }
}
void Socket::set_timeout(int timeout) {
  std::lock_guard<std::mutex> lock{mutex_};
  socket_->setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
}

void Socket::close() {
  std::lock_guard<std::mutex> lock{mutex_};
  if (socket_ != nullptr) {
    socket_->close();
    socket_ = nullptr;
  }
}
bool Socket::is_valid() {
  return (socket_ != nullptr);
}

void Socket::connect(const std::string& address) {
  std::lock_guard<std::mutex> lock{mutex_};
  try {
    socket_->connect(address);
  } catch (const zmq::error_t& error) {
    throw std::runtime_error("[TRANSPORT Error] - Cannot connect to the address/port: " + address +
        ". ZMQ Error: " + error.what());
  }
}
//
//bool Socket::send_multipart(std::vector<Message> & msg, const std::string& custom_error) const {
////  zmq::multipart_t
//
//}

bool Socket::send(zmq::message_t& message, const std::string& custom_error) const {
// Early return if socket_ has not been created yet.
  if (socket_ == nullptr) { return false; }

  std::lock_guard<std::mutex> lock{mutex_};

  try {
    auto topic_ptr = topic_.c_str();

    // Initialize the topic message to be sent.
    zmq::message_t topic_message{topic_ptr, topic_.size()};

    // Send the topic first and add the rest of the message after it.
    auto topic_success = socket_->send(topic_message, zmq::send_flags::sndmore);
    auto message_success = socket_->send(message, zmq::send_flags::dontwait);

    // If something wrong happened, throw zmq::error_t().
    if (topic_success.value() == false || message_success.value() == false) {
      throw zmq::error_t();
    }
  } catch (const zmq::error_t& error) {
    std::cerr << custom_error << "Failed to send the message. ZMQ Error: " << error.what() << std::endl;
    return false;
  }
  // Return the number of bytes sent.
  return true;
}

bool Socket::receive(zmq::message_t &response_message,  std::string const& custom_error){
  // Early return if socket_ has not been created yet.
  if (socket_ == nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lock{mutex_};
  zmq::detail::recv_result_t success;

  // Local variables to check if data after the topic message is available and its size.
  int data_past_topic{0};
  auto data_past_topic_size{sizeof(data_past_topic)};

  // Receive the first bytes, this should match the topic message and can be used to check if the right topic (the
  // right message type) has been received. i.e. the received topic message should match the one of the template
  // argument of this socket (stored in the topic_ member variable).
  try {
    zmq::message_t local_message;
    if (!socket_->recv(local_message)) {
      throw zmq::error_t();
    };

    // Check if the received topic matches the right message topic.
    std::string received_message_type = static_cast<char*>(local_message.data());
    // if (std::strncmp(received_message_type.c_str(), topic_.c_str(), std::strlen(topic_.c_str())) != 0) {
    if (received_message_type.compare(0, topic_.size(), topic_, 0, topic_.size()) != 0) {
      std::cerr << custom_error << "Received message type " << received_message_type << " while expecting " << topic_
                << "." << std::endl;
      return false;
    }

    // If all is good, check that there is more data after the topic.
    socket_->getsockopt(ZMQ_RCVMORE, &data_past_topic, &data_past_topic_size);

    if (data_past_topic == 0 || data_past_topic_size == 0) {
      std::cerr << custom_error << "No data inside message." << std::endl;
      return false;
    }

    // Receive the real message.
    success = socket_->recv(response_message);

    // Check if any data has been received.
    if (success.value() == false || response_message.size() == 0) {
      throw zmq::error_t();
    }
  } catch (const zmq::error_t& error) {
    std::cerr << custom_error << "Failed to receive the message. ZMQ Error: " << error.what() << std::endl;
    return false;
  }
  return success.has_value();
}
