//
// Created by aocsa on 11/1/19.

#include "blazingsql/network/socket.hpp"

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
    throw std::runtime_error("[SIMPLE Error] - Cannot bind to the address/port: " + address +
        ". ZMQ Error: " + error.what());
  }

  // Query the bound endpoint from the ZMQ API.
  char last_endpoint[1024];
  size_t size = sizeof(last_endpoint);
  socket_->getsockopt(ZMQ_LAST_ENDPOINT, &last_endpoint, &size);
  endpoint_ = last_endpoint;
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

bool Socket::send(GenericMessage& msg, const std::string& custom_error) const {
// Early return if socket_ has not been created yet.
  if (socket_ == nullptr) { return false; }

  std::lock_guard<std::mutex> lock{mutex_};

  try {
    auto topic_ptr = topic_.c_str();

    // Initialize the topic message to be sent.
    zmq::message_t topic_message{topic_ptr, topic_.size()};

    // Initialize the message itself using the buffer data.
    zmq::message_t message{msg.data(), msg.size()};

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

bool Socket::receive(GenericMessage& msg,  std::string const& custom_error){
  // Early return if socket_ has not been created yet.
  if (socket_ == nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lock{mutex_};
  zmq::detail::recv_result_t success;

  // Local variables to check if data after the topic message is available and its size.
  int data_past_topic{0};
  auto data_past_topic_size{sizeof(data_past_topic)};

  // Local ZMQ message, it is used to collect the data received from the ZMQ socket.
  // This is the pointer handling the lifetime of the received data.
  zmq::message_t local_message;

  // Receive the first bytes, this should match the topic message and can be used to check if the right topic (the
  // right message type) has been received. i.e. the received topic message should match the one of the template
  // argument of this socket (stored in the topic_ member variable).
  try {
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
    success = socket_->recv(local_message);

    // Check if any data has been received.
    if (success.value() == false || local_message.size() == 0) {
      throw zmq::error_t();
    }
  } catch (const zmq::error_t& error) {
    std::cerr << custom_error << "Failed to receive the message. ZMQ Error: " << error.what() << std::endl;
    return false;
  }

  // At this point we are sure that the message habe been correctly received, it has the right topic type and it is
  // not empty.

  // Set the T msg to the data that has been just received.
  char* data_ptr = (char*)local_message.data();
  auto data_size = local_message.size();

  msg = std::string(data_ptr, data_ptr + data_size);
  return success.has_value();
}
