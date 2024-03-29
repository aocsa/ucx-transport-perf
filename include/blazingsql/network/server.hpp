
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

class Server {
public:
  Server() = default;

  /**
   * @brief Creates a ZMQ_REP socket and connects it to the given address.
   * The user defined callback function is responsible for taking
   * the received request and filling it with the reply data.
   * @param [in] address - address the server binds to, in the form: \<PROTOCOL\>://\<HOSTNAME\>:\<PORT\>. e.g
   * tcp://localhost:5555.
   * @param [in] callback - user defined callback function for incoming requests.
   * @param [in] timeout - Time the server will block the thread waiting for a message. In
   * milliseconds.
   * @param [in] linger - Time the unsent messages linger in memory after the socket
   * is closed. In milliseconds. Default is -1 (infinite).
   */
  explicit Server(const std::string& address, const std::string& topic, const std::function<void(Message&)>& callback, int timeout = 1000,
                  int linger = -1)
      : socket_{new Socket(zmq_socket_type::rep, topic)}, callback_{callback} {
    socket_->set_timeout(timeout);
    socket_->setLinger(linger);
    socket_->bind(address);
    initServer();
  }

  // A Server cannot be copied, only moved
  Server(const Server& other) = delete;
  Server& operator=(const Server& other) = delete;

  /**
   * @brief Move constructor.
   */
  Server(Server&& other) : socket_{std::move(other.socket_)}, callback_{std::move(other.callback_)} {
    other.stop();  //! The moved Server has to be stopped.
    initServer();
  }

  /**
   * @brief Move assignment operator.
   */
  Server& operator=(Server&& other) {
    stop();                 //! Stop the current Server object.
    if (other.isValid()) {  //! Move the Server only if it's a valid one, e.g. if it was not default constructed.
      other.stop();         //! The moved Server has to be stopped.
      socket_ = std::move(other.socket_);
      callback_ = std::move(other.callback_);
      initServer();
    }
    return *this;
  }

  ~Server() { stop(); }

  /**
   * @brief Query the endpoint that this object is bound to.
   *
   * Can be used to find the bound port if binding to ephemeral ports.
   * @return the endpoint in form of a ZMQ DSN string, i.e. "tcp://0.0.0.0:8000"
   */
  const std::string& endpoint() { return socket_->endpoint(); }

private:
  /**
   * @brief Stop the server loop. No further requests will be handled.
   */
  void stop() {
    if (isValid()) {
      alive_->store(false);
      if (server_thread_.joinable()) { server_thread_.join(); }
    }
  }

  /**
   * @brief Checks if the Server is properly initialied and its internal thread is running.
   */
  inline bool isValid() const { return alive_ == nullptr ? false : alive_->load(); }

  /**
   * @brief Initializes the server thread.
   */
  void initServer() {
    alive_ = std::make_shared<std::atomic<bool>>(true);

    // Start the thread of the server if not yet done. Wait for requests on the
    // dedicated thread.
    if (!server_thread_.joinable() && socket_ != nullptr) {
      server_thread_ = std::thread(&Server::awaitRequest, this, alive_, socket_);
    }
  }

  /**
   * @brief Keep waiting for a request to arrive. Process the request using the
   * callback function and reply.
   */
  void awaitRequest(std::shared_ptr<std::atomic<bool>> alive, std::shared_ptr<Socket> socket) {
    while (alive->load()) {
      zmq::message_t response_message;
      if (socket->receive(response_message, "[TRANSPORT Server] - ")) {
        Message msg = Message((const char*)response_message.data(), response_message.size());
        if (alive->load()) {
          callback_(msg);
        }
        if (alive->load()) {
          reply(socket.get(), msg);
        }
      }
    }
  }

  /**
   * @brief Sends the message back to the client who requested it.
   * @param [in] msg - The message to be sent.
   */
  void reply(Socket* socket, Message& msg) {
    zmq::message_t message{msg.data(), msg.size()};
    socket->send(message, "[TRANSPORT Server] - ");
  }

  std::shared_ptr<std::atomic<bool>> alive_{nullptr};  //! Flag keeping track of the internal thread's state.
  std::shared_ptr<Socket> socket_{nullptr};     //! The internal socket.
  std::function<void(Message&)> callback_;                   //! The callback function called at each message arrival.
  std::thread server_thread_{};                        //! The internal Server thread on which the given callback runs.
};