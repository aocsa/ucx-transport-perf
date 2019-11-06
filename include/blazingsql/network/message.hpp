
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


class Message {
public:
  Message() = default;

  Message(const std::string& data) : data_{data} {}

  Message(std::string&& data) : data_{std::move(data)} {}

  Message(const char* data, int length) : data_{data, data + length} {}

  Message(const Message & other) : Message{other.data_} {}

  Message(Message && other) noexcept : data_{std::move(other.data_)} {}

  Message & operator=(const Message & rhs) {
    if (this != std::addressof(rhs)) {
      data_ = rhs.data_;
    }
    return *this;
  }

  Message & operator=(Message && rhs) noexcept {
    if (this != std::addressof(rhs)) {
      data_ = std::move(rhs.data_);
      rhs.data_.clear();
    }
    return *this;
  }
  Message & operator=(std::string rhs) {
    data_ = rhs;
    return *this;
  }

  /**
   * @brief Returns true if lhs is equal to rhs, false otherwise.
   */
  inline bool operator==(const Message & rhs) const {
    return (data_ == rhs.data_);
  }
  /**
   * @brief Returns the string information contained in the message.
   */
  inline std::string get() const {
    return data_;
  }

  /**
   * @brief Modifies the string information contained in the message.
   */
  inline void set(const std::string& data) {
    data_ = data;
  }

  /**
   * @brief Set the message content to an empty string.
   */
  inline void clear() {
    data_.clear();
  }

  /**
   * @brief Returns true if the message is empty, false otherwise.
   */
  inline bool empty() {
    return data_.empty();
  }

  inline size_t size() {
    return data_.length();
  }
  char* data() const {
    return (char*)data_.c_str();
  }

  static std::string getTopic() {
    return "[MESSAGE]";
  }

private:
  std::string data_{""};
};

