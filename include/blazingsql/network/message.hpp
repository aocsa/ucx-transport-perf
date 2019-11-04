
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

  Message(const Message & other, const std::lock_guard<std::mutex>&) : Message{other.data_} {}

  Message(Message && other, const std::lock_guard<std::mutex>&) noexcept : data_{std::move(other.data_)} {}

  Message(const Message & other) : Message{other, std::lock_guard<std::mutex>(other.mutex_)} {}

  Message(Message && other) noexcept
      : Message{std::forward<Message>(other), std::lock_guard<std::mutex>(other.mutex_)} {}

  Message & operator=(const Message & rhs) {
    if (this != std::addressof(rhs)) {
      std::lock(mutex_, rhs.mutex_);
      std::lock_guard<std::mutex> lock{mutex_, std::adopt_lock};
      std::lock_guard<std::mutex> other_lock{rhs.mutex_, std::adopt_lock};
      data_ = rhs.data_;
    }
    return *this;
  }

  Message & operator=(Message && rhs) noexcept {
    if (this != std::addressof(rhs)) {
      std::lock(mutex_, rhs.mutex_);
      std::lock_guard<std::mutex> lock{mutex_, std::adopt_lock};
      std::lock_guard<std::mutex> other_lock{rhs.mutex_, std::adopt_lock};
      data_ = std::move(rhs.data_);
      rhs.data_.clear();
    }
    return *this;
  }
  Message & operator=(std::string rhs) {
    std::lock_guard<std::mutex> lock{mutex_};
    data_ = rhs;
    return *this;
  }

  /**
   * @brief Returns true if lhs is equal to rhs, false otherwise.
   */
  inline bool operator==(const Message & rhs) const {
    std::lock_guard<std::mutex> lock{mutex_};
    return (data_ == rhs.data_);
  }
  /**
   * @brief Returns the string information contained in the message.
   */
  inline std::string get() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return data_;
  }

  /**
   * @brief Modifies the string information contained in the message.
   */
  inline void set(const std::string& data) {
    std::lock_guard<std::mutex> lock{mutex_};
    data_ = data;
  }

  /**
   * @brief Set the message content to an empty string.
   */
  inline void clear() {
    std::lock_guard<std::mutex> lock{mutex_};
    data_.clear();
  }

  /**
   * @brief Returns true if the message is empty, false otherwise.
   */
  inline bool empty() {
    std::lock_guard<std::mutex> lock{mutex_};
    return data_.empty();
  }

  inline size_t size() {
    std::lock_guard<std::mutex> lock{mutex_};
    return data_.length();
  }
  char* data() const {
    std::lock_guard<std::mutex> lock{mutex_};
    return (char*)data_.c_str();
  }

  static std::string getTopic() {
    return "[MESSAGE]";
  }

private:

  mutable std::mutex mutex_{};
  std::string data_{""};
};

