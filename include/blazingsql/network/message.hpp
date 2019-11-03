
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

class GenericMessage {
public:
  virtual ~GenericMessage() = default;

  virtual GenericMessage& operator=(std::string rhs) = 0;

  virtual size_t size()  = 0 ;
  virtual char* data() const = 0 ;
};


class String : public GenericMessage {
public:
  String() = default;

  String(const std::string& data) : data_{data} {}

  String(std::string&& data) : data_{std::move(data)} {}

  String(const char* data) : data_{data} {}

  String(const String& other, const std::lock_guard<std::mutex>&) : String{other.data_} {}

  String(String&& other, const std::lock_guard<std::mutex>&) noexcept : data_{std::move(other.data_)} {}

  String(const String& other) : String{other, std::lock_guard<std::mutex>(other.mutex_)} {}

  String(String&& other) noexcept
      : String{std::forward<String>(other), std::lock_guard<std::mutex>(other.mutex_)} {}

  String& operator=(const String& rhs) {
    if (this != std::addressof(rhs)) {
      std::lock(mutex_, rhs.mutex_);
      std::lock_guard<std::mutex> lock{mutex_, std::adopt_lock};
      std::lock_guard<std::mutex> other_lock{rhs.mutex_, std::adopt_lock};
      data_ = rhs.data_;
    }
    return *this;
  }

  String& operator=(String&& rhs) noexcept {
    if (this != std::addressof(rhs)) {
      std::lock(mutex_, rhs.mutex_);
      std::lock_guard<std::mutex> lock{mutex_, std::adopt_lock};
      std::lock_guard<std::mutex> other_lock{rhs.mutex_, std::adopt_lock};
      data_ = std::move(rhs.data_);
      rhs.data_.clear();
    }
    return *this;
  }
  String& operator=(std::string rhs) {
    std::lock_guard<std::mutex> lock{mutex_};
    data_ = rhs;
    return *this;
  }

  /**
   * @brief Returns true if lhs is equal to rhs, false otherwise.
   */
  inline bool operator==(const String& rhs) const {
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
    return "STRG";
  }

private:

  mutable std::mutex mutex_{};
  std::string data_{""};
};

