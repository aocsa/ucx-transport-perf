#pragma once


#include <blazingdb/uc/Agent.hpp>
#include "../buffers/TCPBuffer.hpp"

namespace blazingdb {
namespace uc {
class Trader;
namespace internal {

class TCPAgent : public Agent {
public:
  explicit TCPAgent(){}

  std::unique_ptr<Buffer>
  Register(const void * &data, std::size_t size) const noexcept final {
    return std::make_unique<TCPBuffer>(data, size);
  }

private:

};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
