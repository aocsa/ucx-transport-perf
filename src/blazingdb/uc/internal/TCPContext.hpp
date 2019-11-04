#pragma once

#include "agents/TCPAgent.hpp"
#include <blazingdb/uc/Context.hpp>
#include <driver_types.h>

#include "blazingdb/uc/internal/ManagedContext.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT TCPContext : public Context {
public:
  TCPContext()
    {}

  ~TCPContext() = default;

  std::unique_ptr<uc::Agent>
  OwnAgent() const final {
    return nullptr;
  }

  std::unique_ptr<uc::Agent>
  PeerAgent() const final {
    return nullptr;
  }

 std::size_t
  serializedRecordSize() const noexcept final {
   //TODO
    return sizeof(size_t); // just vector size value
 }

  std::unique_ptr<uc::Agent> Agent() const final {
    return std::make_unique<TCPAgent>();
  }
};

} // namespace internal
} // namespace uc
} // namespace blazingdb