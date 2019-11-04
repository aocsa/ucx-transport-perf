#include "ViewAgent.hpp"

#include "../buffers/ViewBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

ViewAgent::ViewAgent() {}

std::unique_ptr<Buffer>
ViewAgent::Register(const void* &data, const std::size_t size) const
    noexcept {
  return std::make_unique<ViewBuffer>(data, size);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
