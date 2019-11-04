#pragma once
#include <blazingdb/uc/Buffer.hpp>

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
class Trader;
namespace internal {

class UC_NOEXPORT ViewBuffer : public Buffer {
public:
  explicit ViewBuffer(const void* &data, size_t size);

  ~ViewBuffer() final;

  std::unique_ptr<Transport>
  Link(Buffer * /* buffer */) const final {
    throw std::runtime_error("Not implemented");
  }

  std::unique_ptr<const Record::Serialized>
  SerializedRecord() const noexcept final;

  std::unique_ptr<Transport> Link(const std::uint8_t *recordData,
                                  size_t recordSize) final;
 
  UC_CONCRETE(ViewBuffer);

private:
    const void* &data_;
    size_t size_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
 
