#pragma once
#include "transports/ZCopyTransport.hpp"
#include <blazingdb/uc/Buffer.hpp>
#include <blazingdb/uc/internal/buffers/records/RemotableRecord.hpp>
#include <cassert>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
class Trader;
namespace internal {

class UC_NOEXPORT TCPBuffer : public Buffer {
public:
  explicit TCPBuffer(const void* &data, size_t size)
      : data_{data}, size_{size}
  {}

  ~TCPBuffer() final {

  }

  std::unique_ptr<Transport>
  Link(Buffer * /* buffer */) const final {
    throw std::runtime_error("Not implemented");
  }

  std::unique_ptr<const Record::Serialized>
  SerializedRecord() const noexcept final {
    std::basic_string<uint8_t> bytes(this->size_, '\0');
    auto cudaStatus = cudaMemcpy((void*)bytes.data(), data_, this->size_, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStatus);

    return std::make_unique<TPCViewSerialized>(bytes);
  }

  std::unique_ptr<Transport> Link(const std::uint8_t *host_pointer,
                                  size_t recordSize) final {
    auto cudaStatus = cudaMemcpy((void*)this->data_, host_pointer, recordSize, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStatus);

//    cudaStatus = cudaMemcpy(buffer, data + offset, size, cudaMemcpyHostToDevice);

    return std::make_unique<TCPTransport>();
  }
 
  UC_CONCRETE(TCPBuffer);

private:
    const void* &data_;
    size_t size_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
 
