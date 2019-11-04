#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_TRANSPORTS_ZCOPY_TRANSPORT_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_TRANSPORTS_ZCOPY_TRANSPORT_HPP_

#include <blazingdb/uc/Transport.hpp>

#include <uct/api/uct.h>

#include "../../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {
class AccessibleBuffer;
class RemoteBuffer;

class UC_NOEXPORT ZCopyTransport : public Transport {
public:
  explicit ZCopyTransport(const AccessibleBuffer&    sendingBuffer,
                          const RemoteBuffer&        receivingBuffer,
                          const uct_ep_h&            ep,
                          const uct_md_attr_t&       md_attr,
                          const ucs_async_context_t& async_context,
                          const uct_worker_h&        worker,
                          const uct_iface_h&         iface);

  bool
  Get() final;

private:
  uct_completion_t           completion_;
  const AccessibleBuffer&    sendingBuffer_;
  const RemoteBuffer&        receivingBuffer_;
  const uct_ep_h&            ep_;
  const uct_md_attr_t&       md_attr_;
  const ucs_async_context_t& async_context_;
  const uct_worker_h&        worker_;
  const uct_iface_h&         iface_;
};


/// \brief Relation between two buffers to transport their memory content
class ViewTransport : public Transport {
public:
  explicit ViewTransport() = default;

  /// \brief Share memory from own to peer
  bool Get() final {
//     return std::async(std::launch::async, [](){ return; });
return true;
  }

  /// \brief Share memory from peer to own
  // virtual std::future<void>
  // Put() = 0;

};

class TCPTransport : public Transport {
public:
  explicit TCPTransport() = default;

  /// \brief Share memory from own to peer
  bool Get() final {
//     return std::async(std::launch::async, [](){ return; });
    return true;
  }

  /// \brief Share memory from peer to own
  // virtual std::future<void>
  // Put() = 0;

};




}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
