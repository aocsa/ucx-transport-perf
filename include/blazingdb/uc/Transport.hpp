#ifndef BLAZINGDB_UC_TRANSPORT_HPP_
#define BLAZINGDB_UC_TRANSPORT_HPP_

#include <future>

#include "blazingdb/util/macros.hpp"

namespace blazingdb {
namespace uc {

/// \brief Relation between two buffers to transport their memory content
class Transport {
public:
  /// \brief Share memory from own to peer
  virtual bool Get() = 0;

  /// \brief Share memory from peer to own
  // virtual std::future<void>
  // Put() = 0;

  BZ_INTERFACE(Transport);
};

}  // namespace uc
}  // namespace blazingdb

#endif
