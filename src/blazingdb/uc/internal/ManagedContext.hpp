#ifndef BLAZINGDB_UC_INTERNAL_MANAGED_CONTEXT_HPP_
#define BLAZINGDB_UC_INTERNAL_MANAGED_CONTEXT_HPP_

#include <blazingdb/uc/Context.hpp>

#include <ucs/async/async_fwd.h>
#include <uct/api/uct.h>

#include "macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {
class Resource;

class UC_NOEXPORT ManagedContext : public Context {
public:

  explicit ManagedContext(const Resource &resource, const Trader &trader);

  ~ManagedContext();

  std::unique_ptr<uc::Agent>
  OwnAgent() const final;

  std::unique_ptr<uc::Agent>
  PeerAgent() const final;

  virtual std::unique_ptr<uc::Agent>
  Agent() const;

  const Resource &
  resource() const noexcept {
    return resource_;
  }

  std::size_t
  serializedRecordSize() const noexcept final;

private:
  const Resource &resource_;

  uct_md_config_t *md_config_;
  uct_md_h         md_;
  uct_md_attr_t    md_attr_;

  uct_iface_config_t *iface_config_;

  ucs_async_context_t *async_context_;
  uct_worker_h         worker_;
  uct_iface_h          iface_;

  uct_device_addr_t *device_addr_;
  uct_iface_addr_t * iface_addr_;

  const Trader &trader_;

  uct_ep_h ep_;

  uct_component_h component_;

  UC_CONCRETE(ManagedContext);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
