#include "ManagedContext.hpp"

#include <cassert>
#include <cstring>

#include "Resource.hpp"
#include "agents/AddressableAgent.hpp"
#include "agents/ManagedAgent.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

namespace {
auto defaultCallable = []() {};

template <class Callable = decltype(defaultCallable)>
inline static void check(
    const bool value, const std::string &message,
    Callable &&callable =
        static_cast<typename std::remove_reference<Callable>::type &&>(
            defaultCallable)) {
  if (value) {
    callable();
    std::cerr << "ERROR: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <class Callable = decltype(defaultCallable)>
inline static void check(
    const ucs_status_t &status, const std::string &message,
    Callable &&callable =
        static_cast<typename std::remove_reference<Callable>::type &&>(
            defaultCallable)) {
  check(UCS_OK != status, message, callable);
}
}  // namespace

uct_component_h GetComponent(const Resource &resource) {
  ucs_status_t status;

  uct_component_h *components = nullptr;
  unsigned components_length = 0;

  status = uct_query_components(&components, &components_length);

  uct_component_h component = nullptr;

  for (unsigned i = 0; i < components_length; ++i) {
    uct_component_attr_t component_attr;
    component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT |
                                UCT_COMPONENT_ATTR_FIELD_NAME;

    status = uct_component_query(components[i], &component_attr);

    check(status, "uct component query md resources",
          [components]() { uct_release_component_list(components); });

    if (!std::strcmp(resource.md_name(), component_attr.name)) {
      component = components[i];
      break;
    }
  }

  uct_release_component_list(components);

  if (nullptr == component) {
    throw std::runtime_error(std::string{"Not UCX "} + resource.md_name() +
                             std::string{" support"});
  }

  return component;
}

ManagedContext::ManagedContext(const Resource &resource, const Trader &trader)
    : resource_{resource},
      md_config_{nullptr},
      md_{UCT_MEM_HANDLE_NULL},
      md_attr_{},
      iface_config_{nullptr},
      async_context_{nullptr},
      worker_{nullptr},
      iface_{nullptr},
      device_addr_{nullptr},
      iface_addr_{nullptr},
      trader_{trader} {
  component_ = GetComponent(resource);
  uct_md_config_t *md_config;
  CHECK_UCS(uct_md_config_read(component_, nullptr, nullptr, &md_config));
  CHECK_UCS(uct_md_config_read(component_, nullptr, nullptr, &md_config_));
  CHECK_UCS(uct_md_open(component_, resource.md_name(), md_config_, &md_));
  CHECK_UCS(uct_md_query(md_, &md_attr_));

  CHECK_UCS(uct_md_iface_config_read(
      md_, resource.tl_name(), nullptr, nullptr, &iface_config_));

  // disable cache to avoid issue related with ipc memhandle close
  uct_config_modify(iface_config_, "CACHE", "n");
  CHECK_UCS(ucs_async_context_create(UCS_ASYNC_MODE_THREAD, &async_context_));
  CHECK_UCS(
      uct_worker_create(async_context_, UCS_THREAD_MODE_SINGLE, &worker_));

  uct_iface_params_t iface_params{
      UCT_IFACE_PARAM_FIELD_OPEN_MODE | UCT_IFACE_PARAM_FIELD_DEVICE |
          UCT_IFACE_PARAM_FIELD_STATS_ROOT | UCT_IFACE_PARAM_FIELD_RX_HEADROOM |
          UCT_IFACE_PARAM_FIELD_CPU_MASK,
      {{0}},
      UCT_IFACE_OPEN_MODE_DEVICE,
      {{resource.tl_name(), resource.dev_name()}},
      nullptr,
      0,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr};

  CHECK_UCS(
      uct_iface_open(md_, worker_, &iface_params, iface_config_, &iface_));

  uct_iface_attr_t iface_attr;
  CHECK_UCS(uct_iface_query(iface_, &iface_attr));
  uct_iface_progress_enable(iface_,
                            UCT_PROGRESS_THREAD_SAFE | UCT_PROGRESS_RECV);

  device_addr_ = reinterpret_cast<uct_device_addr_t *>(
      new std::uint8_t[iface_attr.device_addr_len]);
  assert(nullptr != device_addr_);
  CHECK_UCS(uct_iface_get_device_address(iface_, device_addr_));

  iface_addr_ = reinterpret_cast<uct_iface_addr_t *>(
      new std::uint8_t[iface_attr.iface_addr_len]);
  assert(nullptr != iface_addr_);
  CHECK_UCS(uct_iface_get_address(iface_, iface_addr_));
}

ManagedContext::~ManagedContext() {
  delete[] reinterpret_cast<std::uint8_t *>(iface_addr_);
  delete[] reinterpret_cast<std::uint8_t *>(device_addr_);

  uct_iface_close(iface_);

  uct_worker_destroy(worker_);
  ucs_async_context_destroy(async_context_);

  uct_md_close(md_);

  uct_config_release(iface_config_);
  uct_config_release(md_config_);
}

std::unique_ptr<Agent>
ManagedContext::OwnAgent() const {
  return std::make_unique<ManagedAgent>(md_,
                                        md_attr_,
                                        *async_context_,
                                        worker_,
                                        iface_,
                                        *device_addr_,
                                        *iface_addr_,
                                        component_);
}

std::unique_ptr<Agent>
ManagedContext::PeerAgent() const {
  return std::make_unique<AddressableAgent>(md_, md_attr_, trader_, ep_,
                                            *async_context_, worker_, iface_,
                                            component_);
}

std::unique_ptr<Agent>
ManagedContext::Agent() const {
  uct_ep_params_t ep_params;
  ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
      UCT_EP_PARAM_FIELD_IFACE_ADDR |
      UCT_EP_PARAM_FIELD_DEV_ADDR;
  ep_params.iface      = iface_;
  ep_params.iface_addr = iface_addr_;
  ep_params.dev_addr   = device_addr_;

  CHECK_UCS(uct_ep_create(&ep_params, const_cast<uct_ep_h *>(&ep_)));

  return std::make_unique<AddressableAgent>(md_, md_attr_, trader_, ep_,
                                            *async_context_, worker_, iface_,
                                            component_);
}

std::size_t
ManagedContext::serializedRecordSize() const noexcept {
  //! @see const void * for `RemotableBuffer`
  return md_attr_.rkey_packed_size + sizeof(const void *);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
