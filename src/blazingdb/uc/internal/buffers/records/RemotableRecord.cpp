#include "RemotableRecord.hpp"

#include <uct/api/uct.h>

#include <cuda.h>

#include "blazingdb/util/macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

namespace {
class UC_NOEXPORT Offset {
public:
  UC_INLINE void Unpack(const Offset &other, uct_rkey_bundle_t *key_bundle,
                        uct_component_h component) const noexcept {
    //    if (id_ == other.id_) {
    //      key_bundle->rkey = reinterpret_cast<uct_rkey_t>(
    //          reinterpret_cast<const std::uint8_t *>(this) +
    //          kComponentOffset);
    //    } else {
    CHECK_UCS(uct_rkey_unpack(
        component, reinterpret_cast<const void *>(this), key_bundle));
    //    }
  }

  static UC_INLINE Offset *Make(uct_rkey_t rkey) noexcept {
    return reinterpret_cast<Offset *>(rkey);
  }

  static UC_INLINE UC_CONST Offset *Make(uct_mem_h mem) noexcept {
    return reinterpret_cast<Offset *>(reinterpret_cast<std::uint8_t *>(mem) -
                                      kComponentOffset);
  }

  explicit Offset() = delete;
  ~Offset() = delete;

private:
  static constexpr std::ptrdiff_t UCT_MD_COMPONENT_NAME_MAX =
      8;  // @see ucx v1.5.1
  static constexpr std::ptrdiff_t kComponentOffset = UCT_MD_COMPONENT_NAME_MAX;
  static constexpr std::ptrdiff_t kIpcOffset = CU_IPC_HANDLE_SIZE;

  std::uint8_t pad_[kComponentOffset + kIpcOffset];
  std::uint64_t base_;
  std::size_t size_;
  std::uint32_t id_;

  UC_CONCRETE(Offset);
};
}  // namespace

RemotableRecord::RemotableRecord(const void *const pointer,
                                 const uct_mem_h &mem,
                                 const uct_md_attr_t &md_attr, uct_rkey_t *rkey,
                                 std::uintptr_t *address,
                                 uct_rkey_bundle_t *key_bundle,
                                 uct_component_h component)
    : id_{++count},
      pointer_{pointer},
      mem_{mem},
      md_attr_{md_attr},
      rkey_{rkey},
      address_{address},
      key_bundle_{*key_bundle},
      component_{component} {}

std::unique_ptr<const Record::Serialized> RemotableRecord::GetOwn() const
    noexcept {
  return std::make_unique<PlainSerialized>(*rkey_, md_attr_.rkey_packed_size,
                                           pointer_);
}

static inline void PrintRecordData(const void *data, std::size_t size) {
  std::cout << "\033[32m>>> size = " << std::dec << size << std::endl;
  for (std::size_t i = 0; i < size; i++) {
    if (i == size - 8) std::cout << "\033[33m";
    std::cout << ' ' << std::hex << std::uppercase
              << static_cast<unsigned>(
                     static_cast<const unsigned char *>(data)[i]);
  }
  union {
    std::uintptr_t pointer;
    std::uint8_t buffer[8];
  } addr;
  std::memcpy(addr.buffer,
              reinterpret_cast<const std::uint8_t *>(data) + size - 8, 8);
  std::cout << std::endl << ">>> addr = " << addr.pointer;
  std::cout << "\033[0m" << std::endl;
}

void RemotableRecord::SetPeer(const void *bytes) noexcept {
  auto data = static_cast<const std::uint8_t *>(bytes);
  const std::size_t size = md_attr_.rkey_packed_size;
  std::memcpy(reinterpret_cast<void *>(*rkey_), data, size);
  std::memcpy(address_, data + size, sizeof(*address_));
  // if (0U == (md_attr_.cap.reg_mem_types & UC_BIT(UCS_MEMORY_TYPE_CUDA))) {
//  PrintRecordData(bytes, size + sizeof(void *));

  CHECK_UCS(uct_rkey_unpack(component_,
                            reinterpret_cast<void *>(*rkey_), &key_bundle_));

  // } else {
  //   auto rkeyOffset = Offset::Make(*rkey_);
  //   rkeyOffset->Unpack(*Offset::Make(mem_), &key_bundle_,
  //   component_);
  // }
}

// sizeof (uct_cuda_ipc_key_t) 96 + 8
RemotableRecord::PlainSerialized::PlainSerialized(const uct_rkey_t &rkey,
                                                  const std::size_t offset,
                                                  const void *const pointer)
    : size_{offset + sizeof(pointer)}, data_{new std::uint8_t[size_]} {
  std::memcpy(data_, reinterpret_cast<const void *>(rkey), offset);
  std::memcpy(data_ + offset, &pointer, sizeof(pointer));

//  PrintRecordData(data_, size_);
}

std::uint64_t RemotableRecord::count = -1;

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
