#ifndef BLAZINGDB_UC_INTERNAL_RESOURCE_HPP_
#define BLAZINGDB_UC_INTERNAL_RESOURCE_HPP_

#include "blazingdb/util/macros.hpp"

#include "macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT Resource {
public:
  virtual const char *md_name() const noexcept = 0;
  virtual const char *tl_name() const noexcept = 0;
  virtual const char *dev_name() const noexcept = 0;

  BZ_INTERFACE(Resource);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
