#pragma once

#define BZ_INTERFACE(Kind)                                                     \
public:                                                                        \
  virtual ~Kind() = default;                                                   \
                                                                               \
protected:                                                                     \
  explicit Kind() = default;                                                   \
                                                                               \
private:                                                                       \
  Kind(const Kind &)  = delete;                                                \
  Kind(const Kind &&) = delete;                                                \
  void operator=(const Kind &) = delete;                                       \
  void operator=(const Kind &&) = delete
