#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>
#include "blazingdb/uc/API.hpp"

static constexpr std::size_t BUFFER_LENGTH = 20;
static constexpr std::uint64_t  ownSeed    = 0x1111111111111111lu;
static constexpr std::uint64_t  peerSeed   = 0x2222222222222222lu;
static constexpr std::uint64_t  twinSeed   = 0x3333333333333333lu;
static constexpr std::ptrdiff_t ownOffset  = 1;
static constexpr std::ptrdiff_t peerOffset = 3;
static constexpr std::ptrdiff_t twinOffset = 4;


void
Print(const std::string &name, const void *data, const std::size_t size) {
  std::uint8_t *host = new std::uint8_t[size];

  cudaError_t cudaStatus = cudaMemcpy(host, data, size, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStatus);

  std::stringstream ss;

  ss << ">>> [" << std::setw(9) << name << "]";
  for (std::size_t i = 0; i < size; i++) {
    ss << ' ' << std::setfill('0') << std::setw(3)
       << static_cast<std::uint32_t>(host[i]);
  }
  ss << std::endl;
  std::cout << ss.str();

  delete[] host;
}

void *
CreateHostData(const std::size_t    size,
               std::uint64_t        seed,
               const std::ptrdiff_t offset) {
  static const std::uint64_t pn   = 1337;
  std::uint8_t *             data = new std::uint8_t[size];
  assert(nullptr != data);
  auto                it  = reinterpret_cast<std::uint64_t *>(data);
  const std::uint8_t *end = data + size;

  while (reinterpret_cast<std::uint8_t *>(it + 1) <= end) {
    *it  = seed;
    seed = (seed << 1) | (__builtin_parityl(seed & pn) & 1);
    ++it;
  }

  std::memcpy(
      it,
      &seed,
      static_cast<std::size_t>(end - reinterpret_cast<std::uint8_t *>(it)));

  void *buffer = malloc(size);

  std::memcpy(buffer, data + offset, size);

  delete[] data;

  return buffer;
}

void *
CreateData(const std::size_t    size,
           std::uint64_t        seed,
           const std::ptrdiff_t offset) {
  static const std::uint64_t pn   = 1337;
  std::uint8_t *             data = new std::uint8_t[size];
  assert(nullptr != data);
  auto                it  = reinterpret_cast<std::uint64_t *>(data);
  const std::uint8_t *end = data + size;

  while (reinterpret_cast<std::uint8_t *>(it + 1) <= end) {
    *it  = seed;
    seed = (seed << 1) | (__builtin_parityl(seed & pn) & 1);
    ++it;
  }

  std::memcpy(
      it,
      &seed,
      static_cast<std::size_t>(end - reinterpret_cast<std::uint8_t *>(it)));

  void *      buffer;
  cudaError_t cudaStatus = cudaMalloc(&buffer, size);
  assert(cudaSuccess == cudaStatus);

  cudaStatus = cudaMemcpy(buffer, data + offset, size, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStatus);

  delete[] data;

  return buffer;
}


void *
CreateGpuBuffer(const std::size_t    size) {
  void *      buffer;
  cudaError_t cudaStatus = cudaMalloc(&buffer, size);
  assert(cudaSuccess == cudaStatus);
  return buffer;
}

using namespace blazingdb::uc;

std::unique_ptr<Context> CreateUCXContext(std::string context_name){
  if (context_name == "tcp")
    return Context::TCP();
  else if (context_name == "ipc")
    return Context::IPC();
  else if (context_name == "gdr")
    return Context::GDR();
  else if (context_name == "view")
    return Context::IPCView();

  return nullptr;
}
