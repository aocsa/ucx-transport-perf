#pragma once

class StopWatch {
  // This clock should give us wall clock time
  using ClockType = std::chrono::steady_clock;

public:
  StopWatch() {}

  void Start() { start_ = ClockType::now(); }

  // Returns time in nanoseconds.
  uint64_t Stop() {
    auto stop = ClockType::now();
    std::chrono::nanoseconds d = stop - start_;
    assert(d.count() >= 0);
    return static_cast<uint64_t>(d.count());
  }

private:
  std::chrono::time_point<ClockType> start_;
};

struct PerformanceResult {
  int64_t num_records;
  int64_t num_bytes;
};

struct PerformanceStats {
  PerformanceStats() : total_records(0), total_bytes(0) {}
  std::mutex mutex;
  int64_t total_records;
  int64_t total_bytes;

  void Update(const int64_t total_records, const int64_t total_bytes) {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->total_records += total_records;
    this->total_bytes += total_bytes;
  }
};


struct PerformanceTimeStats {
  PerformanceTimeStats() : link_total_time(0), total_bytes(0) {}
  std::mutex mutex;
  double link_total_time;
  int64_t total_bytes;

  void Update(const double total_time, const int64_t total_bytes) {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->link_total_time += total_time;
    this->total_bytes += total_bytes;
  }
  void Reset() {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->link_total_time = 0;
    this->total_bytes = 0;
  }
};