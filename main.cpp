#include "blazingsql/api.hpp"
#include "boost/functional/hash.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/multiprecision/cpp_int.hpp"
#include <boost/process.hpp>
#include <gflags/gflags.h>
#include "blazingsql/utils/server_process.h"

DEFINE_string(server_host, "",
"An existing performance server to benchmark against (leave blank to spawn "
"one automatically)");
DEFINE_int32(server_port, 5555, "The port to connect to");
DEFINE_int32(num_servers, 1, "Number of performance servers to run");
DEFINE_int32(num_streams, 4, "Number of streams for each server");
DEFINE_int32(num_threads, 4, "Number of concurrent gets");
DEFINE_int32(records_per_stream, 10000000, "Total records per stream");
DEFINE_int32(records_per_batch, 4096, "Total records per batch within stream");
DEFINE_bool(test_put, false, "Test DoPut instead of DoGet");

Result<PerformanceResult, Status> RunDoGetTest(Client<String>* client,  std::vector<String>& batches) {
  // This is hard-coded for right now, 4 columns each with int64
  const int bytes_per_record = 32;
  int64_t num_bytes = 0;
  int64_t num_records = 0;
  for(auto& batch : batches) {
    if (client->request(batch )) {
      std::cout << "Received a string: " << batch.size() << std::endl;

      num_records += batch.size();
      num_bytes += batch.size() * bytes_per_record;
    }
  }
  return Ok(PerformanceResult{num_records, num_bytes});
}

Result<bool, Status>  RunPerformanceTest(Client<String>* client, bool test_put) {
  PerformanceStats stats;
  auto test_loop = RunDoGetTest;

  auto ConsumeStream = [&stats, &test_loop, &client](std::vector<String>& batches) {
    // TODO(wesm): Use location from endpoint, same host/port for now
    const auto& result = RunDoGetTest(client, batches);
    if (result.isOk()) {
      const PerformanceResult& perf = result.unwrap();
      stats.Update(perf.num_records, perf.num_bytes);
    }
    return result;
  };
  const int total_records = 1 << 10;
  StopWatch timer;
  timer.Start();
//
//  thread_pool pool;
//  std::vector<std::future<Status>> tasks;
//  for (const auto& endpoint : plan->endpoints()) {
//    tasks.emplace_back(pool->Submit(ConsumeStream, endpoint));
//  }
//
//  // Wait for tasks to finish
//  for (auto&& task : tasks) {
//    task.get();
//  }
  std::string s("*", total_records);
  std::vector<String> batches(total_records, String(s));
  ConsumeStream (batches);

  // Elapsed time in seconds
  uint64_t elapsed_nanos = timer.Stop();
  double time_elapsed =
      static_cast<double>(elapsed_nanos) / static_cast<double>(1000000000);

  constexpr double kMegabyte = static_cast<double>(1 << 20);

  // Check that number of rows read is as expected
//  if (stats.total_records != static_cast<int64_t>(total_records)) {
//    return Err(Status(StatusCode::Invalid, "Did not consume expected number of records"));
//  }
  std::cout << "Bytes read: " << stats.total_bytes << std::endl;
  std::cout << "Nanos: " << elapsed_nanos << std::endl;
  std::cout << "Speed: "
            << (static_cast<double>(stats.total_bytes) / kMegabyte / time_elapsed)
            << " MB/s" << std::endl;
  return Ok(true);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::unique_ptr<TestServer> server;
  std::string hostname = "localhost";
  if (FLAGS_server_host == "") {
    std::cout << "Using remote server: false" << std::endl;
    server.reset(new TestServer("rpc_server", FLAGS_server_port));
    server->Start();
  } else {
    std::cout << "Using remote server: true" << std::endl;
    hostname = FLAGS_server_host;
  }

  std::cout << "Testing method: ";
  if (FLAGS_test_put) {
    std::cout << "DoPut";
  } else {
    std::cout << "DoGet";
  }
  std::cout << std::endl;

  std::cout << "Server host: " << hostname << std::endl
            << "Server port: " << FLAGS_server_port << std::endl;

  std::unique_ptr<Client<String>> client = std::make_unique<Client<String>>("tcp://localhost:" + std::to_string(FLAGS_server_port));

  auto s = RunPerformanceTest(client.get(), FLAGS_test_put);

  if (server) {
    server->Stop();
  }

  if (!s.isOk()) {
    std::cerr << "Failed with error: << " << s.unwrapErr().text << std::endl;
  }

  return 0;
}
