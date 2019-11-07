#include "blazingsql/api.hpp"
#include "boost/functional/hash.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/multiprecision/cpp_int.hpp"
#include <boost/process.hpp>
#include <gflags/gflags.h>
#include "blazingsql/utils/server_process.h"
#include "blazingdb/uc/API.hpp"
#include "blazingsql/api.hpp"
#include "constexpr_header.h"

DEFINE_string(server_host, "",
"An existing performance server to benchmark against (leave blank to spawn "
"one automatically)");
DEFINE_int32(server_port, 5555, "The port to connect to");
DEFINE_int32(records_per_stream, 1 << 14, "Total records per stream");
DEFINE_string(context, "tcp", "UCX Context");
DEFINE_int32(device_id_server, 0, "device_id_server");
DEFINE_int32(device_id_client, 0, "device_id_client");

Result<PerformanceResult, Status> RunDoGetTest(Client* client,  int device_id, std::vector<Message> & batches) {
  cudaSetDevice(device_id);
  using namespace blazingdb::uc;
  const int bytes_per_record = 32;
  int64_t num_bytes = 0;
  int64_t num_records = 0;
  for(auto& batch : batches) {
    const void *data = CreateData(BUFFER_LENGTH, ownSeed, ownOffset);
    auto context = CreateUCXContext(FLAGS_context);
    auto agent   = context->Agent();
    auto buffer  = agent->Register(data, BUFFER_LENGTH);

    auto buffer_descriptors_serialized = buffer->SerializedRecord();
    const uint8_t *buffer_descriptors = buffer_descriptors_serialized->Data();

    Message msg((const char*)buffer_descriptors, buffer_descriptors_serialized->Size());
    auto res = client->send(msg);
    cudaFree((void*)data);

    num_records += BUFFER_LENGTH;
    num_bytes += BUFFER_LENGTH * bytes_per_record;
  }
  return Ok(PerformanceResult{num_records, num_bytes});
}

auto ConsumeStream(PerformanceStats& stats, Client *client, int device_id, std::vector<Message>& batches) -> bool {
  // TODO(wesm): Use location from endpoint, same host/port for now
  const auto& result = RunDoGetTest(client, device_id, batches);
  if (result.isOk()) {
    const PerformanceResult& perf = result.unwrap();
    stats.Update(perf.num_records, perf.num_bytes);
  }
  //    return result;
  return true;
};

Result<bool, Status>  RunPerformanceTest(Client* client) {
  PerformanceStats stats;

  StopWatch timer;
  timer.Start();

  //  thread_pool pool;
  std::vector<Message> batches(FLAGS_records_per_stream, Message(""));

  auto num_concurrent_clients = 1;
  std::vector<std::thread> tasks;
  for (int index = 0; index < num_concurrent_clients; ++index) {
    //    tasks.emplace_back(pool.submit(ConsumeStream, batches));
    // TODO: change 0 by index
    tasks.emplace_back(std::thread(ConsumeStream, std::ref(stats), client, FLAGS_device_id_client, std::ref(batches)));
  }

  // Wait for tasks to finish
  for (auto&& task : tasks) {
    task.join();
  }

  // Elapsed time in seconds
  uint64_t elapsed_nanos = timer.Stop();
  double time_elapsed =
      static_cast<double>(elapsed_nanos) / static_cast<double>(1000000000);

//  constexpr double kMegabyte = static_cast<double>(1 << 20);
  constexpr double kMegabyte = static_cast<double>(1 << 10);

  // Check that number of rows read is as expected
//  if (stats.bandwidth != static_cast<int64_t>(bandwidth)) {
//    return Err(Status(StatusCode::Invalid, "Did not consume expected number of records"));
//  }
  std::cout << "Bytes read: " << stats.total_bytes << std::endl;
  std::cout << "Time: " << elapsed_nanos / 1000.0 / 1000.0 / 1000.0 << std::endl;
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
    server.reset(new TestServer("ucx_server", FLAGS_server_port, FLAGS_context, FLAGS_device_id_server));
    server->Start();
  } else {
    std::cout << "Using remote server: true" << std::endl;
    hostname = FLAGS_server_host;
  }

  std::cout << "Testing method: ";
  std::cout << "DoGet";
  std::cout << std::endl;

  std::cout << "Server host: " << hostname << std::endl
            << "Server port: " << FLAGS_server_port << std::endl;

  Client client("tcp://localhost:" + std::to_string(FLAGS_server_port), "[string]");

  auto s = RunPerformanceTest(&client);

  if (server) {
    server->Stop();
  }

  if (!s.isOk()) {
    std::cerr << "Failed with error: << " << s.unwrapErr().text << std::endl;
  }
  return 0;
}
