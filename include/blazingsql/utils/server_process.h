#pragma once

#include "boost/functional/hash.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/multiprecision/cpp_int.hpp"
#include <boost/process.hpp>

namespace bp = boost::process;
namespace fs = boost::filesystem;

namespace {

Result<bool, Status> ResolveCurrentExecutable(fs::path* out) {
  boost::system::error_code ec;
  *out = fs::canonical("/proc/self/exe", ec);
  if (ec) {
    // XXX fold this into the Status class?
    return Err(Status(StatusCode::IOError, "Can't resolve current exe: " + ec.message()));
  } else {
    return Ok(true);
  }
}

}  // namespace

class TestServer {
public:
  explicit TestServer(const std::string& executable_name, int port)
      : executable_name_(executable_name), port_(port) {}

  void Start() {
    namespace fs = boost::filesystem;

    std::string str_port = std::to_string(port_);
    std::vector<fs::path> search_path = ::boost::this_process::path();
    fs::path current_exe;
    auto st = ResolveCurrentExecutable(&current_exe);
    if (st.isOk()) {
      search_path.insert(search_path.begin(), current_exe.parent_path());
    } else  {
      throw std::runtime_error("not implemented");
    }

    try {
      server_process_ = std::make_shared<bp::child>(bp::search_path(executable_name_, search_path), "-port", str_port);
    } catch (...) {
      std::stringstream ss;
      ss << "Failed to launch test server '" << executable_name_ << "', looked in ";
      for (const auto& path : search_path) {
        ss << path << " : ";
      }
      std::cerr << ss.str();
      throw;
    }
    std::cout << "Server running with pid " << server_process_->id() << std::endl;
  }

  int Stop() {
    if (server_process_ && server_process_->valid()) {
      server_process_->terminate();
      server_process_->wait();
      return server_process_->exit_code();
    } else {
      // Presumably the server wasn't able to start
      return -1;
    }
  }

  bool IsRunning() { return server_process_->running(); }

  int port() const { return port_; }

private:
  Server<String> server;
  std::string executable_name_;
  int port_;
  std::shared_ptr<boost::process::child> server_process_;
};
