#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include "blazingsql/utils/result.h"
#include "join_threads.hpp"
#include "threadsafe_queue.hpp"
#include <atomic>
#include <functional>
#include <future>
#include <thread>
#include <vector>

class thread_pool {
private:
    std::atomic_bool done;
    threadsafe_queue<std::function<bool()>> work_queue;
    std::vector<std::thread> threads;
    join_threads joiner;
    size_t thread_count_;

    void worker_thread() {
        while (!done) {
            std::function<bool()> task;
            if (work_queue.try_pop(task)) {
                task();
            } else {
                std::this_thread::yield();
            }
        }
    }

public:
    thread_pool(): done(false), joiner(threads) {
      thread_count_ = std::thread::hardware_concurrency();

        try {
            for (unsigned i = 0; i < thread_count_; ++i) {
                threads.push_back(std::thread(&thread_pool::worker_thread, this));
            }
        } catch(...) {
            done = true;
            throw;
        }
    }
    size_t thread_count(){
      return thread_count_;
    }

    ~thread_pool() {
        done = true;
    }

    template <typename Function, typename... Args,
    typename ResultType = typename std::result_of<Function && (Args && ...)>::type>

    std::future<ResultType>  submit(Function&& func, Args&&... args) {

        using PackagedTask = std::packaged_task<ResultType()>;
        auto task = PackagedTask(std::bind(std::forward<Function>(func), args...));
        auto fut = task.get_future();

        work_queue.push(std::move(task));

        return fut;
    }
};

#endif
