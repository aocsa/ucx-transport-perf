#ifndef _JOIN_THREADS_H_
#define _JOIN_THREADS_H_

#include <vector>
#include <thread>

class join_threads {
private:
    std::vector<std::thread>& threads_;

public:
    explicit join_threads(std::vector<std::thread>& threads): threads_(threads) {}
    ~join_threads() {
        for (unsigned long i = 0; i < threads_.size(); ++i) {
            if (threads_[i].joinable())
                threads_[i].join();
        }
    }
};

#endif
