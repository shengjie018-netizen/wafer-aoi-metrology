#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace wafer_aoi {

// Modern C++17 ThreadPool for multi-frame / multi-die inspection pipelines
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        if (num_threads == 0) num_threads = 2;
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<ReturnType> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return res;
    }

    size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Bounded thread-safe queue for Producer-Consumer camera-to-inspection pipelines
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return closed_ || queue_.size() < capacity_; });
        if (closed_) return false;
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    void close() {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    size_t capacity_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
};

}  // namespace wafer_aoi
