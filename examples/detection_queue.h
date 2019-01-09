#ifndef _DETECTION_QUEUE_H_
#define _DETECTION_QUEUE_H_

#include <opencv2/core/core.hpp>
#include <atomic>

namespace rtfd {

	//template<typename T=std::pair<cv::Mat, cv::Rect2f>>
	typedef std::pair<cv::Mat, cv::Rect2f> T;

	class DetectQueue {
	public:
		DetectQueue() { }
		DetectQueue(int max_siz) : max_size_(max_siz) {
			if (this->max_size_ < 1)
				this->max_size_ = 1;
			this->element_array = new T[max_size_];
		}
		~DetectQueue() {
			if (element_array) {
				delete[] element_array;
				element_array = nullptr;
			}
		}

		void clear() {
			size_ = 0;
			front_ = 0;
			back_ = -1;
			detect_idx_ = -1;
		}

		const int size() const { return this->size_; }
		const int front_idx() const { return this->front_; }
		const int back_idx() const { return this->back_; }
		const int detect_idx() const { return this->detect_idx_; }

		void reserve(int max_siz) {
			if (max_siz <= this->max_size_)
				return;
			delete[] element_array;
			this->clear();

			this->max_size_ = max_siz;
			this->element_array = new T[this->max_size_];
		}

		void push_back(const T& in_ele) {
			if (this->size_ == this->max_size_)
				return;
			++size_;
			int tmp_b = back_;
			if (++tmp_b == max_size_)
				tmp_b = 0;
			this->element_array[tmp_b] = in_ele;

			back_ = tmp_b;
		}

		void pop_front() {
			if (this->size_ == 0)
				return;
			--size_;
			if (++front_ == max_size_)
				front_ = 0;
		}

		T& get_front() {
			return this->element_array[front_];
		}

		T& get_detect() {
			return this->element_array[detect_idx_];
		}

		bool detectNextNow() {
			if (detect_idx_ == back_)
				return false;
			int tmp_d = detect_idx_;
			if (++detect_idx_ == max_size_)
				detect_idx_ = 0;
			if (tmp_d != -1)
				this->element_array[detect_idx_].second = this->element_array[tmp_d].second;
			return true;
		}

		bool showFrontNow() {
			return (detect_idx_ == front_ || detect_idx_ == -1) ? false : true;
		}

		bool isFull() {
			return (size_ == max_size_);
		}

		T& operator[](int idx_) {
			if (idx_ < 0)
				idx_ += max_size_;
			else if (idx_ >= max_size_)
				idx_ -= max_size_;

			return this->element_array[idx_];
		}


	private:
		int max_size_ = 5;
		std::atomic<int> size_{ 0 };
		std::atomic<int> front_{ 0 };
		std::atomic<int> back_{ -1 };
		std::atomic<int> detect_idx_{ -1 };
		T* element_array = new T[max_size_];
	};

}


#endif
