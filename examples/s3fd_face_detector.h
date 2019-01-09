#ifndef _S3FD_FACE_DETECTOR_H_
#define _S3FD_FACE_DETECTOR_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "net.h"

namespace rtfd {

	class S3FD_Detector {
	public:
		S3FD_Detector(const unsigned char param[], const unsigned char bin[],
			const int blob_in_, const std::vector<int>& blob_out_, const std::vector<int>& blob_tra,
			const int buf_siz = 128) : blob_input_idx(blob_in_), blob_output_idn(blob_out_) {

			this->fd_net.load_param(param);
			this->fd_net.load_model(bin);

			this->pred_bbox_lists.reserve(128);
			this->blob_track_idn[0] = blob_tra[0];
			this->blob_track_idn[1] = blob_tra[1];
		}

		int detect_faces(const cv::Mat& bgr, cv::Rect2f &top_face);
		int track_TopFace(const cv::Mat& last_img, cv::Rect2f &top_face);


	private:
		ncnn::Net fd_net;
		const int thread_nums = 2;
		cv::Size net_input_size = cv::Size(192, 256);
		const int track_squ_size = 56;
		const float mean_vals[3] = { 103.939f, 116.779f, 123.68f };
		const float scale_val[3] = { 1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f };

		// assigned with values from "***.id.h"
		int blob_input_idx;
		std::vector<int> blob_output_idn;
		int blob_track_idn[2];

		typedef struct Float5_ {
			float data[5];

			Float5_() : data{ 0 } { }
			Float5_(float a0, float a1, float a2, float a3, float a4) : data{ a0, a1, a2, a3, a4 } { }

			float& operator[] (int idx) { return data[idx]; }
			const float& operator[] (int idx) const { return data[idx]; }
		} Float5;
		std::vector<Float5> pred_bbox_lists; // x,y,w,h,p

	private:
		// helper functions
		int prepareInput(const cv::Mat &img_data, ncnn::Mat &net_in, float &min_scal_rate);
		void postProcessNetOut(ncnn::Extractor &ex, std::vector<Float5> &pred_bbox_lists_);
		void processNetLayerOut(ncnn::Extractor &ex, std::vector<Float5> &pred_bbox_lists_, const int cls_layer_name, const int reg_layer_name,
			int anchor_size, const cv::Size &input_data_size, int stride_ = -1, const float prob_thres = 0.85f);

		enum NMS_TYPE {
			NMS_MIN,
			NMS_UNION
		};
		void NonMaxSuppress(std::vector<Float5> &pred_bbox_lists_, const float max_overlap_ratio, NMS_TYPE type);
		void PickOutTheWantedFace(const cv::Rect2f& org_img_rect, std::vector<Float5> &pred_bbox_lists_, const float min_scal_rate, cv::Rect2f &top_face);

		void Rect2Square(const cv::Rect2f& in_rect, cv::Rect &out_squ, const float expand_rate);
	};

}






#endif