#include "s3fd_face_detector.h"
#include <cmath>

namespace rtfd {

	int S3FD_Detector::detect_faces(const cv::Mat& bgr, cv::Rect2f &top_face)
	{
		int re_code = 0;

		ncnn::Mat net_in;
		float min_scal_rate;
		re_code = this->prepareInput(bgr, net_in, min_scal_rate);
		if (re_code)
			return re_code;

		ncnn::Extractor fd_ex = this->fd_net.create_extractor();
		fd_ex.set_light_mode(true);
		fd_ex.set_num_threads(this->thread_nums);
		re_code = fd_ex.input(this->blob_input_idx, net_in);
		if (re_code)
			return re_code;

		this->pred_bbox_lists.clear();
		this->postProcessNetOut(fd_ex, this->pred_bbox_lists);

		PickOutTheWantedFace(cv::Rect2f(0, 0, bgr.cols, bgr.rows), this->pred_bbox_lists, min_scal_rate, top_face);

		return re_code;
	}

	int S3FD_Detector::track_TopFace(const cv::Mat& last_img, cv::Rect2f &top_face)
	{
		if (top_face.area() < 1.0e-3)
			return -1;

		if (last_img.empty())
			return -1;

		cv::Mat bgr = last_img;
		if (last_img.channels() == 4)
			cv::cvtColor(last_img, bgr, cv::COLOR_BGRA2BGR);
		else if (last_img.channels() == 1)
			cv::cvtColor(last_img, bgr, cv::COLOR_GRAY2BGR);

		cv::Rect squ_rec;
		Rect2Square(top_face, squ_rec, float(this->track_squ_size) / 32.0f); //1.66667f
		cv::Mat crop_img(squ_rec.size(), CV_8UC3, cv::Scalar(0));

		const cv::Rect org_img_rect(0, 0, bgr.cols, bgr.rows);
		cv::Rect temp = org_img_rect & squ_rec;
		cv::Mat overlap_mat(bgr, temp);
		temp -= squ_rec.tl();
		overlap_mat.copyTo(crop_img(temp));

		ncnn::Mat net_in = ncnn::Mat::from_pixels_resize(crop_img.data, ncnn::Mat::PIXEL_BGR, crop_img.cols, crop_img.rows, this->track_squ_size, this->track_squ_size);
		net_in.substract_mean_normalize(this->mean_vals, this->scale_val);

		ncnn::Extractor track_ex = this->fd_net.create_extractor();
		track_ex.set_light_mode(true);
		track_ex.set_num_threads(this->thread_nums);
		int re_code = track_ex.input(this->blob_input_idx, net_in);
		if (re_code)
			return re_code;

		// anchor_size: 32 * 32
		this->pred_bbox_lists.clear();
		processNetLayerOut(track_ex, this->pred_bbox_lists, this->blob_track_idn[0], this->blob_track_idn[1], 32, cv::Size(this->track_squ_size, this->track_squ_size), 8, 0.85f);
		this->NonMaxSuppress(this->pred_bbox_lists, 0.5f, NMS_TYPE::NMS_UNION);

		if (this->pred_bbox_lists.empty()) {
			//top_face.x = top_face.y = top_face.width = top_face.height = 0.0f;
			return -1;
		}
		float rescal_rate = float(squ_rec.width) / float(this->track_squ_size);
		top_face.x = this->pred_bbox_lists[0][0] * rescal_rate + squ_rec.x;
		top_face.y = this->pred_bbox_lists[0][1] * rescal_rate + squ_rec.y;
		top_face.width = this->pred_bbox_lists[0][2] * rescal_rate;
		top_face.height = this->pred_bbox_lists[0][3] * rescal_rate;

		return re_code;
	}



	// helper functions
	int S3FD_Detector::prepareInput(const cv::Mat &img_data, ncnn::Mat &net_in, float &min_scal_rate)
	{
		if (img_data.empty())
			return -1;

		cv::Mat img_data_bgr = img_data;
		if (img_data.channels() == 4)
			cv::cvtColor(img_data, img_data_bgr, cv::COLOR_BGRA2BGR);
		else if (img_data.channels() == 1)
			cv::cvtColor(img_data, img_data_bgr, cv::COLOR_GRAY2BGR);

		int img_height = img_data.rows;
		int img_width = img_data.cols;
		min_scal_rate = std::min(float(this->net_input_size.height) / img_height, float(this->net_input_size.width) / img_width);
		int scal_img_height = std::min(this->net_input_size.height, (int)(0.5f + min_scal_rate * img_height));
		int scal_img_width = std::min(this->net_input_size.width, (int)(0.5f + min_scal_rate * img_width));
		cv::Mat cv_img_scal(scal_img_height, scal_img_width, CV_8UC3);

		ncnn::resize_bilinear_c3(img_data_bgr.data, img_width, img_height, cv_img_scal.data, scal_img_width, scal_img_height);

		cv::Mat cv_img_norm = cv::Mat::zeros(this->net_input_size, CV_8UC3);
		cv_img_scal.copyTo(cv_img_norm(cv::Rect(0, 0, scal_img_width, scal_img_height)));

		net_in = ncnn::Mat::from_pixels(cv_img_norm.data, ncnn::Mat::PIXEL_BGR, this->net_input_size.width, this->net_input_size.height);
		net_in.substract_mean_normalize(this->mean_vals, this->scale_val);

		return 0;
	}

	void S3FD_Detector::postProcessNetOut(ncnn::Extractor &ex, std::vector<Float5> &pred_bbox_lists_) {
		// anchor_size: 256 * 256
		processNetLayerOut(ex, pred_bbox_lists_, this->blob_output_idn[6], this->blob_output_idn[7], 256, this->net_input_size);

		// anchor_size: 128 * 128
		processNetLayerOut(ex, pred_bbox_lists_, this->blob_output_idn[4], this->blob_output_idn[5], 128, this->net_input_size);

		// anchor_size: 64 * 64
		processNetLayerOut(ex, pred_bbox_lists_, this->blob_output_idn[2], this->blob_output_idn[3], 64, this->net_input_size);

		// anchor_size: 32 * 32
		processNetLayerOut(ex, pred_bbox_lists_, this->blob_output_idn[0], this->blob_output_idn[1], 32, this->net_input_size);

		this->NonMaxSuppress(pred_bbox_lists_, 0.5f, NMS_TYPE::NMS_UNION);
	}

	void S3FD_Detector::processNetLayerOut(ncnn::Extractor &ex, std::vector<Float5> &pred_bbox_lists_, const int cls_layer_name, const int reg_layer_name,
		int anchor_size, const cv::Size &input_data_size, int stride_, const float prob_thres) {
		if (stride_ < 0)
			stride_ = anchor_size / 4;

		ncnn::Mat cls_blob_ptr, reg_blob_ptr;
		ex.extract(cls_layer_name, cls_blob_ptr);
		ex.extract(reg_layer_name, reg_blob_ptr);

		int blob_h = cls_blob_ptr.h;
		int blob_w = cls_blob_ptr.w;
		int padding_h = (stride_ * (blob_h - 1) + anchor_size - input_data_size.height) / 2;
		int padding_w = (stride_ * (blob_w - 1) + anchor_size - input_data_size.width) / 2;

		float* cls_data = cls_blob_ptr.channel(1);
		float* reg_data[4] = { reg_blob_ptr.channel(0), reg_blob_ptr.channel(1), reg_blob_ptr.channel(2), reg_blob_ptr.channel(3) };
		cv::Mat cls_mat(blob_h, blob_w, CV_32FC1, cls_data);

		auto anchor_func = [&](int idx_kth, int proc_nums) -> void {
			for (int i = 0; i < proc_nums; ++i) {
				int reg_offset = idx_kth + i;
				int row_i = reg_offset / cls_mat.cols;
				int col_i = reg_offset - row_i * cls_mat.cols;

				float prob_tmp = cls_mat.at<float>(row_i, col_i);
				if (prob_tmp > prob_thres) {
					cv::Rect2f cur_anchor(0.0f, 0.0f, float(anchor_size), float(anchor_size));
					cur_anchor.x = stride_ * col_i - padding_w + anchor_size / 2;
					cur_anchor.y = stride_ * row_i - padding_h + anchor_size / 2;

					cur_anchor.x += *(reg_data[0] + reg_offset) * cur_anchor.width;
					cur_anchor.y += *(reg_data[1] + reg_offset) * cur_anchor.height;
					cur_anchor.width *= expf(*(reg_data[2] + reg_offset));
					cur_anchor.height *= expf(*(reg_data[3] + reg_offset));
					cur_anchor.x -= 0.5f * cur_anchor.width;
					cur_anchor.y -= 0.5f * cur_anchor.height;

					pred_bbox_lists_.emplace_back(cur_anchor.x, cur_anchor.y, cur_anchor.width, cur_anchor.height, prob_tmp);
				}
			}
		};

		anchor_func(0, (int)cls_mat.total());
	}

	void S3FD_Detector::NonMaxSuppress(std::vector<Float5> &pred_bbox_lists_, const float max_overlap_ratio, NMS_TYPE type)
	{
		std::sort(pred_bbox_lists_.begin(), pred_bbox_lists_.end(), [](const Float5 &A, const Float5 &B) { return A[4] > B[4]; });
		int i, j, res_len;
		int lens = pred_bbox_lists_.size();
		cv::Rect2f overlap_rect_;
		float overlap_ratio_;

		auto bbox_cum_func = [](const Float5 &add_bbox_, Float5 &cum_) -> void {
			for (int n = 0; n < 5; ++n)
				cum_[n] += add_bbox_[n];
		};

		auto bbox_avg_func = [](const Float5 &cum_bbox_, const int N, Float5 &avg_bbox_) -> void {
			if (N < 2)
				return;
			for (int n = 0; n < 5; ++n)
				avg_bbox_[n] = cum_bbox_[n] / N;
		};

		for (i = 0; i < lens; lens = i + res_len, ++i) {
			auto cum_bbox_ = pred_bbox_lists_[i];
			int avg_N = 1;
			cv::Rect2f rect_i(cum_bbox_[0], cum_bbox_[1], cum_bbox_[2], cum_bbox_[3]);
			for (j = i + 1, res_len = 1; j < lens; ++j)
			{
				cv::Rect2f rect_j(pred_bbox_lists_[j][0], pred_bbox_lists_[j][1], pred_bbox_lists_[j][2], pred_bbox_lists_[j][3]);
				overlap_rect_ = rect_i & rect_j;
				switch (type) {
				case NMS_TYPE::NMS_MIN:
					overlap_ratio_ = overlap_rect_.area() / std::min(rect_i.area(), rect_j.area()); break;
				case NMS_TYPE::NMS_UNION:
					overlap_ratio_ = overlap_rect_.area() / (rect_i.area() + rect_j.area() - overlap_rect_.area()); break;
				default:
					printf("Undefined NMS_TYPE!\n"); break;
				}
				if (overlap_ratio_ <= max_overlap_ratio)
					std::swap(pred_bbox_lists_[i + res_len++], pred_bbox_lists_[j]);
				else if (avg_N < 8 && overlap_ratio_ > 1.2f * max_overlap_ratio) {
					++avg_N;
					bbox_cum_func(pred_bbox_lists_[j], cum_bbox_);
				}
			}
			bbox_avg_func(cum_bbox_, avg_N, pred_bbox_lists_[i]);
		}

		pred_bbox_lists_.resize(lens);
		//pred_bbox_lists_.shrink_to_fit();
	}

	void S3FD_Detector::PickOutTheWantedFace(const cv::Rect2f& org_img_rect, std::vector<Float5> &pred_bbox_lists_, const float min_scal_rate, cv::Rect2f &top_face)
	{
		if (pred_bbox_lists_.empty()) {
			top_face.x = top_face.y = top_face.width = top_face.height = 0.0f;
			return;
		}

		for (int k = 0; k < (int)pred_bbox_lists_.size(); ++k) {
			pred_bbox_lists_[k][0] /= min_scal_rate;
			pred_bbox_lists_[k][1] /= min_scal_rate;
			pred_bbox_lists_[k][2] /= min_scal_rate;
			pred_bbox_lists_[k][3] /= min_scal_rate;

			// 1. throw away face rects partial outside the image.
			// TO DO
		}

		const float x_weight = 1.25f;  // 0~2.0f
		cv::Point2f org_img_center(0.5f * org_img_rect.width, 0.5f * org_img_rect.height);	
		float max_dist_off_center = sqrtf(x_weight * org_img_center.x * org_img_center.x + (2.0f - x_weight) * org_img_center.y * org_img_center.y);
		// 2. sort by bbox areas and offsets to the image center.
		auto wanted_rect_ = *std::max_element(pred_bbox_lists_.begin(), pred_bbox_lists_.end(),
			[&](const Float5 &A, const Float5 &B)
		{
			cv::Point2f A_offset(A[0] + 0.5f * A[2] - org_img_center.x, A[1] + 0.5f * A[3] - org_img_center.y);
			cv::Point2f B_offset(B[0] + 0.5f * B[2] - org_img_center.x, B[1] + 0.5f * B[3] - org_img_center.y);
			float A_off_rate = sqrtf(x_weight * A_offset.x * A_offset.x + (2.0f - x_weight) * A_offset.y * A_offset.y) / max_dist_off_center;
			float B_off_rate = sqrtf(x_weight * B_offset.x * B_offset.x + (2.0f - x_weight) * B_offset.y * B_offset.y) / max_dist_off_center;

			return (1.0f + 5.0f * A_off_rate * A_off_rate) * 2.0f / (A[2] + A[3]) / A[4]  > (1.0f + 5.0f * B_off_rate * B_off_rate) * 2.0f / (B[2] + B[3]) / B[4];
		});

		top_face.x = wanted_rect_[0];
		top_face.y = wanted_rect_[1];
		top_face.width = wanted_rect_[2];
		top_face.height = wanted_rect_[3];
	}

	void S3FD_Detector::Rect2Square(const cv::Rect2f& in_rect, cv::Rect &out_squ, const float expand_rate)
	{
		cv::Point2f center_pt;
		center_pt.x = in_rect.x + 0.5f * in_rect.width;
		center_pt.y = in_rect.y + 0.5f * in_rect.height;

		out_squ.width = out_squ.height = 0.5f * expand_rate * (in_rect.width + in_rect.height) + 0.5f;
		//out_squ.width = out_squ.height = expand_rate * std::max(in_rect.width, in_rect.height) + 0.5f;
		out_squ.x = center_pt.x - 0.5f * out_squ.width;
		out_squ.y = center_pt.y - 0.5f * out_squ.height;
	}

}

