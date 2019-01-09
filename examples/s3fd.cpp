// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <iostream>
//#include <algorithm>
#include <vector>
//#include <deque>
#include <thread>
//#include <atomic>
//#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "net.h"
#include "benchmark.h"
#include "smooth_face_bbox.h"
#include "detection_queue.h"

// global variables
ncnn::Net fd_net;
ncnn::Net track_net;
//std::deque<std::pair<cv::Mat, std::vector<std::pair<cv::Rect2f, float>>>> face_buf;
rtfd::DetectQueue face_buf;
cv::VideoCapture cap(0);
float avg_speed = 40.0f; // ms
//std::atomic<int> detect_idx(0);
//std::mutex mtx;           // mutex for critical section
rtfd::FaceBboxSmoother my_smoother;

typedef struct Float5_ {
	float data[5];

	Float5_() : data{0} { }
	Float5_(float a0, float a1, float a2, float a3, float a4) : data{ a0, a1, a2, a3, a4 } { }

	float& operator[] (int idx) { return data[idx]; }
	const float& operator[] (int idx) const { return data[idx]; }
} Float5;
std::vector<Float5> pred_bbox_lists; // x,y,w,h,p



void InitModels(const std::string& fd_param_file, const std::string& fd_bin_file, const std::string& tra_param_file, const std::string& tra_bin_file)
{
	::fd_net.load_param(fd_param_file.c_str());
	::fd_net.load_model(fd_bin_file.c_str());

	::track_net.load_param(tra_param_file.c_str());
	::track_net.load_model(tra_bin_file.c_str());

	::pred_bbox_lists.reserve(128);
}

static int prepareInput(const cv::Mat &img_data, ncnn::Mat &net_in, float &min_scal_rate)
{
	int img_height = img_data.rows;
	int img_width = img_data.cols;

	cv::Mat img_data_bgr = img_data;
	if (img_data.channels() == 4)
		cv::cvtColor(img_data, img_data_bgr, cv::COLOR_RGBA2BGR);
	else if (img_data.channels() == 1)
		cv::cvtColor(img_data, img_data_bgr, cv::COLOR_GRAY2BGR);

	min_scal_rate = std::min(256.0f / img_height, 256.0f / img_width);
	int scal_img_height = std::min(256.0f, (float)round(min_scal_rate * img_height));
	int scal_img_width = std::min(256.0f, (float)round(min_scal_rate * img_width));
	cv::Mat cv_img_scal(scal_img_height, scal_img_width, CV_8UC3);

	ncnn::resize_bilinear_c3(img_data_bgr.data, img_width, img_height, cv_img_scal.data, scal_img_width, scal_img_height);

	cv::Mat cv_img_norm = cv::Mat::zeros(256, 256, CV_8UC3);
	cv_img_scal.copyTo(cv_img_norm(cv::Rect(0, 0, scal_img_width, scal_img_height)));

	net_in = ncnn::Mat::from_pixels(cv_img_norm.data, ncnn::Mat::PIXEL_BGR, 256, 256);

	const float mean_vals[3] = { 103.939f, 116.779f, 123.68f };
	const float scale_val[3] = { 1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f };
	net_in.substract_mean_normalize(mean_vals, scale_val);

	return 0;
}

enum NMS_TYPE {
	NMS_MIN,
	NMS_UNION
};

void NonMaxSuppress(std::vector<Float5> &pred_bbox_lists_, const float max_overlap_ratio, NMS_TYPE type) {
	std::sort(pred_bbox_lists_.begin(), pred_bbox_lists_.end(), [](const Float5 &A, const Float5 &B) { return A[4] > B[4]; });
	int i, j, res_len;
	int lens = pred_bbox_lists_.size();
	cv::Rect2f overlap_rect_, rect_i, rect_j;
	float overlap_ratio_;

	auto bbox_cum_func = [](const Float5 &add_bbox_, Float5 &cum_) -> void {
		for (int i = 0; i < 5; ++i)
			cum_[i] += add_bbox_[i];
	};

	auto bbox_avg_func = [](const Float5 &cum_bbox_, const int N, Float5 &avg_bbox_) -> void {
		if (N < 2)
			return;
		for (int i = 0; i < 5; ++i)
			avg_bbox_[i] = cum_bbox_[i] / N;
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


void processNetLayerOut(ncnn::Extractor &ex, std::vector<Float5> &pred_bbox_lists_, const std::string& cls_layer_name, const std::string& reg_layer_name, int anchor_size, const cv::Size &input_data_size, int stride_ = -1) {
	if(stride_ < 0)
		stride_ = anchor_size / 4;
	
	ncnn::Mat cls_blob_ptr, reg_blob_ptr;
	ex.extract(cls_layer_name.c_str(), cls_blob_ptr);
	ex.extract(reg_layer_name.c_str(), reg_blob_ptr);

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
			if (prob_tmp > 0.85f) {
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

void postProcessNetOut(ncnn::Extractor &ex, std::vector<Float5> &pred_bbox_lists_) {
	cv::Size input_data_size(256, 256);
	// anchor_size: 32 * 32
	processNetLayerOut(ex, pred_bbox_lists_, "detect_4_3_cls_softmax_4_3", "detect_4_3_reg", 32, input_data_size);

	// anchor_size: 64 * 64
	processNetLayerOut(ex, pred_bbox_lists_, "detect_5_3_cls_softmax_5_3", "detect_5_3_reg", 64, input_data_size);

	// anchor_size: 128 * 128
	processNetLayerOut(ex, pred_bbox_lists_, "detect_fc7_cls_softmax_fc7", "detect_fc7_reg", 128, input_data_size);

	// anchor_size: 256 * 256
	processNetLayerOut(ex, pred_bbox_lists_, "detect_6_2_cls_softmax_6_2", "detect_6_2_reg", 256, input_data_size);

	NonMaxSuppress(pred_bbox_lists_, 0.5f, NMS_TYPE::NMS_UNION);
}

void PickOutTheWantedFace(const cv::Rect2f& org_img_rect, std::vector<Float5> &pred_bbox_lists_, const float min_scal_rate, cv::Rect2f &top_face)
{
	if (pred_bbox_lists_.empty()) {
		top_face.x = top_face.y = top_face.width = top_face.height = 0.0f;
		return;
	}

	for (int k = 0; k < (int)pred_bbox_lists_.size(); ++ k) {
		pred_bbox_lists_[k][0] /= min_scal_rate;
		pred_bbox_lists_[k][1] /= min_scal_rate;
		pred_bbox_lists_[k][2] /= min_scal_rate;
		pred_bbox_lists_[k][3] /= min_scal_rate;

		// 1. throw away face rects partial outside the image.
		// TO DO
	}

	cv::Point2f org_img_center(0.5f * org_img_rect.width, 0.5f * org_img_rect.height);
	float max_dist_off_center = cv::norm(org_img_center);
	// 2. sort by bbox areas and offsets to the image center.
	auto wanted_rect_ = *std::max_element(pred_bbox_lists_.begin(), pred_bbox_lists_.end(),
		[&](const Float5 &A, const Float5 &B)
	{
		cv::Point2f A_center(A[0] + 0.5f * A[2], A[1] + 0.5f * A[3]);
		cv::Point2f B_center(B[0] + 0.5f * B[2], B[1] + 0.5f * B[3]);
		float A_off_rate = cv::norm(A_center - org_img_center) / max_dist_off_center;
		float B_off_rate = cv::norm(B_center - org_img_center) / max_dist_off_center;

		return (1.0f + 5.0f * A_off_rate * A_off_rate) * 2.0f / (A[2] + A[3]) / A[4]  < (1.0f + 5.0f * B_off_rate * B_off_rate) * 2.0f / (B[2] + B[3]) / B[4];
	});

	top_face.x = wanted_rect_[0];
	top_face.y = wanted_rect_[1];
	top_face.width = wanted_rect_[2];
	top_face.height = wanted_rect_[3];
}

static int detect_faces(const cv::Mat& bgr, cv::Rect2f &top_face)
{
	ncnn::Mat net_in;
	float min_scal_rate;
	prepareInput(bgr, net_in, min_scal_rate);

	ncnn::Extractor fd_ex = ::fd_net.create_extractor();
	fd_ex.set_light_mode(true);
	fd_ex.set_num_threads(4);
	fd_ex.input("data", net_in);

	//std::vector<std::pair<cv::Rect2f, float>> pred_bbox_lists;
	::pred_bbox_lists.clear();
	postProcessNetOut(fd_ex, ::pred_bbox_lists);

	PickOutTheWantedFace(cv::Rect2f(0, 0, bgr.cols, bgr.rows), ::pred_bbox_lists, min_scal_rate, top_face);

    return 0;
}
//====================================face tracker=========================================================
const float BBOX_UP_SCALING_RATIO = 1.667f;

// 调整boundingbox的长宽及位置
void BbregOnly(const cv::Vec4f &ceff_vec, std::pair<cv::Rect2f, float> &result_bbox) {
	cv::Rect2f &rect_f = result_bbox.first;
	rect_f.x += rect_f.width * ceff_vec[1];
	rect_f.y += rect_f.height * ceff_vec[0];
	rect_f.width *= 1.0f + ceff_vec[3] - ceff_vec[1];
	rect_f.height *= 1.0f + ceff_vec[2] - ceff_vec[0];
}

void enlargeBbox(const cv::Rect &org_img_rect, const std::vector<std::pair<cv::Rect2f, float>> &pred_bbox_lists, std::vector<cv::Rect> &result, const float enlarge_ratio)
{
	result.resize(pred_bbox_lists.size());
	for (int k = 0; k < result.size(); ++k) {
		int x_offset = int(0.5f * (enlarge_ratio - 1.0f) * pred_bbox_lists[k].first.width);
		int y_offset = int(0.5f * (enlarge_ratio - 1.0f) * pred_bbox_lists[k].first.height);

		result[k].x = pred_bbox_lists[k].first.x - x_offset;
		result[k].y = pred_bbox_lists[k].first.y - y_offset;
		result[k].width = int(enlarge_ratio * pred_bbox_lists[k].first.width);
		result[k].height = int(enlarge_ratio * pred_bbox_lists[k].first.height);
		result[k] &= org_img_rect;
	}
	return;
}

void TransposeAndRerec(std::vector<cv::Rect> &bbox) {
	for (auto &rect : bbox) {
		std::swap(rect.x, rect.y);
		std::swap(rect.width, rect.height);

		int max_dim = std::max(rect.width, rect.height);
		rect.x += 0.5f * (rect.width - max_dim + 1);
		rect.y += 0.5f * (rect.height - max_dim + 1);
		rect.width = rect.height = max_dim;
	}
}

cv::Mat inputdataPreprocess(const cv::Mat& org_img) {
	cv::Mat org_img_norm;
	//org_img.convertTo(org_img_norm, CV_32F);
	//org_img_norm = (org_img_norm - 127.5f) * 0.0078125f;
	if (org_img.channels() == 4)
		cv::cvtColor(org_img, org_img_norm, cv::COLOR_BGRA2RGB);
	else if (org_img.channels() == 1)
		cv::cvtColor(org_img, org_img_norm, cv::COLOR_GRAY2RGB);
	else
		cv::cvtColor(org_img, org_img_norm, cv::COLOR_BGR2RGB);
	org_img_norm = org_img_norm.t();  // 转置
	return org_img_norm;
}

cv::Rect2f bboxProjectionBack(const cv::Rect &bbox, const int output_blob_side_lens, const int idx) {
	if (output_blob_side_lens == 1)
		return bbox;
	int row_idx = idx / output_blob_side_lens;
	int col_idx = idx - row_idx * output_blob_side_lens;
	cv::Rect2f result = bbox;
	result.width /= BBOX_UP_SCALING_RATIO;
	result.height /= BBOX_UP_SCALING_RATIO;
	result.x += (bbox.width - result.width) * col_idx / float(output_blob_side_lens - 1);
	result.y += (bbox.height - result.height) * row_idx / float(output_blob_side_lens - 1);
	return result;
}

void selectBoundingBox_ByMaxProbs(ncnn::Extractor &track_ex, const cv::Rect &cur_bbox, std::pair<cv::Rect2f, float> &result_bbox)
{
	double start = ncnn::get_current_time();
	ncnn::Mat cls_blob_, reg_blob_;
	track_ex.extract("prob1", cls_blob_);
	track_ex.extract("conv6-2", reg_blob_);
	double end = ncnn::get_current_time();
	std::cout << "tracking time: " << end - start << std::endl;

	int height = cls_blob_.h;
	int width = cls_blob_.w;
	int area = height * width;

	float* output_reg[4] = { reg_blob_.channel(0), reg_blob_.channel(1), reg_blob_.channel(2), reg_blob_.channel(3) };
	float* output_probs = cls_blob_.channel(1);
	float prob_threshold = 0.985f;

	if (area > 9)
	{
		std::vector<float> output_probs_copy(output_probs, output_probs + area);
		std::nth_element(output_probs_copy.begin(), output_probs_copy.begin() + 8, output_probs_copy.end(), [](const float A, const float B) { return A > B; });
		prob_threshold = std::max(prob_threshold, *(output_probs + 8));
	}
	std::vector<std::pair<cv::Rect2f, float>> curface_bbox_list;
	for (int idx = 0; idx < area; ++idx) {
		if (*(output_probs + idx) < prob_threshold)
			continue;

		cv::Rect2f projection_bbox = bboxProjectionBack(cur_bbox, width, idx);
		curface_bbox_list.emplace_back(projection_bbox, *(output_probs + idx));

		cv::Vec4f ceff_vec;
		for (int i = 0; i < 4; ++i) {
			ceff_vec[i] = *(output_reg[i] + idx);
		}
		BbregOnly(ceff_vec, curface_bbox_list.back());
	}
	int curface_bbox_list_LENS = curface_bbox_list.size();
	if (curface_bbox_list_LENS == 0)
	{
		float* max_prob = std::max_element(output_probs, output_probs + area);
		if (*max_prob < 0.85f)
			return;
		int idx = max_prob - output_probs;

		result_bbox.first = bboxProjectionBack(cur_bbox, width, idx);
		result_bbox.second = *max_prob;

		cv::Vec4f ceff_vec;
		for (int i = 0; i < 4; ++i) {
			ceff_vec[i] = *(output_reg[i] + idx);
		}
		BbregOnly(ceff_vec, result_bbox);
	}
	else
	{
		result_bbox.first = cv::Rect2f(0, 0, 0, 0);
		result_bbox.second = 0.0f;
		for (auto &bb : curface_bbox_list) {
			result_bbox.first.x += bb.first.x;
			result_bbox.first.y += bb.first.y;
			result_bbox.first.width += bb.first.width;
			result_bbox.first.height += bb.first.height;
			result_bbox.second += bb.second;
		}
		result_bbox.first.x /= curface_bbox_list_LENS;
		result_bbox.first.y /= curface_bbox_list_LENS;
		result_bbox.first.width /= curface_bbox_list_LENS;
		result_bbox.first.height /= curface_bbox_list_LENS;

		result_bbox.second /= curface_bbox_list_LENS;
	}

	return;
}

void doForward(const std::vector<cv::Rect> &bbox, const cv::Mat &org_img_norm, std::vector<std::pair<cv::Rect2f, float>> &pred_bbox_lists)
{
	int bbox_nums = bbox.size();
	const int net_input_size = 48 * BBOX_UP_SCALING_RATIO + 1;
	const cv::Rect org_img_rect(0, 0, org_img_norm.cols, org_img_norm.rows);
	for (int k = 0; k < bbox_nums; ++k) {
		cv::Mat crop_img(bbox[k].size(), CV_8UC3, cv::Scalar(128));
		cv::Rect temp = org_img_rect & bbox[k];
		cv::Mat overlap_mat(org_img_norm, temp);
		temp -= bbox[k].tl();
		overlap_mat.copyTo(crop_img(temp));

		ncnn::Mat net_in = ncnn::Mat::from_pixels_resize(crop_img.data, ncnn::Mat::PIXEL_RGB, crop_img.cols, crop_img.rows, net_input_size, net_input_size);
		const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
		const float scale_val[3] = { 1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f };
		net_in.substract_mean_normalize(mean_vals, scale_val);

		ncnn::Extractor track_ex = ::track_net.create_extractor();
		track_ex.set_light_mode(true);
		track_ex.set_num_threads(4);
		track_ex.input("data", net_in);

		// process net output
		selectBoundingBox_ByMaxProbs(track_ex, bbox[k], pred_bbox_lists[k]);
	}

	return;
}

static int track_faces(const cv::Mat& bgr, std::vector<std::pair<cv::Rect2f, float>> &pred_bbox_lists) 
{
	cv::Mat org_img_norm = inputdataPreprocess(bgr);
	std::vector<cv::Rect> last_frame_bbox;
	enlargeBbox(cv::Rect(0, 0, bgr.cols, bgr.rows), pred_bbox_lists, last_frame_bbox, BBOX_UP_SCALING_RATIO);
	TransposeAndRerec(last_frame_bbox);
	doForward(last_frame_bbox, org_img_norm, pred_bbox_lists);

	for (auto &cur_bb : pred_bbox_lists) {
		auto &rect_transpose = cur_bb.first;
		std::swap(rect_transpose.x, rect_transpose.y);
		std::swap(rect_transpose.width, rect_transpose.height);
	}
	
	return 0;
}
//===============================================track faces v2=================================================================

void Rect2Square(const cv::Rect2f& in_rect, cv::Rect &out_squ, const float expand_rate = 1.75f)
{
	cv::Point2f center_pt;
	center_pt.x = in_rect.x + 0.5f * in_rect.width;
	center_pt.y = in_rect.y + 0.5f * in_rect.height;

	out_squ.width = out_squ.height = 0.5f * expand_rate * (in_rect.width + in_rect.height) + 0.5f;
	//out_squ.width = out_squ.height = expand_rate * std::max(in_rect.width, in_rect.height) + 0.5f;
	out_squ.x = center_pt.x - 0.5f * out_squ.width;
	out_squ.y = center_pt.y - 0.5f * out_squ.height;
}

static int track_faces_v2(const cv::Mat& bgr, std::vector<std::pair<cv::Rect2f, float>> &pred_bbox_lists_)
{
	CV_Assert(bgr.type() == CV_8UC3);

	const int net_in_size = 80;
	const float mean_vals[3] = { 103.939f, 116.779f, 123.68f };
	const float scale_val[3] = { 1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f };
	const cv::Rect org_img_rect(0, 0, bgr.cols, bgr.rows);
	for (auto &cur_bb : pred_bbox_lists_) {
		cv::Rect squ_rec;
		Rect2Square(cur_bb.first, squ_rec, 1.66667f);

		cv::Mat crop_img(squ_rec.size(), CV_8UC3, cv::Scalar(0));
		cv::Rect temp = org_img_rect & squ_rec;
		cv::Mat overlap_mat(bgr, temp);
		temp -= squ_rec.tl();
		overlap_mat.copyTo(crop_img(temp));

		ncnn::Mat net_in = ncnn::Mat::from_pixels_resize(crop_img.data, ncnn::Mat::PIXEL_BGR, crop_img.cols, crop_img.rows, net_in_size, net_in_size);
		net_in.substract_mean_normalize(mean_vals, scale_val);

		ncnn::Extractor track_ex = ::track_net.create_extractor();
		track_ex.set_light_mode(true);
		track_ex.set_num_threads(4);
		track_ex.input("data", net_in);

		// anchor_size: 48 * 48
		std::vector<Float5> result_bbox;
		processNetLayerOut(track_ex, result_bbox, "detect_4_3_cls_softmax_4_3", "detect_4_3_reg", 48, cv::Size(net_in_size, net_in_size), 8);
		NonMaxSuppress(result_bbox, 0.5f, NMS_TYPE::NMS_UNION);

		// currently, we fetch the first one of result_bbox as return-result.
		if (result_bbox.empty()) {
			cur_bb.second = 0.0f;
			continue;
		}
		//cur_bb = result_bbox[0];

		float min_scal_rate = float(net_in_size) / float(squ_rec.width);
		cur_bb.first.x /= min_scal_rate;
		cur_bb.first.y /= min_scal_rate;
		cur_bb.first.width /= min_scal_rate;
		cur_bb.first.height /= min_scal_rate;
		cur_bb.first += (cv::Point2f)squ_rec.tl();
	}

	return 0;
}

//=============================================================================================================================

void thread_detectFaces() {
	double keep_time = 0.0;
	int algo_status = 0; //0-detect;1,2-track
	while (true)
	{
		if (face_buf.detectNextNow())
		{
			double start = ncnn::get_current_time();
			
			auto &cur_proc_frame = face_buf.get_detect();
			if (algo_status == 0) //detect
			{
				detect_faces(cur_proc_frame.first, cur_proc_frame.second);
			}
			else // track
			{
				detect_faces(cur_proc_frame.first, cur_proc_frame.second);
				//track_faces_v2(cur_proc_frame.first, cur_proc_frame.second);
			}

			double end = ncnn::get_current_time();
			keep_time += end - start;
			if (++algo_status == 3) {
				::avg_speed = float(keep_time / 3.0);
				keep_time = 0.0;
				algo_status = 0;
			}
			
			std::cout << "front = " << face_buf.front_idx() << " back = " << face_buf.back_idx() << " detect_idx = " << face_buf.detect_idx();
			std::cout << " face_buf_size = " << face_buf.size() << " avg_speed = " << avg_speed << " cur_time = " << end - start << std::endl;
		}
	}
}

void thread_showFaces() {
	//int k = 0;
	while (true)
	{	
		cv::waitKey(avg_speed);
		cv::Mat cur_frame;
		if (!face_buf.isFull()) {
			cap >> cur_frame;
			::face_buf.push_back(std::make_pair(cur_frame, cv::Rect2f(0, 0, 0, 0)));
		}
		if (face_buf.showFrontNow())
		{
			auto &cur_res = face_buf.get_front();
			//for (auto &rec : cur_res.second) {
				my_smoother.SmoothBbox(cur_res.second);

				if (true) {
					cv::rectangle(cur_res.first, cur_res.second, cv::Scalar(255, 0, 0, 255));
					//cv::putText(cur_res.first, std::to_string(rec.second), rec.first.tl(),
					//	cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255));
					//cv::circle(cur_res.first, rec.first.tl() + 0.5f * cv::Point2f(rec.first.width, rec.first.height), 2, cv::Scalar(0, 0, 255, 255));
				}
				//else {
				//	cv::rectangle(cur_res.first, rec.first, cv::Scalar(0, 0, 255, 255));
				//	cv::putText(cur_res.first, std::to_string(rec.second), rec.first.tl(),
				//		cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255, 255));
				//	cv::circle(cur_res.first, rec.first.tl() + 0.5f * cv::Point2f(rec.first.width, rec.first.height), 2, cv::Scalar(0, 0, 255, 255));
				//}				
			//}
			cv::putText(cur_res.first, std::string("FPS = ") + std::to_string(1000.0f / ::avg_speed), cv::Point(20, 20),
				cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0, 255), 2);
			cv::imshow("Face Detect", cur_res.first);		
			//cv::imwrite(std::string("D:/haomaiyi/ncnn-win/build_vs2015/debug_dir/") + std::to_string(k++) + ".jpg", cur_res.first);

			face_buf.pop_front();
		}
	}
}


void test_single_thread_process() {
	cv::Mat cur_frame;
	while (true)
	{
		cap >> cur_frame;
		/*std::vector<std::pair<cv::Rect2f, float>> pred_bbox_lists;
		detect_faces(cur_frame, pred_bbox_lists);
		for (auto &rec : pred_bbox_lists) {
			cv::rectangle(cur_frame, rec.first, cv::Scalar(255, 0, 0, 255));
			cv::putText(cur_frame, std::to_string(rec.second), rec.first.tl(),
				cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255));
		}*/
		cv::imshow("Face Detect", cur_frame);
		cv::waitKey(100);
	}
}

int main(int argc, char** argv)
{
	const std::string model_dir = "D:/haomaiyi/ncnn-win/build_vs2015/tools/caffe/Release/";
	const std::string fd_param_file = model_dir + "MNet18_v2_3.param";
	const std::string fd_bin_file = model_dir + "MNet18_v2_3.bin";
	const std::string tra_param_file = model_dir + "MNet7_track.param";
	const std::string tra_bin_file = model_dir + "MNet7_track.bin";
	::InitModels(fd_param_file, fd_bin_file, tra_param_file, tra_bin_file);


	//test_single_thread_process();
	std::thread thr_1(thread_showFaces);
	std::thread thr_2(thread_detectFaces);
	thr_1.join();
	thr_2.join();
    
	system("pause");
    return 0;
}

