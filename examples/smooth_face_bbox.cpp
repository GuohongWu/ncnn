#include "smooth_face_bbox.h"


namespace rtfd {

	void FaceBboxSmoother::SmoothBbox(cv::Rect2f &in_out_rec)
	{
		this->m_recent_rects.push_back(in_out_rec);
		if (this->m_recent_frame_nums == (int)this->m_recent_rects.size())
		{
			std::vector<float> iou_weights_(this->m_recent_frame_nums, 0.0f);
			for (int i = 0; i + 1 < this->m_recent_frame_nums; ++i) {
				cv::Rect2f overlap_rect_ = this->m_recent_rects[i] & in_out_rec;
				iou_weights_[i] = 1.0f - overlap_rect_.area() / std::min(this->m_recent_rects[i].area(), in_out_rec.area());
			}

			std::vector<float> frame_weights_(this->m_recent_frame_nums, 0.0f);
			float weight_sum = 0.0f;
			float k1 = (this->m_datum_pt[1] - 1.0f) / (this->m_datum_pt[0] * this->m_datum_pt[0]);
			float k2 = (this->m_datum_pt[3] - this->m_datum_pt[1]) / (this->m_datum_pt[2] - this->m_datum_pt[0]);
			for (int i = 0; i < this->m_recent_frame_nums; ++i) {
				//float scal_weight = 1.0f + float(this->m_recent_frame_nums - i - 1) / float(this->m_recent_frame_nums);
				float scal_weight = 1.0f + float(this->m_recent_frame_nums - i - 1) / float(this->m_recent_frame_nums - i);
				iou_weights_[i] *= scal_weight;

				if (iou_weights_[i] < this->m_datum_pt[0])
					frame_weights_[i] = 1.0f + k1 * iou_weights_[i] * iou_weights_[i];
				else if (iou_weights_[i] < this->m_datum_pt[2])
					frame_weights_[i] = k2 * (iou_weights_[i] - this->m_datum_pt[0]) + this->m_datum_pt[1];
				else
					frame_weights_[i] = this->m_datum_pt[3];

				weight_sum += frame_weights_[i];
			}
			for (int i = 0; i < this->m_recent_frame_nums; ++i)
				frame_weights_[i] /= weight_sum;

			cv::Rect2f smooth_result_(0, 0, 0, 0);
			for (int i = 0; i < this->m_recent_frame_nums; ++i) {
				//std::cout << frame_weights_[i] << " ";

				smooth_result_.x += frame_weights_[i] * this->m_recent_rects[i].x;
				smooth_result_.y += frame_weights_[i] * this->m_recent_rects[i].y;
				smooth_result_.width += frame_weights_[i] * this->m_recent_rects[i].width;
				smooth_result_.height += frame_weights_[i] * this->m_recent_rects[i].height;
			}
			//std::cout << std::endl;
			in_out_rec = smooth_result_;
			//this->m_recent_rects.back() = smooth_result_;

			this->m_recent_rects.pop_front();
		}
	}

}

