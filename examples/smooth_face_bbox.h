#ifndef _SMOOTH_FACE_BBOX_H_
#define _SMOOTH_FACE_BBOX_H_

#include <deque>
#include <vector>
#include <opencv2/core/core.hpp>

namespace rtfd {

	class FaceBboxSmoother {
	public:
		FaceBboxSmoother() { }
		FaceBboxSmoother(int frame_nums, float x0, float y0, float x1, float y1) : m_recent_frame_nums(frame_nums), m_datum_pt{ x0, y0, x1, y1 } { }

		void SmoothBbox(cv::Rect2f &in_out_rec);
		void clear() { this->m_recent_rects.clear(); }


	private:
		std::deque<cv::Rect2f> m_recent_rects;
		int m_recent_frame_nums = 3;
		//float m_weight_factor = 1.25f;
		float m_datum_pt[4] = { 0.1f, 0.8f, 0.42f, 0.0f };   //x0, y0, x1, y1
	};

}


#endif
