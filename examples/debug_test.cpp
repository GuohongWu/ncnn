#include <vector>
#include <iostream>


int main()
{
	std::vector<float[5]> pred_bbox_lists(1);
	pred_bbox_lists[0][0] = 0;
	pred_bbox_lists[0][1] = 1;
	float tmp[5] = {0, 1, 2, 3, 4};
	std::cout << sizeof(float[5]);
	pred_bbox_lists.emplace_back(1,1, 1,1,1);
	return 0;
}