#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>

namespace SimpleImage
{

	class Size
	{
		size_t m_width, m_height;

	public:

		Size(cv::Size size);

		Size(size_t width = 0, size_t height = 0);

		~Size();

		size_t width()  const;
		size_t height() const;

		size_t area() const;
		size_t total() const;

		std::string toString() const;
	};

}

std::ostream & operator<<(std::ostream &os, const SimpleImage::Size &size);
