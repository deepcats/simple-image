#include "Size.h"

using namespace SimpleImage;

std::ostream & operator<<(std::ostream &os, const Size &size)
{
	os << size.toString();
	return os;
}

Size::Size(cv::Size cv_size) {
	this->m_width  = cv_size.width;
	this->m_height = cv_size.height;
}

Size::Size(size_t width, size_t height) {
	this->m_width = width;
	this->m_height = height;
}

Size::~Size() {}

size_t Size::width() const
{
	return this->m_width;
}

size_t Size::height() const
{
	return this->m_height;
}

size_t Size::area() const
{
	return this->width() * this->height();
}

size_t Size::total() const
{
	return this->area();
}

std::string Size::toString() const {
	return std::string("[width = ") + std::to_string(this->width()) + std::string(" x height = ") + std::to_string(this->height()) + std::string("]");
}
