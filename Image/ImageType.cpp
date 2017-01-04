#include "ImageType.h"

#include "opencv2/opencv.hpp"

using namespace SimpleImage;

ImageType::ImageType(DataType dataType, size_t channels)
{

	this->type = dataType;
	this->m_channels = channels;

	switch (dataType)
	{
		case UNSIGNED_CHAR:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_8U, channels));
			break;
		case SIGNED_CHAR:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_8S, channels));
			break;
		case UNSIGNED_SHORT:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_16U, channels));
			break;
		case SIGNED_SHORT:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_16S, channels));
			break;
		case INT:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_32S, channels));
			break;
		case FLOAT:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_32F, channels));
			break;
		case DOUBLE:
			this->data = static_cast<int>(CV_MAKE_TYPE(CV_64F, channels));
			break;
	}
}

ImageType::ImageType(size_t cvCode)
{
	this->data = static_cast<int>(cvCode);
	this->m_channels = cv::Mat(cv::Size(1, 1), this->data).channels();

	cv::Mat channel;

	cv::extractChannel(cv::Mat(cv::Size(1, 1), this->data), channel, 0);

	this->type = static_cast<DataType>(channel.type());
}

ImageType::~ImageType()
{}

int ImageType::cvCode() const
{
	return this->data;
}

size_t ImageType::channels() const
{
	return this->m_channels;
}

size_t ImageType::depth() const
{
	if (this->type <= SIGNED_CHAR) {
		return 8;
	}

	if (this->type <= SIGNED_SHORT) {
		return 16;
	}

	if (this->type <= FLOAT) {
		return 32;
	}

	return 64;
}

std::string ImageType::toString() const
{
	std::string typeDescription;

	switch (this->type)
	{
		case UNSIGNED_CHAR:
			typeDescription = "unsigned char";
			break;
		case SIGNED_CHAR:
			typeDescription = "signed char";
			break;
		case UNSIGNED_SHORT:
			typeDescription = "unsigned short";
			break;
		case SIGNED_SHORT:
			typeDescription = "signed short";
			break;
		case INT:
			typeDescription = "int";
			break;
		case FLOAT:
			typeDescription = "float";
			break;
		case DOUBLE:
			typeDescription = "double";
			break;
	}

	return std::string("[") + typeDescription + std::string(" x ") + std::to_string(this->channels()) + std::string("]");
}

std::string ImageType::toCVString() const
{
	std::string typeDescription;

	switch (this->type)
	{
		case UNSIGNED_CHAR:
			typeDescription = "CV_8U";
			break;
		case SIGNED_CHAR:
			typeDescription = "CV_16S";
			break;
		case UNSIGNED_SHORT:
			typeDescription = "CV_16U";
			break;
		case SIGNED_SHORT:
			typeDescription = "CV_16S";
			break;
		case INT:
			typeDescription = "CV_32S";
			break;
		case FLOAT:
			typeDescription = "CV_32F";
			break;
		case DOUBLE:
			typeDescription = "CV_64F";
			break;
	}

	if (int(this->m_channels) <= 4) {
		return typeDescription + std::string("C") + std::to_string(this->m_channels);
	}

	return std::string("CV_MAKE_TYPE(") + typeDescription + std::string(", ") + std::to_string(this->m_channels) + std::string(")");	
}

DataType ImageType::dataType() const
{
	return this->type;
}

std::ostream & operator <<(std::ostream &os, const ImageType &imgType)
{
	os << imgType.toString();
	return os;
}