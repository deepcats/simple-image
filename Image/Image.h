#pragma once

#include "opencv2/opencv.hpp"

#include "Size.h"
#include "ImageType.h"

//#define TINY_DNN
//#define QT

#ifdef TINY_DNN
#include "tiny_dnn/tiny_dnn.h"
#endif

#ifdef QT
#include <Qimage>
#include <QPixmap>
#endif



namespace SimpleImage
{
	typedef double numerical;


	class Image
	{
		cv::Mat data;

	public:

		Image(const cv::Mat &mat);
		Image(const std::string &fileName);

		Image();
		Image(ImageType type, Size size);
		Image(ImageType type, size_t rows, size_t cols);
		Image(Size size, ImageType type = ImageType(UCHAR));
		Image(size_t rows, size_t cols, ImageType type = ImageType(UCHAR));

		Image(const Image &img);

		~Image();

		Image toGrayScale() const;
		void meToGrayScale();

		Image adjust() const;
		void meAdjust();

		Image convertTo(ImageType type) const;
		void meConvertTo(ImageType type);

		size_t channels() const;
		size_t depth() const;

		Size size() const;
		size_t cols() const;
		size_t sizeX() const;
		size_t width() const;
		size_t rows() const;
		size_t sizeY() const;
		size_t height() const;

		ImageType type() const;

		void imshow(std::string windowName = std::string("Image"), bool blocking = true, size_t delay = 0) const;
		void show(std::string windowName = std::string("Image"), bool blocking = true, size_t delay = 0) const;

		void read(const std::string &fileName);
		void load(const std::string &fileName);
		void write(const std::string &fileName);
		void save(const std::string &fileName);

		Image resize(Size newSize) const;
		void  meResize(Size newSize);

		Image cropBorders(double percentWidth, double percentHeight) const;
		void meCropBorders(double percentWidth, double percentHeight);

		Image rotate(float angle, bool inDegrees = true) const;
		void meRotate(float angle, bool inDegrees = true);

		Image rotateCrop(float angle, bool inDegrees = true) const;
		void meRotateCrop(float angle, bool inDegrees = true);

		Image rotateXYZ(float angleX, float angleY, float angleZ, bool inDegrees = true, float verticalFieldOfViewInDegrees = 30.0f) const;
		void meRotateXYZ(float angleX, float angleY, float angleZ, bool inDegrees = true, float verticalFieldOfViewInDegrees = 30.0f);

		Image rotateXYZCrop(float angleX, float angleY, float angleZ, bool inDegrees = true, float verticalFieldOfViewInDegrees = 30.0f) const;
		void meRotateXYZCrop(float angleX, float angleY, float angleZ, bool inDegrees = true, float verticalFieldOfViewInDegrees = 30.0f);

		std::vector<Image> split() const;
		static std::vector<Image> split(const Image &multiChannelImage);

		static Image merge(std::vector<Image> images);

		Image mixChannels(const std::vector<size_t> &channelIndices) const;

		operator cv::Mat() const;

#ifdef TINY_DNN

		operator tiny_dnn::vec_t   () const;
		operator tiny_dnn::tensor_t() const;

		static Image fromVec_t   (const tiny_dnn::vec_t    &vec,    Size imgSize);
		static Image fromTensor_t(const tiny_dnn::tensor_t &tensor, Size imgSize);
#endif

#ifdef QT

		operator QImage() const;
		operator QPixmap() const;

		static Image fromQImage(const QImage   &qimage);
		static Image fromQPixmap(const QPixmap &qpixmap);

#endif

        void setRGB(size_t x, size_t y, numerical red, numerical green, numerical blue);

		void setRed(size_t x, size_t y, numerical red);
		void setGreen(size_t x, size_t y, numerical green);
		void setBlue(size_t x, size_t y, numerical blue);

		void setGrayTone(size_t x, size_t y, numerical intensity);

		void makeRed(size_t x, size_t y, numerical red);
		void makeGreen(size_t x, size_t y, numerical green);
		void makeBlue(size_t x, size_t y, numerical blue);

		void makeGrayTone(size_t x, size_t y, numerical intensity);

		void setChannelAt(size_t x, size_t y, size_t channel, numerical value);
		numerical getChannelAt(size_t x, size_t y, size_t channel) const;

		numerical grayTone(size_t x, size_t y) const;

		numerical red(size_t x, size_t y) const;
		numerical green(size_t x, size_t y) const;
		numerical blue(size_t x, size_t y) const;

		numerical alpha(size_t x, size_t y) const;

		numerical operator() (size_t x, size_t y, size_t channel = 0) const;
		Image operator[] (size_t channel) const;

		Image operator +(const Image &) const;
		Image operator -(const Image &) const;

		Image& operator = (const Image &);
		Image& operator += (const Image &);
		Image& operator -= (const Image &);

		template<typename T>
		void setTo(T t) {
			this->data.setTo(t);
		}

		template<typename T>
		Image operator + (T k) const
		{
			Image t(*this);
			t.data.setTo(k);

			return t + *this;
		}

		template<typename T>
		Image operator - (T k) const
		{
			Image t(*this);
			t.data.setTo(k);

			return *this - t;
		}

		template<typename T>
		Image operator * (T k) const
		{
			Image t(*this);
			t.data.setTo(k);

			return this->data.mul(static_cast<cv::Mat>(t));
		}

		template<typename T>
		Image operator / (T k) const
		{
			Image t(*this);
			t.data.setTo(static_cast<T>(1) / k);

			return this->data.mul(static_cast<cv::Mat>(t));
		}

		template<typename T>
		Image& operator += (T k) {
			*this = *this + k;
			return *this;
		}

		template<typename T>
		Image& operator -= (T k) {
			*this = *this - k;
			return *this;
		}

		template<typename T>
		Image& operator *= (T k) {
			*this = ((*this) * k);
			return *this;
		}

		template<typename T>
		Image& operator /= (T k) {
			*this = ((*this) / k);
			return *this;
		}

private:

		void *at(size_t x, size_t y, size_t channel = 0);
		const void *at(size_t x, size_t y, size_t channel = 0) const;

		static void Stretchlim(cv::Mat image, float* l, float* h);
		static cv::Mat adjustArray(cv::Mat image, float l, float h);
		static cv::Mat imadjustGray(cv::Mat image);
		static void StretchlimGray(cv::Mat image, float* l, float* h);
		static cv::Mat adjustArrayGray(cv::Mat image, float l, float h);

		void warpMatrix(cv::Size sz, double theta, double phi, double gamma, double scale, double fovY, cv::Mat &M, std::vector<cv::Point2f> *corners = static_cast<std::vector<cv::Point2f> *>(NULL), std::vector<cv::Point2f> *extraPoints = static_cast<std::vector<cv::Point2f> *>(NULL)) const;
	};


	template<typename T>
	Image operator + (T k, const Image &inp)
	{
		Image t(inp);
		t.data.setTo(k);

		return t + inp;
	}

	template<typename T>
	Image operator - (T k, const Image &inp)
	{
		Image t(inp);
		t.data.setTo(k);

		return t - inp;
	}

	template<typename T>
    Image operator * (T k, const Image &inp) {
		Image t(inp);
		t.data.setTo(k);

        return t.data.mul(static_cast<cv::Mat>(t));
	}

}

std::ostream & operator <<(std::ostream & os, const SimpleImage::Image &img);
