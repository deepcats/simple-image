#include "Image.h"

#include "ImageTransformer.h"

#include <cmath>

using namespace SimpleImage;

Image::Image(const cv::Mat &mat)
{
	mat.copyTo(this->data);
}

Image::Image(const std::string &fileName)
{
	cv::imread(fileName).copyTo(this->data);

	if (data.total() == 0) {
		throw std::string("Can't load image file") + fileName;
	}
}

Image::Image(const Image &img)
{
	img.data.copyTo(this->data);
}

Image::Image()
{
	this->data = cv::Mat();
}

Image::Image(ImageType type, Size size)
{
	this->data.create(cv::Size(static_cast<int>(size.width()), static_cast<int>(size.height())), type.cvCode());
	this->data.setTo(0);
}

Image::Image(Size size, ImageType type)
{
	this->data.create(cv::Size(static_cast<int>(size.width()), static_cast<int>(size.height())), type.cvCode());
	this->data.setTo(0);
}

Image::Image(size_t rows, size_t cols, ImageType type)
{
	this->data.create(cv::Size(static_cast<int>(cols), static_cast<int>(rows)), type.cvCode());
	this->data.setTo(0);
}

Image::Image(ImageType type, size_t rows, size_t cols)
{
	this->data.create(cv::Size(static_cast<int>(cols), static_cast<int>(rows)), type.cvCode());
	this->data.setTo(0);
}

Image::~Image()
{

}

Image Image::toGrayScale() const
{
	cv::Mat answerData;

	if (this->channels() == 3) {
		cv::cvtColor(this->data, answerData, CV_BGR2GRAY);
	} else if (this->channels() == 4) {
		cv::cvtColor(this->data, answerData, CV_BGRA2GRAY);
	} else if (this->channels() == 1) {
		this->data.copyTo(answerData);
	}

	return Image(answerData);
}

Image Image::adjust() const
{
	float l, h;
	Image::Stretchlim(static_cast<cv::Mat>(*this), &l, &h);
	cv::Mat result = adjustArray(static_cast<cv::Mat>(*this), l, h);
	return static_cast<Image>(result);
}

void Image::meAdjust()
{
	*this = this->adjust();
}

Image Image::convertTo(ImageType type) const
{
	cv::Mat answerData;

	this->data.convertTo(answerData, type.cvCode());

	return Image(answerData);
}

void Image::meConvertTo(ImageType type) 
{
	this->data.convertTo(this->data, type.cvCode());
}

void Image::meToGrayScale()
{
	if (this->channels() == 3) {
		cv::cvtColor(this->data, this->data, CV_BGR2GRAY);
	} else if (this->channels() == 4) {
		cv::cvtColor(this->data, this->data, CV_BGRA2GRAY);
	}
}



size_t Image::channels() const
{
	return static_cast<size_t>(this->data.channels());
}

size_t Image::depth() const
{
	return static_cast<size_t>(this->data.depth());
}

Size Image::size() const
{
	return Size(this->data.size());
}

size_t Image::cols() const
{
	return static_cast<size_t>(this->data.cols);
}

size_t Image::sizeX() const
{
	return this->cols();
}

size_t Image::width() const
{
	return this->cols();
}

size_t Image::rows() const
{
	return static_cast<size_t>(this->data.rows);
}

size_t Image::sizeY() const
{
	return this->rows();
}

size_t Image::height() const
{
	return this->rows();
}

ImageType Image::type() const
{
	return ImageType(this->data.type());
}

void Image::imshow(std::string windowName, bool blocking, size_t delay) const
{
	cv::imshow(windowName, this->data);
	cv::waitKey(blocking ? 0 : static_cast<int>(delay));
}

void Image::show(std::string windowName, bool blocking, size_t delay) const
{
	this->imshow(windowName, blocking, delay);
}

void Image::read(const std::string &fileName)
{
	*this = Image(fileName);
}

void Image::load(const std::string &fileName)
{
	this->read(fileName);
}

void Image::write(const std::string &fileName)
{
	cv::imwrite(fileName, this->data);
}

void Image::save(const std::string &fileName)
{
	this->write(fileName);
}

Image Image::resize(Size newSize) const
{
	Image result;
	cv::resize(this->data, result.data, cv::Size(newSize.width(), newSize.height()));
	return result;
}

void Image::meResize(Size newSize)
{
	cv::resize(this->data, this->data, cv::Size(newSize.width(), newSize.height()));
}

Image Image::cropBorders(double percentWidth, double percentHeight) const
{
	int width  = int(double(this->data.cols) * (100.0 - percentWidth) / 100.0);
	int height = int(double(this->data.rows) * (100.0 - percentHeight) / 100.0);

	cv::Point center = cv::Point(this->data.cols / 2, this->data.rows / 2);
	cv::Point shift = cv::Point(width / 2, height / 2);

	cv::Rect rect(center - shift, center + shift);

	return Image(this->data(rect));
}

void Image::meCropBorders(double percentWidth, double percentHeight)
{
	*this = this->cropBorders(percentWidth, percentHeight);
}

Image Image::rotate(float angle, bool inDegrees) const
{
	return this->rotateXYZ(.0f, .0f, angle, inDegrees);
}

void Image::meRotate(float angle, bool inDegrees)
{
	this->meRotateXYZ(.0f, .0f, angle, inDegrees);
}

Image Image::rotateCrop(float angle, bool inDegrees) const
{
	return this->rotateXYZCrop(.0f, .0f, angle, inDegrees);
}

void Image::meRotateCrop(float angle, bool inDegrees)
{
	this->meRotateXYZCrop(.0f, .0f, angle, inDegrees);
}

Image Image::rotateXYZ(float angleX, float angleY, float angleZ, bool inDegrees, float fovY) const
{
	double fovYDeg = (inDegrees ? static_cast<double>(fovY) : (static_cast<double>(fovY) * 180.0 / CV_PI));

	double halfFovy = 0.5 * static_cast<double>(fovYDeg);
    double d = std::hypot(this->data.cols, this->data.rows);
	double scale = 1.0;
	double sideLength = scale * d / cos(halfFovy * CV_PI / 180.0);

	double theta = (inDegrees ? static_cast<double>(angleZ) : (static_cast<double>(angleZ) * 180.0 / CV_PI));
	double phi   = (inDegrees ? static_cast<double>(angleX) : (static_cast<double>(angleX) * 180.0 / CV_PI));
	double gamma = (inDegrees ? static_cast<double>(angleY) : (static_cast<double>(angleY) * 180.0 / CV_PI));

	cv::Mat M;
	std::vector<cv::Point2f> corners;

	this->warpMatrix(this->data.size(), theta, phi, gamma, scale, fovYDeg, M, &corners);

	cv::Mat result;
	cv::warpPerspective(this->data, result, M, cv::Size(int(sideLength), int(sideLength)));

	return Image(result(cv::boundingRect(corners)));
}

void Image::meRotateXYZ(float angleX, float angleY, float angleZ, bool inDegrees, float fovY)
{
	*this = this->rotateXYZ(angleX, angleY, angleZ);
}

Image Image::rotateXYZCrop(float angleX, float angleY, float angleZ, bool inDegrees, float fovY) const
{
	double fovYDeg = (inDegrees ? static_cast<double>(fovY) : (static_cast<double>(fovY) * 180.0 / CV_PI));

	double halfFovy = 0.5 * static_cast<double>(fovYDeg);
	double d = std::hypot(this->data.cols, this->data.rows);
	double scale = 1.0;
	double sideLength = scale * d / cos(halfFovy * CV_PI / 180.0);

	double theta = (inDegrees ? static_cast<double>(angleZ) : (static_cast<double>(angleZ) * 180.0 / CV_PI));
	double phi = (inDegrees ? static_cast<double>(angleX) : (static_cast<double>(angleX) * 180.0 / CV_PI));
	double gamma = (inDegrees ? static_cast<double>(angleY) : (static_cast<double>(angleY) * 180.0 / CV_PI));

	cv::Mat M;
	std::vector<cv::Point2f> corners;

	std::vector<cv::Point2f> centerPoint;
	centerPoint.push_back(cv::Point2f(static_cast<float>(this->data.cols / 2), static_cast<float>(this->data.rows / 2)));

	this->warpMatrix(this->data.size(), theta, phi, gamma, scale, fovYDeg, M, &corners, &centerPoint);

	cv::Mat rotatedMat;
	cv::warpPerspective(this->data, rotatedMat, M, cv::Size(int(sideLength), int(sideLength)));

	std::vector<std::vector<cv::Point> > imageContour;
	imageContour.resize(1);

	for (int i = 0; i < 4; ++i) {
		imageContour[0].push_back(cv::Point(static_cast<int>(corners[i].x), static_cast<int>(corners[i].y)));
	}

	cv::Mat imageMask = cv::Mat::zeros(cv::Size(rotatedMat.size()), CV_8U);
	cv::drawContours(imageMask, imageContour, 0, cv::Scalar(255, 0, 0), CV_FILLED);

	cv::Rect boundBox = cv::boundingRect(corners);

	cv::Mat resultNonCropped = rotatedMat(boundBox);
	imageMask = imageMask(boundBox);

	cv::Point centerPointInt = cv::Point(static_cast<int>(centerPoint[0].x), static_cast<int>(centerPoint[0].y));
	centerPointInt = centerPointInt - boundBox.tl();

	int centeredFrameWidthHalf  = std::min(centerPointInt.x, resultNonCropped.cols - centerPointInt.x);
	int centeredFrameHeightHalf = std::min(centerPointInt.y, resultNonCropped.rows - centerPointInt.y);

	cv::Point tl = cv::Point(centerPointInt.x - centeredFrameWidthHalf, centerPointInt.y - centeredFrameHeightHalf);
	cv::Point br = cv::Point(centerPointInt.x + centeredFrameWidthHalf, centerPointInt.y + centeredFrameHeightHalf);

	cv::Mat centeredRotatedMask = imageMask(cv::Rect(tl, br));

	cv::Mat tlMat = centeredRotatedMask.colRange(0, centeredRotatedMask.cols / 2).rowRange(0, centeredRotatedMask.rows / 2),
			trMat = centeredRotatedMask.colRange(centeredRotatedMask.cols / 2, centeredRotatedMask.cols - 1).rowRange(0, centeredRotatedMask.rows / 2), 
			blMat = centeredRotatedMask.colRange(0, centeredRotatedMask.cols / 2).rowRange(centeredRotatedMask.rows / 2, centeredRotatedMask.rows - 1),
			brMat = centeredRotatedMask.colRange(centeredRotatedMask.cols / 2, centeredRotatedMask.cols - 1).rowRange(centeredRotatedMask.rows / 2, centeredRotatedMask.rows - 1);

	cv::resize(trMat, trMat, tlMat.size());
	cv::resize(blMat, blMat, tlMat.size());
	cv::resize(brMat, brMat, tlMat.size());

	cv::Mat trMatFlip, blMatFlip, brMatFlip;

	cv::flip(trMat, trMatFlip, 1);
	cv::flip(blMat, blMatFlip, 0);
	cv::flip(brMat, brMatFlip, -1);

	cv::Mat mask = tlMat & trMatFlip & blMatFlip & brMatFlip;

	bool squareIncreasing = true;
	int diagPointCoord = 0;
	do {
		if (std::min(mask.cols - diagPointCoord - 1, mask.rows - diagPointCoord - 1) > 0 && mask.at<uchar>(cv::Point(mask.cols - diagPointCoord - 1, mask.rows - diagPointCoord - 1))) {
			diagPointCoord += 1;
		} else {
			squareIncreasing = false;
		}
	} while (squareIncreasing);

	diagPointCoord -= 1;
	int colShift = diagPointCoord;
	while (mask.cols - colShift - 1 > 0 && mask.at<uchar>(cv::Point(mask.cols - colShift - 1, mask.rows - diagPointCoord - 1))) {
		++colShift;
	}

	int rowShift = diagPointCoord;
	while (mask.rows - rowShift - 1 > 0 && mask.at<uchar>(cv::Point(mask.cols - diagPointCoord - 1, mask.rows - rowShift - 1))) {
		++rowShift;
	}

	cv::Point shift = cv::Point(colShift - 1, rowShift - 1);

	cv::Rect maxRect = cv::Rect(centerPointInt - shift, centerPointInt + shift);

	cv::Mat resultCropped = resultNonCropped(maxRect);

	return Image(resultCropped);
}

void Image::meRotateXYZCrop(float angleX, float angleY, float angleZ, bool inDegrees, float verticalFieldOfViewInDegrees)
{
	*this = this->rotateXYZCrop(angleX, angleY, angleZ, inDegrees, 30.0);
}

std::vector<Image> Image::split() const
{
	std::vector<Image> result;
	for (size_t channel = 0; channel < this->channels(); ++channel) {
		cv::Mat tmp;
		cv::extractChannel(this->data, tmp, channel);
		result.push_back(Image(tmp));
	}
	return result;
}

std::vector<Image> Image::split(const Image &input)
{
	std::vector<Image> result;
	for (size_t channel = 0; channel < input.channels(); ++channel) {
		cv::Mat tmp;
		cv::extractChannel(input.data, tmp, channel);
		result.push_back(Image(tmp));
	}
	return result;
}

Image Image::merge(std::vector<Image> inputs)
{
	cv::Mat resultMat;
	std::vector<cv::Mat> inputsMat;

	for (size_t i = 0; i < inputs.size(); ++i) {
		inputsMat.push_back(inputs[i].data.clone());
	}

	cv::merge(inputsMat, resultMat);

	return Image(resultMat);
}

Image Image::mixChannels(const std::vector<size_t> &indices) const
{
	if (indices.size() != this->channels()) {
		throw "Number of the indices must equal to the number of image channels!";
	}

	std::vector<Image> channelImages = this->split();
	std::vector<Image> mixedChannels;

	for (size_t channel = 0; channel < channelImages.size(); ++channel) {
		mixedChannels.push_back(channelImages[indices[channel]]);
	}

	return Image::merge(mixedChannels);
}

Image::operator cv::Mat() const
{
	return this->data.clone();
}

#ifdef TINY_DNN

Image::operator tiny_dnn::vec_t() const
{
	return ImageTransformer::image2vec_t(*this);
}

Image::operator tiny_dnn::tensor_t() const
{
	return ImageTransformer::image2tensor_t(*this);
}

Image Image::fromVec_t(const tiny_dnn::vec_t &vec, Size imgSize)
{
	return ImageTransformer::vec_t2image(vec, imgSize);
}

Image Image::fromTensor_t(const tiny_dnn::tensor_t &tensor, Size imgSize)
{
	return ImageTransformer::tensor_t2image(tensor, imgSize);
}

#endif

#ifdef QT

Image::operator QImage() const
{
	return ImageTransformer::image2qimage(*this);
}

Image::operator QPixmap() const
{
	return ImageTransformer::image2qpixmap(*this);
}

Image Image::fromQImage(const QImage &qimage)
{
	return ImageTransformer::qimage2image(qimage);
}

Image Image::fromQPixmap(const QPixmap &qpixmap)
{
	return ImageTransformer::qpixmap2image(qpixmap);
}

#endif

void Image::setRGB(size_t x, size_t y, numerical red, numerical green, numerical blue) 
{
	this->setRed(x, y, red);
	this->setGreen(x, y, green);
	this->setBlue(x, y, blue);
}

void Image::setRed(size_t x, size_t y, numerical red) {
	this->setChannelAt(x, y, 2, red);
}

void Image::setGreen(size_t x, size_t y, numerical green) {
	this->setChannelAt(x, y, 1, green);
}

void Image::setBlue(size_t x, size_t y, numerical blue) {
	this->setChannelAt(x, y, 0, blue);
}

void Image::setGrayTone(size_t x, size_t y, numerical intensity)
{
	if (this->channels() != 1) {
		throw "This method works only for grayscale (single-channel) images!";
	}

	this->setChannelAt(x, y, 0, intensity);
}

void Image::makeRed(size_t x, size_t y, numerical red) {
	this->setRGB(x, y, red, 0, 0);
}

void Image::makeGreen(size_t x, size_t y, numerical green) {
	this->setRGB(x, y, 0, green, 0);
}

void Image::makeBlue(size_t x, size_t y, numerical blue) {
	this->setRGB(x, y, 0, 0, blue);
}

void Image::makeGrayTone(size_t x, size_t y, numerical intensity) {
	this->setRGB(x, y, intensity, intensity, intensity);
}

numerical Image::grayTone(size_t x, size_t y) const {
	
	if (this->channels() != 1) {
		throw "This method works only for grayscale (single-channel) images!";
	}

	return this->getChannelAt(x, y, 0);
}

numerical Image::red(size_t x, size_t y) const {
	return this->getChannelAt(x, y, 2);
}

numerical Image::green(size_t x, size_t y) const {
	return this->getChannelAt(x, y, 1);
}

numerical Image::blue(size_t x, size_t y) const {
	return this->getChannelAt(x, y, 0);
}

numerical Image::alpha(size_t x, size_t y) const {
	return this->getChannelAt(x, y, 3);
}

numerical Image::operator() (size_t x, size_t y, size_t channel) const
{
	return this->getChannelAt(x, y, channel);
}

Image Image::operator[] (size_t channel) const
{
	return (this->split())[channel];
}

Image Image::operator +(const Image &inp) const
{
	Image result;
	cv::add(this->data, inp.data, result.data);
	return result;
}

Image Image::operator -(const Image &inp) const
{
	Image result;
	cv::subtract(this->data, inp.data, result.data);
	return result;
}

Image& Image::operator = (const Image &inp)
{
	inp.data.copyTo(this->data);
	return *this;
}

Image& Image::operator += (const Image &inp)
{
	Image t(*this);
	cv::add(t.data, inp.data, this->data);
	return *this;
}

Image& Image::operator -= (const Image &inp)
{
	Image t(*this);
	cv::subtract(t.data, inp.data, this->data);
	return *this;
}

numerical Image::getChannelAt(size_t x, size_t y, size_t channel) const {
	if (this->type().dataType() == UCHAR) {
		return static_cast<numerical>(*static_cast<const uchar *>(this->at(x, y, channel)));
	}
	if (this->type().dataType() == SCHAR) {
		return static_cast<numerical>(*static_cast<const schar *>(this->at(x, y, channel)));
	}
	if (this->type().dataType() == USHORT) {
		return static_cast<numerical>(*static_cast<const ushort *>(this->at(x, y, channel)));
	}
	if (this->type().dataType() == SSHORT) {
		return static_cast<numerical>(*static_cast<const signed short *>(this->at(x, y, channel)));
	}
	if (this->type().dataType() == INT) {
		return static_cast<numerical>(*static_cast<const int *>(this->at(x, y, channel)));
	}
	if (this->type().dataType() == FLOAT) {
		return static_cast<numerical>(*static_cast<const float *>(this->at(x, y, channel)));
	}
	if (this->type().dataType() == DOUBLE) {
		return static_cast<numerical>(*static_cast<const double *>(this->at(x, y, channel)));
	}

	return numerical(0);
}

void Image::setChannelAt(size_t x, size_t y, size_t channel, numerical value)
{
	if (this->type().dataType() == UCHAR) {
		*static_cast<uchar *>(this->at(x, y, channel)) = static_cast<uchar>(value);
	}
	if (this->type().dataType() == SCHAR) {
		*static_cast<schar *>(this->at(x, y, channel)) = static_cast<schar>(value);
	}
	if (this->type().dataType() == USHORT) {
		*static_cast<ushort *>(this->at(x, y, channel)) = static_cast<ushort>(value);
	}
	if (this->type().dataType() == SSHORT) {
		*static_cast<signed short *>(this->at(x, y, channel)) = static_cast<signed short>(value);
	}
	if (this->type().dataType() == INT) {
		*static_cast<int *>(this->at(x, y, channel)) = static_cast<int>(value);
	}
	if (this->type().dataType() == FLOAT) {
		*static_cast<float *>(this->at(x, y, channel)) = static_cast<float>(value);
	}
	if (this->type().dataType() == DOUBLE) {
		*static_cast<double *>(this->at(x, y, channel)) = static_cast<double>(value);
	}
}

void *Image::at(size_t x, size_t y, size_t channel) 
{

	if (this->channels() > 20) {
		throw "at: too many channels. Maximum possible for this method is 20";
	}

#define RETURN_BY_CHANNEL_SET(i) \
			if (this->channels() == i) { \
				if (this->type().dataType() == UCHAR) {\
					return static_cast<void *>(&(this->data.at<cv::Vec<uchar, i> >(int(y), int(x))[channel])); \
												} \
				if (this->type().dataType() == SCHAR) { \
					return static_cast<void *>(&(this->data.at<cv::Vec<schar, i> >(int(y), int(x))[channel])); \
												} \
				if (this->type().dataType() == USHORT) { \
					return static_cast<void *>(&(this->data.at<cv::Vec<unsigned short, i> >(int(y), int(x))[channel]));\
												}\
				if (this->type().dataType() == SSHORT) { \
					return static_cast<void *>(&(this->data.at<cv::Vec<signed short, i> >(int(y), int(x))[channel]));\
												}\
				if (this->type().dataType() == INT) {\
					return static_cast<void *>(&(this->data.at<cv::Vec<int, i> >(int(y), int(x))[channel]));\
												}\
				if (this->type().dataType() == FLOAT) {\
					return static_cast<void *>(&(this->data.at<cv::Vec<float, i> >(int(y), int(x))[channel]));\
												}\
				if (this->type().dataType() == DOUBLE) {\
					return static_cast<void *>(&(this->data.at<cv::Vec<double, i> >(int(y), int(x))[channel]));\
												}\
						}
#define R(i) RETURN_BY_CHANNEL_SET(i)
	R(1) R(2) R(3) R(4) R(5) R(6) R(7) R(8) R(9) R(10) R(11) R(12) R(13) R(14) R(15) R(16) R(17) R(18) R(19) R(20)

#undef RETURN_BY_CHANNEL_SET
#undef R

		return NULL;
}

const void *Image::at(size_t x, size_t y, size_t channel) const 
{

	if (this->channels() > 20) {
		throw "at: too many channels. Maximum possible for this method is 20";
	}

#define RETURN_BY_CHANNEL_SET(i) \
			if (this->channels() == i) { \
				if (this->type().dataType() == UCHAR) {\
					return static_cast<const void *>(&(this->data.at<cv::Vec<uchar, i> >(int(y), int(x))[channel])); \
																} \
				if (this->type().dataType() == SCHAR) { \
					return static_cast<const void *>(&(this->data.at<cv::Vec<schar, i> >(int(y), int(x))[channel])); \
																} \
				if (this->type().dataType() == USHORT) { \
					return static_cast<const void *>(&(this->data.at<cv::Vec<unsigned short, i> >(int(y), int(x))[channel]));\
																}\
				if (this->type().dataType() == SSHORT) { \
					return static_cast<const void *>(&(this->data.at<cv::Vec<signed short, i> >(int(y), int(x))[channel]));\
																}\
				if (this->type().dataType() == INT) {\
					return static_cast<const void *>(&(this->data.at<cv::Vec<int, i> >(int(y), int(x))[channel]));\
																}\
				if (this->type().dataType() == FLOAT) {\
					return static_cast<const void *>(&(this->data.at<cv::Vec<float, i> >(int(y), int(x))[channel]));\
																}\
				if (this->type().dataType() == DOUBLE) {\
					return static_cast<const void *>(&(this->data.at<cv::Vec<double, i> >(int(y), int(x))[channel]));\
																}\
									}
#define R(i) RETURN_BY_CHANNEL_SET(i)
	R(1) R(2) R(3) R(4) R(5) R(6) R(7) R(8) R(9) R(10) R(11) R(12) R(13) R(14) R(15) R(16) R(17) R(18) R(19) R(20)

#undef RETURN_BY_CHANNEL_SET
#undef R

		return NULL;
}

void Image::Stretchlim(cv::Mat image, float *l, float *h)
{
	float tol_low = 0.01f;
	float tol_high = 0.99f;
	int nbins = 256;

	cv::Mat hist = cv::Mat::zeros(1, nbins, CV_32S);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int val = (int)image.at<cv::Vec3b>(i, j)[0];
			hist.at<int>(0, val)++;
			val = (int)image.at<cv::Vec3b>(i, j)[1];
			hist.at<int>(0, val)++;
			val = (int)image.at<cv::Vec3b>(i, j)[2];
			hist.at<int>(0, val)++;


		}
	}

	cv::Mat cumulativeSum = cv::Mat::zeros(1, 256, CV_32F);
	float cumulativeNum = 0.0f;
	for (int i = 0; i < 256; i++)
	{
		cumulativeNum += (int)hist.at<int>(0, i);
		cumulativeSum.at<float>(0, i) = cumulativeNum;
	}
	float scale = (float)1 / cumulativeNum;
	cumulativeSum *= scale;

	cv::Mat low(cumulativeSum.size(), CV_8UC1);
	cv::compare(cumulativeSum, double(tol_low), low, cv::CMP_GT);
	cv::Mat high(cumulativeSum.size(), CV_8UC1);
	cv::compare(cumulativeSum, double(tol_high), high, cv::CMP_GE);

	int ilow = 0;
	int ihigh = 0;

	for (int i = 0; i < low.cols; i++)
	{
		if ((int)low.at<uchar>(0, i) != 0)
		{
			ilow = i;
			break;
		}
	}

	for (int i = 0; i < high.cols; i++)
	{
		if ((int)high.at<uchar>(0, i) != 0)
		{
			ihigh = i;
			break;
		}
	}

	if (ilow == ihigh)
	{
		ilow = 0; ihigh = nbins - 1;
	}
	(*l) = (float)ilow / (nbins - 1);
	(*h) = (float)ihigh / (nbins - 1);
}
cv::Mat Image::adjustArray(cv::Mat image, float l, float h)
{
	cv::Mat img;
	image.convertTo(img, CV_32F);
	img = img / 255;
	cv::Mat img0(img.size(), img.type());
	cv::Mat res(img.size(), CV_8UC3);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float b = img.at<cv::Vec3f>(i, j)[0];
			float g = img.at<cv::Vec3f>(i, j)[1];
			float r = img.at<cv::Vec3f>(i, j)[2];

			float minb = b;
			float ming = g;
			float minr = r;
			if (b > h){ minb = h; }
			if (g > h){ ming = h; }
			if (r > h){ minr = h; }

			float maxminb = l;
			float maxming = l;
			float maxminr = l;
			if (minb > l){ maxminb = minb; }
			if (ming > l){ maxming = ming; }
			if (minr > l){ maxminr = minr; }

			img0.at<cv::Vec3f>(i, j)[0] = maxminb;
			img0.at<cv::Vec3f>(i, j)[1] = maxming;
			img0.at<cv::Vec3f>(i, j)[2] = maxminr;

			res.at<cv::Vec3b>(i, j)[0] = (int)(255 * (img0.at<cv::Vec3f>(i, j)[0] - l) / (h - l));
			res.at<cv::Vec3b>(i, j)[1] = (int)(255 * (img0.at<cv::Vec3f>(i, j)[1] - l) / (h - l));
			res.at<cv::Vec3b>(i, j)[2] = (int)(255 * (img0.at<cv::Vec3f>(i, j)[2] - l) / (h - l));
		}
	}

	return res;
}

cv::Mat Image::imadjustGray(cv::Mat image)
{
	float l, h;
	StretchlimGray(image, &l, &h);
	cv::Mat result = adjustArrayGray(image, l, h);
	return result;
}

void Image::StretchlimGray(cv::Mat image, float *l, float *h)
{
	float tol_low = 0.01f;
	float tol_high = 0.99f;
	int nbins = 256;

	cv::Mat hist = cv::Mat::zeros(1, nbins, CV_32S);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int val = (int)image.at<uchar>(i, j);
			hist.at<int>(0, val)++;
		}
	}

	cv::Mat cumulativeSum = cv::Mat::zeros(1, 256, CV_32F);
	float cumulativeNum = 0.0;
	for (int i = 0; i < 256; i++)
	{
		cumulativeNum += (int)hist.at<int>(0, i);
		cumulativeSum.at<float>(0, i) = cumulativeNum;
	}
	float scale = (float)1 / cumulativeNum;
	cumulativeSum *= scale;

	cv::Mat low(cumulativeSum.size(), CV_8UC1);
	cv::compare(cumulativeSum, double(tol_low), low, cv::CMP_GT);
	cv::Mat high(cumulativeSum.size(), CV_8UC1);
	cv::compare(cumulativeSum, double(tol_high), high, cv::CMP_GE);

	int ilow = 0;
	int ihigh = 0;

	for (int i = 0; i < low.cols; i++)
	{
		if ((int)low.at<uchar>(0, i) != 0)
		{
			ilow = i;
			break;
		}
	}

	for (int i = 0; i < high.cols; i++)
	{
		if ((int)high.at<uchar>(0, i) != 0)
		{
			ihigh = i;
			break;
		}
	}

	if (ilow == ihigh)
	{
		ilow = 0; ihigh = nbins - 1;
	}
	(*l) = (float)ilow / (nbins - 1);
	(*h) = (float)ihigh / (nbins - 1);
}

cv::Mat Image::adjustArrayGray(cv::Mat image, float l, float h)
{
	cv::Mat img;
	image.convertTo(img, CV_32F);
	img = img / 255;
	cv::Mat img0(img.size(), img.type());
	cv::Mat res(img.size(), CV_8U);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float v = img.at<float>(i, j);

			float minv = v;

			if (v > h){ minv = h; }

			float maxminv = l;

			if (minv > l){ maxminv = minv; }

			img0.at<float>(i, j) = maxminv;

			res.at<uchar>(i, j) = (int)(255 * (img0.at<float>(i, j) - l) / (h - l));
		}
	}
	return res;
}

void Image::warpMatrix(cv::Size sz, double theta, double phi, double gamma, double scale, double fovy, cv::Mat &M, std::vector<cv::Point2f> *corners, std::vector<cv::Point2f> *extraPoints) const
{
#define deg2Rad(x) ((x) * CV_PI / 180.0)

	if (sz.width > 1) {
		sz.width -= 1;
	} else if (sz.height > 1) {
		sz.height -= 1;
	}

	double st = sin(deg2Rad(theta));
	double ct = cos(deg2Rad(theta));
	double sp = sin(deg2Rad(phi));
	double cp = cos(deg2Rad(phi));
	double sg = sin(deg2Rad(gamma));
	double cg = cos(deg2Rad(gamma));

	double halfFovy = 0.5 * fovy;
	double d = hypot(sz.width, sz.height);
	double sideLength = scale * d / cos(deg2Rad(halfFovy));
	double h = d / (2.0 * sin(deg2Rad(halfFovy)));
	double n = h - (d / 2.0);
	double f = h + (d / 2.0);

	cv::Mat F = cv::Mat(4, 4, CV_64FC1);           // Allocate 4x4 transformation matrix F
	cv::Mat Rtheta = cv::Mat::eye(4, 4, CV_64FC1); // Allocate 4x4 rotation matrix around Z-axis by theta degrees
	cv::Mat Rphi = cv::Mat::eye(4, 4, CV_64FC1);   // Allocate 4x4 rotation matrix around X-axis by phi degrees
	cv::Mat Rgamma = cv::Mat::eye(4, 4, CV_64FC1); // Allocate 4x4 rotation matrix around Y-axis by gamma degrees

	cv::Mat T = cv::Mat::eye(4, 4, CV_64FC1);   // Allocate 4x4 translation matrix along Z-axis by -h units
	cv::Mat P = cv::Mat::zeros(4, 4, CV_64FC1); // Allocate 4x4 projection matrix

	// Rtheta
	Rtheta.at<double>(0, 0) = Rtheta.at<double>(1, 1) = ct;
	Rtheta.at<double>(0, 1) = -st; Rtheta.at<double>(1, 0) = st;

	// Rphi
	Rphi.at<double>(1, 1) = Rphi.at<double>(2, 2) = cp;
	Rphi.at<double>(1, 2) = -sp; Rphi.at<double>(2, 1) = sp;

	// Rgamma
	Rgamma.at<double>(0, 0) = Rgamma.at<double>(2, 2) = cg;
	Rgamma.at<double>(0, 2) = sg; Rgamma.at<double>(2, 0) = sg;

	// T
	T.at<double>(2, 3) = -h;

	// P
	P.at<double>(0, 0) = P.at<double>(1, 1) = 1.0 / tan(deg2Rad(halfFovy));
	P.at<double>(2, 2) = -(f + n) / (f - n);
	P.at<double>(2, 3) = -(2.0 * f * n) / (f - n);
	P.at<double>(3, 2) = -1.0;

	//Compose transformations
	F = P * T * Rphi * Rtheta * Rgamma; // Matrix-multiply to produce master matrix

	//Transform 4x4 points
	double ptsIn[4 * 3];
	double ptsOut[4 * 3];
	double halfW =  sz.width / 2, halfH = sz.height / 2;

	ptsIn[0] = -halfW + 1; ptsIn[1]  =  halfH;
	ptsIn[3] =  halfW; ptsIn[4]  =  halfH;
	ptsIn[6] =  halfW; ptsIn[7]  = -halfH + 1;
	ptsIn[9] = -halfW + 1; ptsIn[10] = -halfH + 1;
	// Set Z component to zero for all 4 components
	ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0;

	cv::Mat ptsInMat(1, 4, CV_64FC3, ptsIn);
	cv::Mat ptsOutMat(1, 4, CV_64FC3, ptsOut);

	// Transform points
	perspectiveTransform(ptsInMat, ptsOutMat, F);

    // Get 3x3 transform and warp image
	cv::Point2f ptsInPt2f[4];
	cv::Point2f ptsOutPt2f[4];

	for (int i = 0; i < 4; ++i) {
		cv::Point2f ptIn (static_cast<float>(ptsIn [i * 3 + 0]), static_cast<float>(ptsIn [i * 3 + 1]));
		cv::Point2f ptOut(static_cast<float>(ptsOut[i * 3 + 0]), static_cast<float>(ptsOut[i * 3 + 1]));
		ptsInPt2f[i]  =   ptIn + cv::Point2f(static_cast<float>(halfW), static_cast<float>(halfH));
		ptsOutPt2f[i] = (ptOut + cv::Point2f(1, 1)) * (0.5 * sideLength);
	}

	M = getPerspectiveTransform(ptsInPt2f, ptsOutPt2f);

	//Load corners vector
	if (corners) {
		corners->clear();
		corners->push_back(ptsOutPt2f[0]); // Push Top    Left  corner
		corners->push_back(ptsOutPt2f[1]); // Push Top    Right corner
		corners->push_back(ptsOutPt2f[2]); // Push Bottom Right corner
		corners->push_back(ptsOutPt2f[3]); // Push Bottom Left  corner
	}

	if (extraPoints && !(extraPoints->empty())) {
		
		std::vector<cv::Point2f> extraPointsTransformed;

		cv::perspectiveTransform(*extraPoints, extraPointsTransformed, M);
		extraPoints->clear();

		for (size_t i = 0; i < extraPointsTransformed.size(); ++i) {
			extraPoints->push_back(extraPointsTransformed[i]);
		}
	}

#undef deg2Rad
}

std::ostream & operator <<(std::ostream & os, const SimpleImage::Image &img)
{
	std::string s;
	for (size_t y = 0; y < img.height(); ++y) {
		for (size_t x = 0; x < img.width(); ++x) {
			os << '[';
			for (size_t channel = 0; channel < img.channels(); ++channel) {
				os << img(x, y, channel);
				if (channel != img.channels() - 1) {
					os << " x ";
				}
			}
			os << "] ";
		}
		os << '\n';
	}

	return os;
}

