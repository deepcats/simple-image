#include "ImageTransformer.h"

using namespace SimpleImage;

#ifdef TINY_DNN

tiny_dnn::vec_t ImageTransformer::image2vec_t(const Image &image)
{
	tiny_dnn::vec_t vec;
	cv::Mat mat(static_cast<cv::Mat>(image));

	if (mat.channels() != 1) {
		throw "mat must be grayscale!";
	}

	cv::Mat matToWork = mat;

	if (matToWork.type() == CV_MAKETYPE(CV_8U, 3)) {
		cv::cvtColor(mat, matToWork, CV_BGR2GRAY);
	}

	matToWork.convertTo(matToWork, CV_32F);

	vec.clear();
	vec.resize(matToWork.total());
	for (int x = 0; x < matToWork.cols; ++x) {
		for (int y = 0; y < matToWork.rows; ++y) {
			vec[y * matToWork.cols + x] = matToWork.at<float>(y, x) / 255.0f;
		}
	}

	return vec;
}

tiny_dnn::tensor_t ImageTransformer::image2tensor_t(const Image &image)
{
	tiny_dnn::tensor_t tensor;

	size_t channelCount = image.channels();

	std::vector<Image> channels = image.split();

	tensor.clear();
	tensor.resize(channelCount);
	for (size_t channel = 0; channel < channelCount; ++channel) {
		tensor[channel] = ImageTransformer::image2vec_t(channels[channel]);
	}

	return tensor;
}

Image ImageTransformer::vec_t2image(const tiny_dnn::vec_t &vec, Size size)
{
	if (vec.size() != size.area()) {
		throw "vec.size() != size.area()";
	}

	cv::Mat mat;
	mat.create(cv::Size(size.width(), size.height()), CV_32F);
	for (size_t x = 0; x < size.width(); ++x) {
		for (size_t y = 0; y < size.height(); ++y) {
			mat.at<tiny_dnn::float_t>(y, x) = vec[y * size.width() + x] * 255.0f;
		}
	}

	mat.convertTo(mat, CV_8U);

	return Image(mat);
}

Image ImageTransformer::tensor_t2image(const tiny_dnn::tensor_t &tensor, Size size)
{
	for (size_t i = 0; i < tensor.size(); ++i) {
		if (tensor[i].size() != size.area()) {
			throw "tensor[i].size() != size.area()";
		}
	}

	size_t channelCount = tensor.size();

	std::vector<Image> channels;
	channels.resize(tensor.size());
	for (size_t channel = 0; channel < channelCount; ++channel) {
		channels[channel] = ImageTransformer::vec_t2image(tensor[channel], size);
	}

	return Image::merge(channels);
}

#endif

#ifdef QT

QImage ImageTransformer::image2qimage(const Image &image)
{
	cv::Mat temp;

	int cvType = image.type().cvCode();
	switch (cvType) {

		case CV_8UC4:

			cv::cvtColor(static_cast<cv::Mat>(image), temp, CV_BGRA2RGBA);
			break;

		case CV_8UC3:

			cv::cvtColor(static_cast<cv::Mat>(image), temp, CV_BGR2RGB);
			break;

		case CV_8UC1:

			cv::cvtColor(static_cast<cv::Mat>(image), temp, CV_GRAY2RGB);
			break;

		default: 
			throw "Cannot convert Image of this type to QImage!";

	}


	QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
	dest.bits();

	return dest;
}

QPixmap ImageTransformer::image2qpixmap(const Image &image)
{
	return QPixmap::fromImage(ImageTransformer::image2qimage(image));
}

Image ImageTransformer::qimage2image(const QImage &qimage)
{
	int cvTypeCode = -1;

	if (qimage.isGrayscale()) {
		cvTypeCode = CV_8UC1;
	} else {
		cvTypeCode = CV_8UC4;
	}

	return cv::Mat(qimage.height(), qimage.width(), cvTypeCode, (uchar*)qimage.bits(), qimage.bytesPerLine());
}

Image ImageTransformer::qpixmap2image(const QPixmap &qpixmap)
{
	return ImageTransformer::qimage2image(qpixmap.toImage());
}

#endif