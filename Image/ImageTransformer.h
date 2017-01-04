#pragma once

#include "opencv2/opencv.hpp"
#include "Image.h"

#ifdef TINY_DNN
#include "tiny_dnn/tiny_dnn.h"
#endif

#ifdef QT
#include <QImage>
#include <QPixmap>
#endif


namespace SimpleImage
{

	struct ImageTransformer
	{
#ifdef TINY_DNN
		
		static tiny_dnn::vec_t    image2vec_t   (const Image &image);
		static tiny_dnn::tensor_t image2tensor_t(const Image &image);

		static SimpleImage::Image vec_t2image   (const tiny_dnn::vec_t &vec,       Size size);
		static SimpleImage::Image tensor_t2image(const tiny_dnn::tensor_t &tensor, Size size);

#endif

#ifdef QT

		static QImage  image2qimage (const Image &image);
		static QPixmap image2qpixmap(const Image &image);

		static SimpleImage::Image qimage2image (const QImage &qimage);
		static SimpleImage::Image qpixmap2image(const QPixmap &qpixmap);

#endif
	};

}