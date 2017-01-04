#pragma once

#include <iostream>

namespace SimpleImage
{
	enum DataType
	{
		UNSIGNED_CHAR, SIGNED_CHAR, UNSIGNED_SHORT, SIGNED_SHORT, INT, FLOAT, DOUBLE,
		UCHAR = UNSIGNED_CHAR, CHAR = SIGNED_CHAR, SCHAR = SIGNED_CHAR, USHORT = UNSIGNED_SHORT, SHORT = SIGNED_SHORT, SSHORT = SIGNED_SHORT
	};

	class ImageType
	{
		int data;

		DataType type;
		size_t   m_channels;

	public:

		ImageType(DataType dataType, size_t channels = 1);
		ImageType(size_t cvCode);

		~ImageType();

		int cvCode() const;

		size_t depth() const;
		size_t channels() const;

		std::string toString() const;
		std::string toCVString() const;

		DataType dataType() const;
	};

}

std::ostream &operator <<(std::ostream &, const SimpleImage::ImageType &);