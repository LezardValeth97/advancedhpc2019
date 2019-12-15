#pragma once

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cstring>

typedef unsigned char uchar;

class PPMBitmap {
public:
	struct RGBcol {
		uchar r, g, b;

		explicit RGBcol() : r( 0 ), g ( 0 ), b ( 0 ) {}
		explicit RGBcol( uchar red, uchar green, uchar blue ) 
			: r( red ), g( green ), b( blue ) {};
	};
private:
	union {
		uchar	*m_ptr;
		RGBcol	*m_pixels;
	};

	int m_width;
	int m_height;

public:
	//
	PPMBitmap() = delete;
	PPMBitmap( const int width, const int height );
	PPMBitmap( const char *const name );
	~PPMBitmap() { delete m_ptr; }
	PPMBitmap& operator=(const PPMBitmap&);
	PPMBitmap(const PPMBitmap&);

	int		getWidth()	const	{ return m_width; }
	int		getHeight()	const	{ return m_height; }
	size_t	getSizeInBytes()	const	{ return m_width * m_height * 3; }
	uchar *	getPtr()	const	{ return m_ptr; }

	RGBcol getPixel( const unsigned x, const unsigned y ) const {
		return m_pixels[ x + y * m_width];
	}
	void setPixel( const unsigned x, const unsigned y, const RGBcol &RGBcol ) {
		m_pixels[ x + y * m_width ] = RGBcol;
	}
	
	void getLine( std::ifstream &file, std::string &s ) const;

	void saveTo( const char *name ) const;
};


