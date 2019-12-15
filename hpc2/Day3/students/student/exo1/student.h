#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>

// ==========================================================================================
// a simple colored object ... having mainly a color ;-)
struct ColoredObject {
	// definition of a color
	enum Color : long long
	{ 
		WHITE, BLUE, RED
	};
	// attribute color
	Color color;
	// some data ...
	int pad[3];
	// constructor
	__device__ __host__
	ColoredObject(ColoredObject::Color color, const int p0) 
		: color(color) 
	{ 
		pad[0] = p0; 
	}
	// cast operator -> returns a Color
	__device__ __host__
	operator ColoredObject::Color() const { return color; }
	// default ...
	ColoredObject() = default;
	~ColoredObject() = default;
	ColoredObject& operator=(const ColoredObject&) = default;
	ColoredObject(const ColoredObject&) = default;
};

class StudentWork1 : public StudentWork
{
public:
	// Given a vector of ColoredObject, this function returns a new vector that contains only blue objects
	thrust::device_vector<ColoredObject> compactBlue(
		const thrust::device_vector<ColoredObject>& input
	);

	bool isImplemented() const ;

	StudentWork1() = default; 
	StudentWork1(const StudentWork1&) = default;
	~StudentWork1() = default;
	StudentWork1& operator=(const StudentWork1&) = default;
};