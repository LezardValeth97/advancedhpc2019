#pragma once

#include <chrono>

class ChronoCPU {
	// need C++ 11 to work
	std::chrono::time_point<std::chrono::high_resolution_clock> start_chr;
	std::chrono::time_point<std::chrono::high_resolution_clock> stop_chr;

	bool is_started;

public:
	ChronoCPU();
	~ChronoCPU();

	void		start();
	void		stop();
	long long	elapsedTimeInSeconds();
	long long	elapsedTimeInMilliSeconds();
	long long	elapsedTimeInMicroSeconds();
};


