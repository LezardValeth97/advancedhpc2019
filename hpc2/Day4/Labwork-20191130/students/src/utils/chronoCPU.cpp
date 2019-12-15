#include "chronoCPU.hpp"

#include <iostream>

ChronoCPU::ChronoCPU() 
	: is_started( false ) {
}

ChronoCPU::~ChronoCPU() {
	if ( is_started ) {
		stop();
		std::cerr << "ChronoCPU::~ChronoCPU(): chrono wasn't turned off!" << std::endl; 
	}
}

void ChronoCPU::start() {
	if ( !is_started ) {
		is_started = true;
		start_chr = std::chrono::high_resolution_clock::now();
	}
	else
		std::cerr << "ChronoCPU::start(): chrono wasn't turned off!" << std::endl;
}

void ChronoCPU::stop() {
	if ( is_started ) {
		is_started = false;
		stop_chr = std::chrono::high_resolution_clock::now();
	}
	else
		std::cerr << "ChronoCPU::stop(): chrono wasn't started!" << std::endl;
}

long long ChronoCPU::elapsedTimeInSeconds() {  
	if ( is_started ) {
		std::cerr << "ChronoCPU::elapsedTime(): chrono wasn't turned off!" << std::endl;
		stop_chr = std::chrono::high_resolution_clock::now();
	}
	return 
		std::chrono::duration_cast<std::chrono::seconds>(
			stop_chr - start_chr).count(); 
}

long long ChronoCPU::elapsedTimeInMilliSeconds() {  
	if ( is_started ) {
		std::cerr << "ChronoCPU::elapsedTime(): chrono wasn't turned off!" << std::endl;
		stop_chr = std::chrono::high_resolution_clock::now();
	}
	return 
		std::chrono::duration_cast<std::chrono::milliseconds>(
			stop_chr - start_chr).count(); 
}

long long ChronoCPU::elapsedTimeInMicroSeconds() {  
	if ( is_started ) {
		std::cerr << "ChronoCPU::elapsedTime(): chrono wasn't turned off!" << std::endl;
		stop_chr = std::chrono::high_resolution_clock::now();
	}
	return 
		std::chrono::duration_cast<std::chrono::microseconds>(
			stop_chr - start_chr).count(); 
}
