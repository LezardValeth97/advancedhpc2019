#pragma once

class ChronoGPU {
private:
	cudaEvent_t m_start;
	cudaEvent_t m_end;
	
	bool m_started;
public:
	ChronoGPU();
	~ChronoGPU();

	void	start();
	void	stop();
	float	elapsedTime();
};


