#include <include/timer.h>
#include <stdlib.h>

void Timer::start() {
    startTime.tv_sec = startTime.tv_usec = 0;
    gettimeofday(&startTime, NULL);
}

double Timer::getElapsedTimeInMilliSec() {
    timeval endCount;
    gettimeofday(&endCount, NULL);

    double startTimeInMicroSec = (startTime.tv_sec * 1000000.0) + startTime.tv_usec;
    double endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
    return (endTimeInMicroSec - startTimeInMicroSec) * 0.001;
}