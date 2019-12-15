#!/bin/bash
@mkdir build
@cd build
@cmake -DCUDA_SAMPLES_INC="/usr/local/cuda/samples/common/inc"  ..
@cmake --build . --config Release
