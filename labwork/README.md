MI 3.22a Advanced Programming for HPC Labwork
=====================================================

What?
---------

This is a skeleton labwork for students. We will use this labwork throughout the course.


Why?
---------

It provides basic building block for your labwork. You therefore don't need to reinvent the wheel, only focus on programming for HPC techniques.


How?
----------

1. Dependencies

To build it, you will need the following dependencies:

* OS: GNU/Linux or macOS. Windows is not supported by the lecturer, as usual :-)
* CUDA SDK 7+.
* Compiler: ```nvcc``` (bundled with CUDA SDK) and ```gcc``` 4.8+ (for C++11 standard support).
* ```libjpeg``` to encode/decode JPEG files.

Optional:

* ```cmake``` can help you build even easier. A ```CMakeLists.txt``` is provided. If not, manual ```nvcc``` is still applicable.

2. Build

Any ```cmake``` project is typically built as follows:

```bash
mkdir build
cd build
cmake ..
make -j
```

In this case, the intermediate object files (```*.o```) will not be put in the same directory as your source, but in a separated ```build``` directory instead.

3. Run and test

To test, you will also need a NVIDIA GPU with proper driver installed.

Check if you have it: ```nvidia-smi```

To run your labwork: ```./labwork```

Extras
---------------

If you have problems of compatibility between GCC 8+ with CUDA 10.0, try installing older version of gcc (e.g. 6) and use these before ```cmake ..``

```
export CC=/usr/bin/gcc-6
export CXX=/usr/bin/g++-6
```
