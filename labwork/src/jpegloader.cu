#include <include/jpegloader.h>

// Some supporting structs and methods
struct ErrorManager {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};
typedef struct ErrorManager * my_error_ptr;

void errorHandler(j_common_ptr info) {
    my_error_ptr myerr = (my_error_ptr) info->err;
    (*info->err->output_message) (info);
    longjmp(myerr->setjmp_buffer, 1);
}

/**
 * Perform a load jpeg file into a RGB 1-dimension char [] array
 */
JpegInfo *JpegLoader::load(std::string filename) {
    struct ErrorManager jerr;
    struct jpeg_decompress_struct info;
    FILE * infile;
    JSAMPARRAY buffer;
    int rowStride;

    // open file
    if ((infile = fopen(filename.c_str(), "rb")) == NULL) {
        return NULL;
    }

    // error handler
    info.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = errorHandler;

    // create structs and initialization
    jpeg_create_decompress(&info);
    jpeg_stdio_src(&info, infile);
    jpeg_read_header(&info, TRUE);
    jpeg_start_decompress(&info);
    rowStride = info.output_width * info.output_components;
    if (info.out_color_components < 3) {
        // RGB only. We don't use grayscale images
        // jpeg_finish_decompress(&info);
        jpeg_destroy_decompress(&info);
        fclose(infile);
        return NULL;
    }

    int bufferSize = info.output_height * rowStride;
    unsigned char *imageBuffer = new unsigned char[bufferSize]();

    buffer = (*info.mem->alloc_sarray) ((j_common_ptr) &info, JPOOL_IMAGE, rowStride, 1);
    while (info.output_scanline < info.output_height) {
        jpeg_read_scanlines(&info, buffer, 1);
        memcpy(&imageBuffer[(info.output_scanline - 1) * rowStride], *buffer, rowStride);
    }
    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    fclose(infile);

    JpegInfo *jpegInfo = new JpegInfo();
    jpegInfo->buffer = (char*)imageBuffer;
    jpegInfo->width = info.output_width;
    jpegInfo->height = info.output_height;

    return jpegInfo;
}

/**
 * Save an 1d array of width * height pixel data into a jpeg
 */
bool JpegLoader::save(std::string filename, char *imageBuffer, int width, int height, int quality) {
    struct jpeg_compress_struct info;
    struct jpeg_error_mgr jerr;
    FILE * outfile;
    JSAMPROW rowPointer[1];
    int rowStride;

    // allocate and initialize JPEG compression
    info.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&info);

    // prepare the file
    if ((outfile = fopen(filename.c_str(), "wb")) == NULL) {
        return FALSE;
    }
    jpeg_stdio_dest(&info, outfile);

    // set parameters for compression
    info.image_width = width;
    info.image_height = height;
    info.input_components = 3;
    info.in_color_space = JCS_RGB;
    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, quality, TRUE);

    // do the compression
    jpeg_start_compress(&info, TRUE);
    rowStride = width * 3;  // RGB channels

    while (info.next_scanline < info.image_height) {
        rowPointer[0] = (JSAMPROW) &imageBuffer[info.next_scanline * rowStride];
        jpeg_write_scanlines(&info, rowPointer, 1);
    }

    // finish
    jpeg_finish_compress(&info);
    fclose(outfile);
    jpeg_destroy_compress(&info);
    return TRUE;
}
