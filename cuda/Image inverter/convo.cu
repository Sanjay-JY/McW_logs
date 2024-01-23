// #include<iostream>
// #include<cassert>
// #include<cstdlib>
// #include <fstream>
// #include <cstdint>
// #include <vector>
// #define maskdim 3
// #define maskoffset (maskdim/2)
// #define N 475
// __constant__ int mask[maskdim*maskdim];
// #pragma pack(push, 1)
// struct BitmapHeader {
//     uint16_t signature;
//     uint32_t fileSize;
//     uint32_t reserved;
//     uint32_t dataOffset;
//     uint32_t headerSize;
//     int32_t  width;
//     int32_t  height;
//     uint16_t planes;
//     uint16_t bitDepth;
//     uint32_t compression;
//     uint32_t imageSize;
//     int32_t  xPixelsPerMeter;
//     int32_t  yPixelsPerMeter;
//     uint32_t colorsUsed;
//     uint32_t colorsImportant;
// };
// #pragma pack(pop)
// struct Pixel {
//     uint8_t blue;
//     uint8_t green;
//     uint8_t red;
// };
// struct Bitmap {
//     BitmapHeader header;
//     std::vector<Pixel> pixels;
// };
// void readBitmap(const char* filename, Bitmap& bitmap) {
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }
//     file.read(reinterpret_cast<char*>(&bitmap.header), sizeof(BitmapHeader));
//     if (bitmap.header.signature != 0x4D42 || bitmap.header.bitDepth != 24 || bitmap.header.compression != 0) {
//         std::cerr << "Invalid BMP file format or unsupported features." << std::endl;
//         file.close();
//         return;
//     }
//     uint32_t rowSize = (bitmap.header.width * 3 + 3) & ~3;
//     bitmap.pixels.resize(bitmap.header.height * bitmap.header.width);
//     for (int y = bitmap.header.height - 1; y >= 0; --y) {
//         file.read(reinterpret_cast<char*>(bitmap.pixels.data() + y * bitmap.header.width), rowSize);
//     }
//     file.close();
// }
// void writeBitmap(const char* filename, const Bitmap& bitmap) {
//     std::ofstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file for writing: " << filename << std::endl;
//         return;
//     }
//     file.write(reinterpret_cast<const char*>(&bitmap.header), sizeof(BitmapHeader));
//     uint32_t rowSize = (bitmap.header.width * 3 + 3) & ~3;
//     for (int y = bitmap.header.height - 1; y >= 0; --y) {
//         file.write(reinterpret_cast<const char*>(bitmap.pixels.data() + y * bitmap.header.width), rowSize);
//     }
//     file.close();
// }
// __global__ void convolution_2d(Pixel *matrix, Pixel *result)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int startrow = row - maskoffset;
//     int startcol = col - maskoffset;
//     int temp1 = 0;
//     int temp2 = 0;
//     int temp3 = 0;
//     for (int i = 0; i < maskdim; i++)
//     {
//         for (int j = 0; j < maskdim; j++)
//         {
//             if (startrow + i >= 0 && (startrow + i) < N && startcol + j >= 0 && (startcol + j) < N)
//             {
//                 temp1 += matrix[(startrow + i) * N + (startcol + j)].red * mask[i * maskdim + j];
//                 temp2 += matrix[(startrow + i) * N + (startcol + j)].blue * mask[i * maskdim + j];
//                 temp3 += matrix[(startrow + i) * N + (startcol + j)].green * mask[i * maskdim + j];
//             }
//         }
//     }
//     if (row < N && col < N)
//     {
//         result[row * N + col].red = temp1;
//         result[row * N + col].blue = temp2;
//         result[row * N + col].green = temp3;
//     }
// }
// int main()
// {
//     int *result = new int[N * N];
//     int hmask[3 * 3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
//     cudaMemcpyToSymbol(mask, hmask, sizeof(int) * maskdim * maskdim);
//     const char *filename = "input.bmp";
//     Bitmap bitmap;
//     readBitmap(filename, bitmap);
//     std::cout << "Width: " << bitmap.header.width << std::endl;
//     Pixel *d_pixel;
//     Pixel *d_result;
//     cudaMalloc(&d_pixel, bitmap.header.height * bitmap.header.width * sizeof(Pixel));
//     cudaMalloc(&d_result, bitmap.header.height * bitmap.header.width * sizeof(Pixel));
//     cudaMemcpy(d_pixel, bitmap.pixels.data(), bitmap.header.height * bitmap.header.width * sizeof(Pixel), cudaMemcpyHostToDevice);
//     std::cout << "Width: " << bitmap.header.width << std::endl;
//     std::cout << "Height: " << bitmap.header.height << std::endl;
//     dim3 blockSize(16, 16);
//     dim3 gridSize((bitmap.header.width + blockSize.x - 1) / blockSize.x, (bitmap.header.height + blockSize.y - 1) / blockSize.y);
//     convolution_2d<<<gridSize, blockSize>>>(d_pixel, d_result);
//     std::vector<Pixel> a(bitmap.header.height * bitmap.header.width);
//     cudaMemcpy(a.data(), d_result, bitmap.header.height * bitmap.header.width * sizeof(Pixel), cudaMemcpyDeviceToHost);
//     Bitmap modifiedBitmap;
//     modifiedBitmap.header = bitmap.header;
//     modifiedBitmap.pixels = a;
//     writeBitmap("output.bmp", modifiedBitmap);
//     cudaFree(d_pixel);
//     cudaFree(d_result);
//     delete[] result;
//     return 0;
// }


#include<iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature;
    uint32_t file_size;
    uint32_t reserved;
    uint32_t data_offset;
    uint32_t header_size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bit_count;
    uint32_t compression;
    uint32_t image_size;
    int32_t x_pixels_per_meter;
    int32_t y_pixels_per_meter;
    uint32_t colors_used;
    uint32_t colors_important;
};

struct Pixel {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
};

#pragma pack(pop)

__global__ void edge(Pixel* pixels,int* mask,int size,int m,Pixel* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r = m/2;
    int row_i = row - r;
    int col_i = col - r;
    int rval = 0;
    int gval = 0;
    int bval = 0;
    if (row_i >= 0 && row_i < size && col_i >= 0 && col_i < size)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                rval += pixels[(row_i + i) * size + (col_i + j)].red * mask[i * m + j];
                gval += pixels[(row_i + i) * size + (col_i + j)].blue * mask[i * m + j];
                bval += pixels[(row_i + i) * size + (col_i + j)].green * mask[i * m + j];
            }
        }
    }

    if (row < size && col < size)
    {
        result[row * size + col].red = rval;
        result[row * size + col].blue = gval;
        result[row * size + col].green = bval;
    }
}


int main() {

    int mask[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    std::ifstream file("input1.bmp", std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    BMPHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

    if (header.signature != 0x4D42) {
        std::cerr << "Not a BMP file!" << std::endl;
        return 1;
    }

    if (header.bit_count != 24 || header.compression != 0) {
        std::cerr << "Unsupported BMP format!" << std::endl;
        return 1;
    }

    std::vector<Pixel> pixels(header.width * header.height);
    const int padding = 4 - ((header.width * 3) % 4); 
    file.seekg(header.data_offset);
    for (int y = header.height - 1; y >= 0; --y) {
        file.read(reinterpret_cast<char*>(&pixels[y * header.width]), header.width * sizeof(Pixel));
        file.seekg(padding, std::ios::cur);
    }
    file.close();

    std::cout<<"Width: "<<header.width<<"\n";
    std::cout<<"Height: "<<header.height<<"\n";
    std::cout<<"Length of pixels: "<<pixels.size()<<"\n";
    
    std::vector<Pixel> ans_pixels(header.width * header.height);
    Pixel* d_pixels;
    Pixel* a_pixels;
    int* d_mask;
    int N=header.width*header.height*sizeof(Pixel);

    cudaMalloc(&d_pixels, N );
    cudaMalloc(&a_pixels, N );
    cudaMalloc(&d_mask, 9*sizeof(int) );
    cudaMemcpy(d_pixels, pixels.data(),N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,mask,9*sizeof(int),cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((header.width + block.x - 1) / block.x, (header.height + block.y - 1) / block.y);

    edge<<<grid, block>>>(d_pixels,d_mask,header.width,3,a_pixels);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaError) << std::endl;
    }
    cudaMemcpy(ans_pixels.data(), a_pixels, N , cudaMemcpyDeviceToHost);

    std::ofstream output_file("output_edge.bmp", std::ios::out | std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error creating output file!" << std::endl;
        return 1;
    }
    
    output_file.write(reinterpret_cast<char*>(&header), sizeof(BMPHeader));
    for (int y = header.height - 1; y >= 0; --y) {
        output_file.write(reinterpret_cast<char*>(&ans_pixels[y*header.width]), header.width * sizeof(Pixel));
        output_file.seekp(padding, std::ios::cur); 
    }
    output_file.close();

    std::cout << "Edge detection completed successfully!" << std::endl;

    return 0;
}
