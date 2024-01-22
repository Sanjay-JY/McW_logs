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

__global__ void invertImage(Pixel* pixels,int size) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;
    if(id<size){
        pixels[id].red = 255 - pixels[id].red;
        pixels[id].green = 255 - pixels[id].green;
        pixels[id].blue = 255 - pixels[id].blue;
    }
}


int main() {
    std::ifstream file("input.bmp", std::ios::in | std::ios::binary);
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
    int N=header.width*header.height*sizeof(Pixel);

    cudaMalloc(&d_pixels, N );
    cudaMemcpy(d_pixels, pixels.data(),N,cudaMemcpyHostToDevice);

    int thr_per_blk = 1024;
    dim3 dimBlock(32,32,1);
    dim3 dimGrid();
	int blk_in_grid = ceil(float(header.width*header.height) / thr_per_blk);   //used N instead of header.width*header.height which lead to illegal memory access

    invertImage<<<blk_in_grid, thr_per_blk>>>(d_pixels, N);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaError) << std::endl;
    }
    cudaMemcpy(ans_pixels.data(), d_pixels, N , cudaMemcpyDeviceToHost);

    std::ofstream output_file("output_inverted.bmp", std::ios::out | std::ios::binary);
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

    std::cout << "Image inversion completed successfully!" << std::endl;

    return 0;
}
