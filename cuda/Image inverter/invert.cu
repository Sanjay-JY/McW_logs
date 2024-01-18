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

__global__ void invertImage(Pixel* pixels,int size,int* temp) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;
    temp[id]=threadIdx.x;
    if(id<size){
        // temp[id]=static_cast<int>(pixels[id].blue);
        pixels[id].red = 255 - pixels[id].red;
        pixels[id].green = 255 - pixels[id].green;
        pixels[id].blue = 255 - pixels[id].blue;
    }
    //__syncthreads();
}

// void invertImage(std::vector<Pixel>& pixels) {
//     for (auto& pixel : pixels) {
//         pixel.red = 255 - pixel.red;
//         pixel.green = 255 - pixel.green;
//         pixel.blue = 255 - pixel.blue;
//     }
// }

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
    const int padding = 4 - ((header.width * 3) % 4); // Calculate padding
    file.seekg(header.data_offset);
    for (int y = header.height - 1; y >= 0; --y) {
        file.read(reinterpret_cast<char*>(&pixels[y * header.width]), header.width * sizeof(Pixel));
        file.seekg(padding, std::ios::cur); // Skip padding bytes
    }
    file.close();

    int* temp;
    std::cout<<"Width: "<<header.width<<"\n";
    std::cout<<"Height: "<<header.height<<"\n";
    std::cout<<"Length of pixels: "<<pixels.size()<<"\n";
    std::vector<Pixel> d_pixels(header.width * header.height);
    Pixel* ans_pixels;
    int N=header.width*header.height;
    std::vector<int> temp1(N);
    std::cout<<"Before: "<<static_cast<int>(pixels[0].blue)<<"\n";
    std::cout<<"Before: "<<static_cast<int>(d_pixels[0].blue)<<"\n";

    cudaMalloc(reinterpret_cast<void**>(&ans_pixels),N);
    cudaMalloc(reinterpret_cast<void**>(&temp), N * sizeof(int));

    cudaMemcpy(ans_pixels, pixels.data(),N,cudaMemcpyHostToDevice);
    
    int thr_per_blk = 256;
	int blk_in_grid = ceil(float(N) / thr_per_blk);

    invertImage<<<blk_in_grid, thr_per_blk>>>(ans_pixels, N, temp);

    //cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaError) << std::endl;
        // Additional error handling if needed
    }
    cudaMemcpy(d_pixels.data(), ans_pixels, N , cudaMemcpyDeviceToHost);
    cudaMemcpy(temp1.data(), temp, N , cudaMemcpyDeviceToHost);

    std::cout<<"After: "<<static_cast<int>(pixels[0].blue)<<"\n";
    std::cout<<"After: "<<temp1[1]<<"\n";

    std::ofstream output_file("output_inverted.bmp", std::ios::out | std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error creating output file!" << std::endl;
        return 1;
    }

    output_file.write(reinterpret_cast<char*>(&header), sizeof(BMPHeader));
    for (int y = header.height - 1; y >= 0; --y) {
        output_file.write(reinterpret_cast<char*>(&d_pixels[y * header.width]), header.width * sizeof(Pixel));
        output_file.seekp(padding, std::ios::cur); // Add padding bytes
    }
    output_file.close();

    std::cout << "Image inversion completed successfully!" << std::endl;

    return 0;
}
