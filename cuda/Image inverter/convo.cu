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
