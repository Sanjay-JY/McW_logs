#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

#pragma pack(push, 1)

struct BitmapHeader {
    uint16_t signature;
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
    uint32_t headerSize;
    int32_t  width;
    int32_t  height;
    uint16_t planes;
    uint16_t bitDepth;
    uint32_t compression;
    uint32_t imageSize;
    int32_t  xPixelsPerMeter;
    int32_t  yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

struct Pixel {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
};

struct Bitmap {
    BitmapHeader header;
    std::vector<Pixel> pixels;
};

void readBitmap(const char* filename, Bitmap& bitmap) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file.read(reinterpret_cast<char*>(&bitmap.header), sizeof(BitmapHeader));

    if (bitmap.header.signature != 0x4D42 || bitmap.header.bitDepth != 24 || bitmap.header.compression != 0) {
        std::cerr << "Invalid BMP file format or unsupported features." << std::endl;
        file.close();
        return;
    }

    uint32_t rowSize = (bitmap.header.width * 3 + 3) & ~3;
    bitmap.pixels.resize(bitmap.header.height * bitmap.header.width);

    for (int y = bitmap.header.height - 1; y >= 0; --y) {
        file.read(reinterpret_cast<char*>(bitmap.pixels.data() + y * bitmap.header.width), rowSize);
    }

    file.close();
}

void writeBitmap(const char* filename, const Bitmap& bitmap) {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&bitmap.header), sizeof(BitmapHeader));

    uint32_t rowSize = (bitmap.header.width * 3 + 3) & ~3;

    for (int y = bitmap.header.height - 1; y >= 0; --y) {
        file.write(reinterpret_cast<const char*>(bitmap.pixels.data() + y * bitmap.header.width), rowSize);
    }

    file.close();
}

__global__ void temp(Pixel *x, int width, int height) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int id_y = blockDim.y * blockIdx.y + threadIdx.y;
   

    if (id_x < width && id_y < height) {
        int id = id_y * width + id_x;

        if (blockIdx.x % 2 == 0 && blockIdx.y % 2 == 0) {
            x[id].red = 255 - x[id].red;
            x[id].blue = 255 - x[id].blue;
            x[id].green = 255 - x[id].green;
        }
    }
}

int main() {
    const char* filename = "input1.bmp";
    Bitmap bitmap;

    readBitmap(filename, bitmap);
    std::cout << "Width: " << bitmap.header.width << std::endl;
    std::cout << "Height: " << bitmap.header.height << std::endl;

    Pixel *d_pixel;
    cudaMalloc(&d_pixel, bitmap.header.height * bitmap.header.width * sizeof(Pixel));
    cudaMemcpy(d_pixel, bitmap.pixels.data(), bitmap.header.height * bitmap.header.width * sizeof(Pixel), cudaMemcpyHostToDevice);

    dim3 blockSize(32,32);  
    dim3 gridSize((bitmap.header.width + blockSize.x - 1) / blockSize.x, (bitmap.header.height + blockSize.y - 1) / blockSize.y);

    temp<<<gridSize, blockSize>>>(d_pixel, bitmap.header.width, bitmap.header.height);

    std::vector<Pixel> a(bitmap.header.height * bitmap.header.width);
    cudaMemcpy(a.data(), d_pixel, bitmap.header.height * bitmap.header.width * sizeof(Pixel), cudaMemcpyDeviceToHost);

    std::cout << "Modified Pixel - R: " << static_cast<int>(a[10].red) << " G: " << static_cast<int>(a[10].green) << " B: " << static_cast<int>(a[10].blue) << std::endl;

    Bitmap modifiedBitmap;
    modifiedBitmap.header = bitmap.header;
    modifiedBitmap.pixels = a;

    writeBitmap("output.bmp", modifiedBitmap);

    cudaFree(d_pixel);

    return 0;
}
