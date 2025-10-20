#include <iostream>
#include <string>
#include <unistd.h>
#include <thread>
#include <vector>
#include <chrono>

#include "BmpImage.h"
#include "Tiler.h"
#include "Comparator.h"

void printUsage() {
    std::cout << "Usage: ./image_comparator -i <input.bmp> [-t <num_threads>] [--gpu]\n"
              << "  -i <input.bmp>    : Path to the input BMP image file.\n"
              << "  -t <num_threads>  : Number of CPU threads for comparison. Defaults to hardware concurrency.\n"
              << "  --gpu             : Enable GPU acceleration for color calculations.\n";
}

int main(int argc, char* argv[]) {
    std::string input_file;
    unsigned int num_threads = std::thread::hardware_concurrency();
    bool use_gpu = false;

    int opt;
    while ((opt = getopt(argc, argv, "i:t:")) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 't':
                num_threads = std::stoi(optarg);
                break;
            case '?':
                printUsage();
                return 1;
        }
    }
    
    // A simple way to check for --gpu flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gpu") {
            use_gpu = true;
            break;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: Input file is required." << std::endl;
        printUsage();
        return 1;
    }

    #ifndef USE_GPU
    if (use_gpu) {
        std::cerr << "Warning: Program was compiled without GPU support. Falling back to CPU." << std::endl;
        use_gpu = false;
    }
    #endif

    auto start_time = std::chrono::high_resolution_clock::now();

    BmpImage image;
    if (!image.load(input_file)) {
        return 1;
    }
    std::cout << "Loaded image: " << image.getWidth() << "x" << image.getHeight() << std::endl;

    const uint32_t TILE_SIZE = 512;
    std::vector<Tile> tiles = Tiler::createTiles(image, TILE_SIZE);
    std::cout << "Created " << tiles.size() << " tiles." << std::endl;

    Comparator comparator(tiles, use_gpu);
    std::vector<ComparisonResult> results = comparator.run(num_threads);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Processing finished in " << elapsed.count() << " seconds." << std::endl;

    // Print first 10 results as an example
    std::cout << "Top 10 Results (Tile1, Tile2, DeltaE):" << std::endl;
    for (size_t i = 0; i < std::min(results.size(), (size_t)10); ++i) {
        const auto& r = results[i];
        std::cout << "(" << r.tile1_id << ", " << r.tile2_id << "): " << r.color_difference << std::endl;
    }

    return 0;
}
