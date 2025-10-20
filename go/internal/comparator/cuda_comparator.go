package comparator

import (
	"fmt"
	"image"
	"log"
	"math"
	"runtime"

	"gocv.io/x/gocv"
)

// CUDAComparator uses an NVIDIA GPU to compare images using CUDA.
// It requires OpenCV to be installed with CUDA support.
type CUDAComparator struct{}

// NewCUDAComparator creates a new CUDA comparator and checks for CUDA devices.
func NewCUDAComparator() (*CUDAComparator, error) {
	if gocv.GetCudaEnabledDeviceCount() == 0 {
		return nil, fmt.Errorf("no CUDA-enabled device found. Please ensure OpenCV is compiled with CUDA support and a compatible GPU is available")
	}
	return &CUDAComparator{}, nil
}

// Compare calculates the average CIE Delta E 76 color difference between two images on the GPU.
// This is a simpler metric than CIEDE2000 but is well-suited for GPU acceleration.
func (c *CUDAComparator) Compare(imgA, imgB image.Image) float64 {
	defer func() {
		if r := recover(); r != nil {
			// gocv can panic on certain image types, so we recover to avoid crashing.
			log.Printf("Recovered from panic in gocv: %v. This may be due to an unsupported image format for gocv conversion.", r)
		}
	}()

	matA, err := gocv.ImageToMatRGB(imgA)
	if err != nil {
		log.Printf("Error converting image A to Mat: %v", err)
		return math.MaxFloat64
	}
	defer matA.Close()

	matB, err := gocv.ImageToMatRGB(imgB)
	if err != nil {
		log.Printf("Error converting image B to Mat: %v", err)
		return math.MaxFloat64
	}
	defer matB.Close()

	if matA.Cols() != matB.Cols() || matA.Rows() != matB.Rows() {
		return math.MaxFloat64
	}

	pixelCount := float64(matA.Cols() * matA.Rows())
	if pixelCount == 0 {
		return 0
	}

	gpuMatA := gocv.NewGpuMat()
	defer gpuMatA.Close()
	gpuMatA.Upload(matA)

	gpuMatB := gocv.NewGpuMat()
	defer gpuMatB.Close()
	gpuMatB.Upload(matB)

	// Convert from RGB to LAB color space on GPU
	gpuLabA := gocv.NewGpuMat()
	defer gpuLabA.Close()
	gocv.CudaCvtColor(gpuMatA, &gpuLabA, gocv.ColorRGBToLab)

	gpuLabB := gocv.NewGpuMat()
	defer gpuLabB.Close()
	gocv.CudaCvtColor(gpuMatB, &gpuLabB, gocv.ColorRGBToLab)

	// Calculate difference in LAB space: (L2-L1), (a2-a1), (b2-b1)
	diff := gocv.NewGpuMat()
	defer diff.Close()
	gocv.CudaSubtract(gpuLabA, gpuLabB, &diff)

	// Square the differences: (L2-L1)^2, (a2-a1)^2, (b2-b1)^2
	gocv.CudaPow(diff, 2, &diff)

	// Split channels
	channels := gocv.CudaSplit(diff)
	defer func() {
		for _, ch := range channels {
			ch.Close()
		}
	}()

	// Sum of squared differences: (L2-L1)^2 + (a2-a1)^2
	sumSq := gocv.NewGpuMat()
	defer sumSq.Close()
	gocv.CudaAdd(channels[0], channels[1], &sumSq)
	// Sum of squared differences: (L2-L1)^2 + (a2-a1)^2 + (b2-b1)^2
	gocv.CudaAdd(sumSq, channels[2], &sumSq)

	// Square root to get Delta E 76 for each pixel
	deltaE := gocv.NewGpuMat()
	defer deltaE.Close()
	gocv.CudaSqrt(sumSq, &deltaE)

	// Sum all Delta E values
	sumScalar := gocv.CudaSum(deltaE)
	totalDifference := sumScalar.Val1

	runtime.KeepAlive(imgA, imgB, matA, matB, gpuMatA, gpuMatB, gpuLabA, gpuLabB, diff, channels, sumSq, deltaE)

	return totalDifference / pixelCount
}
