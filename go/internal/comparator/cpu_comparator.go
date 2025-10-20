package comparator

import (
	"image"
	"math"

	"github.com/lucasb-eyer/go-colorful"
)

// CPUComparator uses the CPU to compare images.
type CPUComparator struct{}

// Compare calculates the difference between two images on the CPU.
// It assumes images are *image.RGBA for optimized pixel access.
func (c *CPUComparator) Compare(imgA, imgB image.Image) float64 {
	rgbaA := imgA.(*image.RGBA)
	rgbaB := imgB.(*image.RGBA)
	bounds := rgbaA.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	if width != rgbaB.Bounds().Dx() || height != rgbaB.Bounds().Dy() {
		return math.MaxFloat64
	}

	pixelCount := float64(width * height)
	if pixelCount == 0 {
		return 0
	}

	var totalDifference float64
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// The sub-images created by `SubImage` on an `*image.RGBA` are normalized
			// to have their bounds start at (0,0) and their `Pix` slice adjusted accordingly.
			// Therefore, the offset is calculated directly from the relative y/x coordinates.
			offset := y*rgbaA.Stride + x*4
			c1 := colorful.Color{
				R: float64(rgbaA.Pix[offset+0]) / 255.0,
				G: float64(rgbaA.Pix[offset+1]) / 255.0,
				B: float64(rgbaA.Pix[offset+2]) / 255.0,
			}
			c2 := colorful.Color{
				R: float64(rgbaB.Pix[offset+0]) / 255.0, // Stride is the same for both sub-images
				G: float64(rgbaB.Pix[offset+1]) / 255.0,
				B: float64(rgbaB.Pix[offset+2]) / 255.0,
			}
			totalDifference += c1.DistanceCIEDE2000(c2)
		}
	}

	return totalDifference / pixelCount
}
