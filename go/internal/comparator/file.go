package comparator

import (
	"fmt"
	"image"
	"image/draw"
	"golang.org/x/image/bmp"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// loadImage opens and decodes an image from the given file path.
// It converts the image to RGBA format for consistent processing.
func loadImage(path string, imageType string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open file: %w", err)
	}
	defer file.Close()

	var decodedImg image.Image
	if imageType != "" {
		switch strings.ToLower(imageType) {
		case "jpeg", "jpg":
			decodedImg, err = jpeg.Decode(file)
		case "png":
			decodedImg, err = png.Decode(file)
		case "bmp":
			decodedImg, err = bmp.Decode(file)
		default:
			return nil, fmt.Errorf("unsupported image type specified: %s", imageType)
		}
	} else {
		decodedImg, _, err = image.Decode(file)
	}

	if err != nil {
		return nil, fmt.Errorf("could not decode image: %w", err)
	}

	if rgba, ok := decodedImg.(*image.RGBA); ok {
		return rgba, nil
	}

	// Convert to RGBA for consistent and potentially faster processing.
	bounds := decodedImg.Bounds()
	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, rgbaImg.Bounds(), decodedImg, bounds.Min, draw.Src)

	return rgbaImg, nil
}

// saveDifferentPairs reads from the results channel and saves the image pairs
// to the specified output directory.
func saveDifferentPairs(results <-chan unitPair, outputDir string) {
	for pair := range results {
		dirName := fmt.Sprintf("unit_%d_%d_vs_unit_%d_%d",
			pair.UnitA.ID.X, pair.UnitA.ID.Y,
			pair.UnitB.ID.X, pair.UnitB.ID.Y,
		)
		pairDir := filepath.Join(outputDir, dirName)

		if err := os.MkdirAll(pairDir, 0755); err != nil {
			log.Printf("Error creating directory %s: %v", pairDir, err)
			continue
		}

		saveUnit(pair.UnitA, pairDir)
		saveUnit(pair.UnitB, pairDir)
		log.Printf("Saved differing pair to %s", dirName)
	}
}

// saveUnit saves a single unit's image to a PNG file.
func saveUnit(u *unit, dir string) {
	fileName := fmt.Sprintf("unit_%d_%d.png", u.ID.X, u.ID.Y)
	filePath := filepath.Join(dir, fileName)

	outFile, err := os.Create(filePath)
	if err != nil {
		log.Printf("Error creating file %s: %v", filePath, err)
		return
	}
	defer outFile.Close()

	if err := png.Encode(outFile, u.Img); err != nil {
		log.Printf("Error encoding png %s: %v", filePath, err)
	}
}
