# raster-mask-to-cocojson

This code is a boilerplate for segmentation mask extraction software written in a high-performance C++17 to extract polygonal ground-truth masks from the [SemSeg Outdoor Pano dataset (all‑rgb‑masks)](https://github.com/semihorhan/semseg-outdoor-pano/tree/main) and convert them into COCO-style JSON annotations.
It is especially useful for preparing training datasets for semantic segmentation AI models, converting color-based mask images into structured annotation formats, or uploading annotation data to platforms like Deep Block, which supports COCO-format inputs.
Simply compile the code and run the resulting binary to accelerate your annotation pipeline with minimal overhead.

## 🚀 Overview

- **Data Source**: Mask images are taken from the "all-rgb-masks" folder of the SemSeg Outdoor Pano dataset. The tool parses each colored mask, extracts contours per class, and outputs a single COCO-format JSON file for downstream ML pipelines.
- **Performance**: Written in C++17 with `std::thread` parallelism, it processes 30,000+ mask images in seconds on modern multi-core Linux—far faster than Python-based scripts.
- **Cross-Platform**: We tested our code on Linux with `g++` compiler, using OpenCV4 for image I/O and RapidJSON for JSON serialization.

## ⚙️ Requirements

- **Compiler**: `g++` (C++17)
- **Libraries**:
  - OpenCV4
  - RapidJSON

## 📦 Installation & Compilation

```bash
# Build the tool
g++ -std=c++17 -O3 -pthread \
    -o mask_to_coco mask_to_coco.cpp \
    `pkg-config --cflags --libs opencv4`
```


## 🔧 Customization

You can customize the following parameters as needed:

- **Directory Path**: Change the `maskFolder` constant at the top of `main.cpp`.
- **Thread count**: Change the `NUM_THREADS` constant at the top of `main.cpp` to fit your hardware and workload.
- **Image resolution**: Adjust the `width` and `height` values in `writeCocoJson()` to suit your dataset’s dimensions.
- **Class mapping**: Modify the `colorToLabel` map in `main.cpp` if your mask uses a different color palette or class order.


## 🏢 About DeepBlock

[`DEEP BLOCK`](https://deepblock.net) empowers researchers and engineers with fast, scalable tooling for computer vision and AI. This open-source converter is released under the MIT License; contributions and forks are welcome.
