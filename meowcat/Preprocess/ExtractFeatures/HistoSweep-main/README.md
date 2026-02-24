<p align="left">
  <img src="assets/HistoSweepLogo.png" width="400"/>
</p>


# HistoSweep: Automated H&E Image Quality Filtering

HistoSweep is an unsupervised, fast, computationally efficient tool designed to perform a **full quality sweep filtering procedure** for histology images, identifying **high-quality superpixels** for downstream spatial transcriptomics and image analysis.


---

## 📂 How to Use HistoSweep

HistoSweep is designed to be **easy to run** with no GPU requirements:

1. **Download** the `HistoSweep/` folder from the HistoSweep GitHub repository.  
   (It contains all `.py` scripts and the Jupyter Notebook)

2. **Place** your H&E image in your working directory (e.g. HE/demo):
   - If **unscaled/unprocessed**, name it: `he-raw.jpg` (.tiff, .png, .jpg are all compatible)  
   - If **scaled/unprocessed**, name it: `he-scaled.jpg` (.tiff, .png, .jpg are all compatible)  
   - If **already scaled and preprocessed**, name it: `he.jpg` (.tiff, .png, .jpg are all compatible)  

3. **Open** the Jupyter Notebook: `Run_HistoSweep.ipynb`

4. **Set** the input parameters (see below).

5. **Run** the notebook — it automatically calls all necessary scripts.

---

## ⚙️ Input Parameters

### USER-DEFINED PARAMETERS

```python
# Path prefix to your H&E image folder (This folder should be placed inside the HistoSweep main directory)
HE_prefix = 'HE/demo/'

# Flag for whether to rescale the image
need_scaling_flag = False  # True if image resolution ≠ 0.5µm (or desired size) per pixel

# Flag for whether to preprocess the image
need_preprocessing_flag = False  # True if image dimensions are not divisible by patch_size (i.e. image needs padding)

# The pixel size (in microns) of the raw H&E image
pixel_size_raw = 0.5  # Typically provided by scanner metadata (e.g., ~0.25 µm/pixel for 40x)

# Parameter used to determine the amount of density filtering (e.g., artifact removal)
density_thresh = 100  # Typically 100 works well; may need adjustment based on artifacts

# Flag for whether to clean background (i.e., remove isolated debris and small specks outside tissue)
clean_background_flag = True  # Set False if you want to preserve all fibrous regions (e.g., adipose)

# Minimum object size for debris removal
min_size = 10  # Lower for helping retain fibrous regions (e.g., 5), higher for large debris (e.g., 50)
```

### Additional Parameters (Typically Do Not Need to Change)

```python
# Size of one square patch (superpixel) used throughout processing
patch_size = 16  # 16x16 pixels (typically ~8µm if pixel_size = 0.5)

# Target pixel size (in microns)
pixel_size = 0.5  # Final desired resolution; keep as 0.5µm for standardization

# Directory for output 
output_directory = "HistoSweep_Output" #Folder for HistoSweep output/results
```

---

## 🔄 How It Works

1. **Optional Scaling & Preprocessing**: Adjust image pixel size to 0.5µm and pad image dimensions if needed.

2. **Superpixelization**: Image is divided into 16x16 pixel patches.

3. **Density-Based Filtering**: Removes low-density artifacts such as tissue folds, debris, and background noise.

4. **Texture-Based Filtering**: Computes GLCM texture features (energy, entropy, homogeneity) and filters poor-quality patches.

5. **Ratio-Based Filtering**: Uses otsu's thresholding on the ratio of standard deviation to weighted mean RGB.

6. **Final Quality Sweep**: Combines all masks to retain only **high-confidence tissue regions**.

7. **Output**: High-quality superpixel mask (`mask.png` and `mask-small.png`), visualizations, and QC plots.

---

## 📁 Output Files
- `mask.png`: Final pixel-level tissue mask.
- `mask-small.png`: Superpixel-level tissue mask.
- `conserve_index_mask.pickle`: Pixel-level mask saved as pickle for easy downstream use.
- `conserve_index_mask.pickle`: Superpixel-level mask saved as pickle for easy downstream use.
- 'AdditionalPlots' : Folder containing additional plots intended to provide further insights into the filtering process.

---

## ✨ Why Use HistoSweep?

- Fully unsupervised: No training data required.
- Fast and lightweight.
- Highly flexible: Works across diverse staining qualities, tissue types, and disease states.
- Improves downstream computational analyses by removing background, debris, fibrous tissue, and artifacts!

---

## Running the Demo

1. Download `HistoSweep/`.
2. Open `Run_HistoSweep.ipynb`.
3. Press **Run All**!

## Example Command Flow for New H&E Image

1. Download `HistoSweep/`.
2. Create new project directory in **HE** folder.
3. Place `he-raw.jpg` (.tiff, .png, .jpg are all compatible) in your project folder.
4. Open and configure `Run_HistoSweep.ipynb`.
5. Press **Run All**!
6. Enjoy your clean, high-quality masks and plots.

---


