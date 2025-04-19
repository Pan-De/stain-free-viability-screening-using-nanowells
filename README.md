# stain-free-viability-screening-using-nanowells
[![DOI](https://zenodo.org/badge/960107108.svg)](https://doi.org/10.5281/zenodo.15243856)

This document contains instructions on how to access the data associated with the submitted manuscript **"Regularized single-cell imaging enables generalizable AI models for stain-free cell viability screening"**.


**Datasets for AI model development** <br>
Cell Type: MDA-MB-231 cell line.<br>
Cells were imaged in [nanowell-in-microwell plates](https://www.imagecyte.bio/). <br>
Cell Imaging: Brightfield (phase-contrast) + LIVE/DEAD Cell Staining (Used as ground truth to generate text labels)<br>
Treatments: ethanol, andrographolide, daunorubicin, and serum starvation (5% FBS).<br>
| Class             | Download Link|File size|
|-------------------|--------------|---------|
| single cells      | [link](https://drive.google.com/file/d/1hjE3h5lt3Ub4w-1WMR6eiDITXW5Gz6U1/view?usp=sharing) |~850 MB|
| non-single cells  | [link](https://drive.google.com/file/d/1Sph_qZ8ELw5VxT3KKRVlaJHL9NmhYhvF/view?usp=sharing) |~850 MB|
| live single cells | [link](https://drive.google.com/file/d/1wg0-1F6XGVBxklr2-0WR9m3AHO989qtq/view?usp=sharing) |~550 MB|
| dead single cells | [link](https://drive.google.com/file/d/19F9KXQjNrG1D4ZAQyg6w4EFOeHNxrSCo/view?usp=sharing) |~520 MB|
* single cell: 1 cell/nanowell <br>
* non-single cell: 0 or 2+ cells/nanowell <br>
* live single cell: A cell that is positive for the live stain, has a size no smaller than 30% of the average size of cells in culture medium, and shows no visible blebs. <br>
* dead single cell: A cell that is positive for the dead stain, or a live stain-positive cell that is smaller than 30% of the average size of cells in culture medium or has visible blebs. <br>

**Source data producing the graphs in the manuscript** <br>
All images producing the graphs are available following this link (requires ~35 GB). 

**Step1: Installation requirements:** <br>
* Linux or macOS with Python ≥ 3.6
* TensorFlow ≥ 3.7
* scikit-learn
* matplotlib
* OpenCV
* NumPy
* pandas
* Keras



**Step2: Download trained model** <br>
model for single cell identification: [single cell model](https://drive.google.com/file/d/1E49LOYc56UYo5xKoDYxZbbY9ZKN1VzQV/view?usp=sharing)<br>
model for live and dead assessment: [live/dead model](https://drive.google.com/file/d/19yrt8uCJc25KblAyd7UQ6oIn2IU1cKmK/view?usp=sharing)

**Step3: Cell viability analysis** <br>
Run apply_mode.py <br>
The parameters like thresholds and nanowell dimensions can be adjusted to suit different needs.
