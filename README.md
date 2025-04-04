# stain-free-viability-screening-using-nanowells

This document contains instructions on how to access the data associated with the submitted manuscript "".


**Datasets for AI model development** <br>
Cell Type: MDA-MB-231 cell line.<br>
Cells were imaged in [nanowell-in-microwell plates](https://www.imagecyte.bio/). <br>
Cell Imaging: Brightfield (phase-contrast) + LIVE/DEAD Cell Staining (Used as ground truth to generate text labels)<br>
Treatments: ethanol, andrographolide, daunorubicin, and serum starvation (5% FBS).<br>
| Class             | Download Link|
|-------------------|--------------|
| Single cells      | link |
| non-single cells  | link |
| live single cells | link |
| dead single cells | link |
* single cell: 1 cell/nanowell <br>
* non-single cell: 0 or 2+ cells/nanowell <br>
* live single cell: A cell that is positive for the live stain, has a size no smaller than 30% of the average cell size, and shows no visible blebs. <br>
* dead single cell: A cell that is positive for the dead stain, or a live stain-positive cell that is smaller than 30% of the average cell size or has visible blebs. <br>


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
model for single cell identification: [single cell model](https://ubcca-my.sharepoint.com/:u:/r/personal/pandeng_student_ubc_ca/Documents/Project%20viability%20screening/trained%20model/1st%20CNN.h5?csf=1&web=1&e=fjIaV2)<br>
model for live and dead assessment: [live/dead model](https://ubcca-my.sharepoint.com/:u:/r/personal/pandeng_student_ubc_ca/Documents/Project%20viability%20screening/trained%20model/2nd%20CNN.h5?csf=1&web=1&e=wftX5p)

**Step3: Cell viability analysis** <br>
Run apply_mode.py <br>
The parameters like thresholds and nanowell dimensions can be adjusted to suit different needs.
