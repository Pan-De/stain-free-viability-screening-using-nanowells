# stain-free-viability-screening-using-nanowells

This document contains instructions on how to access the data associated with the submitted manuscript "".
Cells were imaged in [nanowell-in-microwell plates] (https://www.imagecyte.bio/). 
The datasets for AI model development are available:
cell Type: MDA-MB-231 cell line.
Treatments: ethanol, andrographolide, daunorubicin, and serum starvation (5% FBS).
| Single cells      | link |
| non-single cells  | link |
| live single cells | link |
| dead single cells | link |


Step1: Installation requirements:
Linux or macOS with Python ≥ 3.6
TensorFlow ≥ 3.7
scikit-learn
matplotlib
OpenCV
NumPy
pandas
Keras

Step2: Download trained model
model for single cell identificaiton: [1st Model](https://ubcca-my.sharepoint.com/:u:/r/personal/pandeng_student_ubc_ca/Documents/Project%20viability%20screening/trained%20model/1st%20CNN.h5?csf=1&web=1&e=fjIaV2)
model for live and dead assessment: [2nd Model] (https://ubcca-my.sharepoint.com/:u:/r/personal/pandeng_student_ubc_ca/Documents/Project%20viability%20screening/trained%20model/2nd%20CNN.h5?csf=1&web=1&e=wftX5p)

Step3: Cell viability analysis
Run apply_mode.py
The parameters like thresholds and nanowell dimensions can be adjusted to suit different needs.
