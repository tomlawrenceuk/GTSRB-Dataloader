# GTSRB-Dataloader

Loads the official GTSRB (German Traffic Sign Recognition) training and test sets found here:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Tested with PyTorch 1.0

Modifications I made to the original dataset:

- Training Set - Created one CSV file containing all image paths and classes
- Test Set - Created one CSV file containing all image paths and classes

# Setup
 
- Download GTSRB.zip from OneDrive https://1drv.ms/u/s!An8jrZtDgrMljdt7o2khe7TGmZWbUg
- Unzip
- Use gtsrb_dataset.py to load dataset (see example.py)
