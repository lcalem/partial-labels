# partial-labels


## Setup

### Datasets

1. Download
Make a dataset folder and `cd` in it (it will be called <dataset_root>)
Download Pascal-VOC 2007 dataset
- trainval `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar`
- test `wget wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar`

2. Untar everything in place and all annotations / images should come in place nicely
- `tar xvf VOCtrainval_06-Nov-2007.tar`
- `tar xvf VOCtest_06-Nov-2007.tar`

3. Preprocess using `data/preprocessing/pp_multilabel.py` to create a single annotation csv
(do it once for trainval and a second time for test, separately)

- `python3 pp_multilabel.py <dataset_root> trainval`
- `python3 pp_multilabel.py <dataset_root> test`

