# partial-labels


## Setup

### 1. Download repo

1. Go in some folder <repo_root>
2. `git clone git@github.com:lcalem/partial-labels.git .`
3. Put `<repo_root>/partial-labels` in your `PYTHONPATH` by putting this line in your `/.bashrc` or whatever file you're using:

```
export PYTHONPATH="${PYTHONPATH}:/<repo_root>/partial-labels"
```


### 2. Datasets

#### 2.1. PascalVOC

2.1.1. Download
<br/>Make a dataset folder and `cd` in it (it will be called <dataset_root>)
<br/>Download Pascal-VOC 2007 dataset
- trainval `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar`
- test `wget wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar`

2.1.2. Untar everything in place and all annotations / images should come in place nicely
- `tar xvf VOCtrainval_06-Nov-2007.tar`
- `tar xvf VOCtest_06-Nov-2007.tar`

2.1.3. Preprocess using `data/pascalvoc/preprocessing/pp_multilabel.py` to create a single annotation csv
(do it once for trainval and a second time for test, separately)

- `python3 pp_multilabel.py <dataset_root> trainval`
- `python3 pp_multilabel.py <dataset_root> test`


#### 2.2. MS COCO

Coming soon<sup>TM</sup>