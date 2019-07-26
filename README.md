# partial-labels


## Presentation

This is the working repository for a research project about partial labels. Authors are [Laura Calem](https://lcalem.github.io/) and [Olivier Petit](https://www.olivier-petit.fr/)


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

- `python3 pp_multilabel.py <dataset_root> train`
- `python3 pp_multilabel.py <dataset_root> val`

2.1.4. Use `data/pascalvoc/preprocessing/partial_datasets.py` to create partial datasets:

`python3 partial_datasets.py <dataset_root>/Annotations`



#### 2.2. MS COCO

2.2.1. Download
<br/>Make a dataset folder and `cd` in it (it will be called <dataset_root>)
<br/>Download MSCOCO 2014 dataset

- train images `wget http://images.cocodataset.org/zips/train2014.zip`
- val images `wget http://images.cocodataset.org/zips/val2014.zip`
- train+val annotations `wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip`

2.2.2. Unzip everything in place

2.2.3. Preprocess using `data/coco/preprocessing/pp_multilabel.py` to create a single annotation csv (do it once for train and once for val separately):
- `python3 pp_multilabel.py <dataset_root> train2014`
- `python3 pp_multilabel.py <dataset_root> val2014`

This operation will create csv complete datasets `<dataset_root>/annotations/multilabel_train2014.csv` and `<dataset_root>/annotations/multilabel_val2014.csv`.

2.2.4. Use `data/coco/preprocessing/partial_datasets.py` to create partial datasets (one with 10% known labels, one with 20% known labels, and so on til 100% known labels which should be identical to `multilabel_train2014.csv`):

`python3 partial_datasets.py <dataset_root>/annotations/multilabel_train2014.csv`

