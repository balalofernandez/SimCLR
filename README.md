# Instructions

The following are instructions to run our code. Note our tutorial is focused on linux machines that run pip virtual environments.

## 1 Installing required packages

To install required packages run the following command:

```pip install -r requirements.txt```

Otherwise these three packages on top of the CW1-PT environment of COMP0197 will be fine. We had extra packages on requirements however only these three are needed.

1. matplotlib 
2. scikit-learn 
3. pyav

## 2 Downloading Datasets
We pre-train our model using iNat we recommend downloading this dataset. Use the following command from torch to first download the dataset. Set root to whichever folder you wish but we recommend `root = ”./datasets/i- Nat kingdom/2021 train mini”`. We can call the following command

```python
iNatdata = datasets.INaturalist(root=root, version=’2021_train_mini’,
    target_type = ’kingdom’)
```

We then transform the above dataset as follows:
```python
python image_resizer_imagent.py --in_dir root --out_dir out --size 64
```
Where `root = ”./datasets/iNat kingdom/2021 train mini”` and `out = ”./dataset- s/iNat64”`.

Please change the following lines in SimCLR Scratch Dev.py starting at line 109:

```python
inat_dataset_train = MixedINat128(
’/cs/student/projects3/COMP0087/grp1/simclr_pytorch_new/datasets/iNat64/lanczos’,
           transform=transform_augmented,
       )
```

to the following:

```python
inat_dataset_train = MixedINat128(
’./datasets/iNat64’,
           transform=transform_augmented,
       )
```

## 3 Saving and Loading models
Throughout our code we have saved our models in custom paths in the UCL shared CS folders however we recommend changing them so that the models get saved and loaded into your folders.
We recommend modifying the `save path` variable throughout the code to your desired directories.
Similarly when loading the model we call the variable checkpoint path please change this to the `save path` you saved the model in.


## 4 Running all our code
After modifying the variables in which our data and models are saved and stored everything can be run with `runner.py`. Type `python runner.py`