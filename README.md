# DisentangleSum model of "Disentangling Specificity for Abstractive Multi-document Summarization"

### Project Description

Official Code for DisentangleSum model "Disentangling Specificity for Abstractive Multi-document Summarization"

### Environments

To run the code, please install the following packages 
```text
chardet==3.0.4
future==0.17.1
idna==2.8
numpy==1.19.5
opencv-python==4.6.0.66
requests==2.21.0
six==1.12.0
tqdm==4.31.1
urllib3==1.24.1
torchtext==0.3.1
#transformer
scikit-learn
matplotlib
transformers
seaborn
torch==1.7.1
torchvision==0.8.2
```

### Preprocess
To get the training and validation data in a correct format, a preprocess is required. The raw data should be in text files with one record per line. You need to modify the path to text files in the `run_preprocess.sh` before running the script.
```sh
$ ./run_preprocess.sh
```

### Data
For the datasets have been used, please refer to the paper.

### Training
Run
```sh
$ ./train.sh
```
Related hyper-parameters can be tuned in the shell script.


### Testing
Run
```sh
$ ./test.sh
```
Related hyper-parameters can be tuned in the shell script.


### Acknowledgement

The code is built on OpenNMT, we thank the author of OpenNMT project.
