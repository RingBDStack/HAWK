# HAWK
Source code for paper: HAWK: a Rapid Android Malware Detectionthrough Heterogeneous Graph Attention Networks. 

We present HAWK, a rapid Android malware detectionframework that inductively learns and detects new Androidapplications in an incremental manner.  HAWK can promptlyidentify  previously unknown malware at millisecond-level andhas the highest precision compared against all baselines.

# Requirements
python 3.6

tensorflow-gpu v1.12.0 

scikit-learn  v0.21.3

# Datasets
Our test data includes benign app from Googlestore (https://play.google.com) and malicious Apps from VirusShare (https://virusshare.com),  CICAndMal (https://www.unb.ca/cic/datasets).


# Data preprocess
First, use the smalibat.py in the directory tool to decompile Apks.
Then, exract the entities from the decompiled apks following the lists in the directory feature_list and contruct these information into matrices.

# Results 
**Detect in-sample Apps**
Model|F1|Acc
:---:|:---:|:---:
HAWK|98.78%|98.78%

**Detect out-of-sample Apps**
DataSet|F1
:---:|:---:
v2013|92.84%
v2014|98.04%
v2015|97.36%
v2016|96.87%
v2017|96.95%
v2018|98.65%
v2019|98.58%
c2017|95.61%
c2019|94.93%

# Statement
Due to the limitation of storage, we only upload part of the processed data temporarily.
The source code is based on HAN、
