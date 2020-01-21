This is a forked version of ERAN, the information of which is in org_README.d

In this repository, we extended ERAN to compute robustness radius.



Usage
-------------

```
cd tf_verify

python3 . --netname <path to the network file> --epsilon <float between 0 and 1> --domain <deepzono/deeppoly/refinezono/refinepoly> --category <path to images file> --startnum <int starting index> --endnum <int end index> --dataset <mnist/cifar10/>   [optional] --complete <True/False> --timeout_lp <float> --timeout_milp <float> 
```


* ```<netname>```: the path of the network, we put all networks in the paper at ./tf_verify/nets
* ```<epsilon>```: maxmial possible robustness radius
* ```<domain>```: the abstract domain used for verification
* ```<complete>```: True/False for complete/incomplete verification, default is False
* ```<category>```: the name of the file with input images, the file must be stored in ./tr_verify/data
* ```<startnum/endnum>```: since the file with input images are with many images, we support only computing a part of the images, the indexes of which are between startnum and endnum 

* Refinezono and RefinePoly refines the analysis results from the DeepZ and DeepPoly domain respectively. The optional parameters timeout_lp and timeout_milp (default is 1 sec for both) specify the timeouts for the LP and MILP forumlations of the network respectively. 

* Since Refinezono and RefinePoly uses timeout for the gurobi solver, the results will vary depending on the processor speeds. 


Example
-------------

L_oo Specification
```
python3 . --netname ./nets/acnn.h5 --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_valid
```

It will compute the robustness radius of the first 100 images in ./data/mnistcnn_valid.csv on the network ./nets/acnn.h5 with domain deepzono. The result will be sotred in "bound_mnistcnn_valid", which contains three parts:

1. index
2. robustness radius
3. average time of each call to "isrobust"


More Experiments
-----------------
We also provide code in ./generate_images_cifar and ./generate_images_mnist to have more networks and images, to run them, please install [IBM/adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox.git) first. The code is straightforward, here
are some examples.

In generate_images_mnist
```
python3 train_fnn.py 3 100 relu
```
It will train a FNN network with 3 hiden layers each with 100 neurons.

```
python3 train_cnn.py 
```
It will train a CNN network on Mnist

```
python3 fgsm1.py acnn.h5 fgsm1.csv
```
It will generate at most 100 adversarial examples from successful FGSM (epsilon = 0.1) attacks on acnn.h5, saved in fgsm1.csv

```
python3 valid_wrong.py acnn.h5 valid.csv wrong.csv
```
It will generate at most 100 valid and wrong images on acnn.h5 saved in valid.csv and wrong.csv respectively








