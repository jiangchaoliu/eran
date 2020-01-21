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
 


