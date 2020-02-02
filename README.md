This is a modified version of ERAN, the information of which (including how to install it)
is in ORG_README.md

We have extended ERAN to compute robustness radius.



Usage
-------------

```
cd tf_verify

python3 . --netname <path to the network file> --epsilon <float between 0 and 1> --domain <deepzono/deeppoly/refinezono/refinepoly> --category <path to images file> --startnum <int starting index> --endnum <int end index> --dataset <mnist/cifar10>   [optional] --complete <True/False> 
```


* ```<netname>```: the path of the network, we put all networks in the paper at ./tf_verify/nets
* ```<epsilon>```: maxmial possible robustness radius
* ```<domain>```: the abstract domain used for verification
* ```<complete>```: True/False for complete/incomplete verification, default is False
* ```<dataset>```: Mnist/Cifar10 for different datasets
* ```<category>```: the name of the file with input images, the file must be stored in ./tr_verify/data
* ```<startnum/endnum>```: since the file with input images are with many images, we support only computing a part of the images, the indexes of which are between startnum and endnum 

* Refinezono and RefinePoly refines the analysis results from the DeepZ and DeepPoly domain respectively. The optional parameters timeout_lp and timeout_milp (default is 1 sec for both) specify the timeouts for the LP and MILP forumlations of the network respectively. 

* Since Refinezono and RefinePoly uses timeout for the gurobi solver, the results will vary depending on the processor speeds. 


Example
-------------

```
python3 . --netname ./nets/rcnn6t16_1.h5 --domain deepzono   --dataset mnist --start 0 --endnum 100 --category rcnn6t16_1_valid
```

It will compute the robustness radius of the first 100 images in ./data/rcnn6t16_1_valid.csv on the network ./nets/rcnn6t16_1.h5 with domain deepzono. The result will be sotred in "bound_rcnn6t16_1_valid.txt", which contains two parts:

1. index
2. robustness radius


Compute Average Robustness Radius

We provide the code to compute the average radius of several inputs. Its usage is quite similar to ``__main__.py''. Here is an example 

```
python3 average.py --netname ./nets/rcnn6t16_1.h5 --domain deepzono   --dataset mnist --start 0 --endnum 100 --category rcnn6t16_1_valid
```

This command will give the average robustness radius of the first 100 images in ./data/rcnn6t16_1_valid.csv on the network ./nets/rcnn6t16_1.h5 with domain deepzono.





