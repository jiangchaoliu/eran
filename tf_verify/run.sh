python3 . --netname ./nets/acnn.h5 --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_valid
python3 . --netname ./nets/acnn.h5  --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_fgsm1
python3 . --netname ./nets/acnn.h5  --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_fgsm05
python3 . --netname ./nets/acnn.h5  --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_wrong
python3 . --netname ./nets/acnn.h5  --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_cw
python3 . --netname ./nets/acnn.h5  --domain deepzono   --dataset mnist --skipnum 0 --endnum 100 --category mnistcnn_hop
