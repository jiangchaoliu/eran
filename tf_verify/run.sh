python3 . --netname ./nets/rcnn6t16_1.h5 --domain deepzono   --dataset mnist --startnum 0 --endnum 100 --category rcnn6t16_1_valid
python3 . --netname ./nets/rcnn6t16_1.h5  --domain deepzono   --dataset mnist --startnum 0 --endnum 100 --category rcnn6t16_1_fgsm1
python3 . --netname ./nets/rcnn6t16_1.h5  --domain deepzono   --dataset mnist --startnum 0 --endnum 100 --category rcnn6t16_1_fgsm05
python3 . --netname ./nets/rcnn6t16_1.h5  --domain deepzono   --dataset mnist --startnum 0 --endnum 100 --category rcnn6t16_1_wrong
python3 . --netname ./nets/rcnn6t16_1.h5  --domain deepzono   --dataset mnist --startnum 0 --endnum 100 --category rcnn6t16_1_cw
python3 . --netname ./nets/rcnn6t16_1.h5  --domain deepzono   --dataset mnist --startnum 0 --endnum 100 --category rcnn6t16_1_hop
