python3 dit/object_detection/inference.py \
--image_path data/mydata/ss2.png \
--output_file_name outputs/mydata/ss2_layout_analysis.jpg \
--config dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_large.yaml \
--opts MODEL.WEIGHTS models/publaynet_dit-l_mrcnn.pth \