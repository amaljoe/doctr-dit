# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=dit/object_detection:$PYTHONPATH
python3 dit/object_detection/inference.py \
--image_path data/mydata/reg.png \
--output_file_name outputs/mydata/reg.jpg \
--config-file dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_large.yaml \
--opts MODEL.WEIGHTS models/publaynet_dit-l_mrcnn.pth