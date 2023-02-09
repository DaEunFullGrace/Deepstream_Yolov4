Basic usage of deepstream_yolov4 and yolov7 is same\
This repository supports YOLOv4-serise\
If you want to use YOLOv7 in this respo., modify model name in the nvdsinfer_custom_impl_Yolov4/nvdsinfer_yolo_engine.cpp (but keep yolotype=yolov4)

## Note
In the deepstream_yolov7, mask were set up to use 3-FPN layer with 4 anchors
If you want to use default setting from YOLOv7 paper, change the variable "kMASKS" from nvdsparsebbox_Yolo.cpp

## How to execute 
1. cd cfg
2. Download weights, cfg files in /cfg directory
3. cd ..
4. deepstream app -c deepstream_app_config_{model_name}.txt


## How to customize your networks
1. cd nvdsinfer_custom_impl_Yolov4
2. vi nvdsparsebbox_Yolo.cpp, and change kANCHORS, NUM_CLASSES_YOLO 
3. (optinal) vi nvdsinfer_yolo_engine.cpp, and change networkInfo.networkSize
(If variable "size" is declared, than change the variable "size" to use different input resolution)
(Change the value according to the cfg file)
4. make


## How to change deepstream config file
### From config_infer_primary_{model_name}.txt
* If you want to use differnt cfg file or weight, than change "custom-network-config" and "model-file"
* To use quantization, change "network-mode"
### From deepstream_app_config_{model_name}.txt
* To change input video, change "uri"
