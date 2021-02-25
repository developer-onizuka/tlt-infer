# 1. docker pull
```
sudo docker pull nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3
```

# 2. docker run
```
$ sudo docker run --gpus all -it -v /mnt/docker/tlt-experiments:/workspace/tlt-experiments -p 8888:8888 nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3 /bin/bash
```

# 3. check your GPU by nvidia-smi
```
root@1594d9b196fc:/workspace# nvidia-smi
Sun Feb 21 12:30:45 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro P400         Off  | 00000000:01:00.0  On |                  N/A |
| 34%   44C    P8    N/A /  N/A |    228MiB /  1993MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

# 4. ngc's authentication
```
root@1594d9b196fc:/workspace# ngc config set
Enter API key [no-apikey]. Choices: [<VALID_APIKEY>, 'no-apikey']: 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Enter CLI output format type [ascii]. Choices: [ascii, csv, json]: 
Enter org [no-org]. Choices: ['xxxxxxxxxxxx']: xxxxxxxxxxxx
Enter team [no-team]. Choices: ['no-team']: no-team
Successfully saved NGC configuration to /root/.ngc/config
```

# 5. download peoplenet
```
root@1594d9b196fc:/workspace/tlt-experiments# ngc registry model list nvidia/tlt_*
+---------------+---------------+---------------+---------------+-----------+-----------+---------------+------------+
| Name          | Repository    | Latest        | Application   | Framework | Precision | Last Modified | Permission |
|               |               | Version       |               |           |           |               |            |
+---------------+---------------+---------------+---------------+-----------+-----------+---------------+------------+
| VehicleMakeNe | nvidia/tlt_ve | unpruned_v1.0 | Classificatio | Transfer  | FP32      | Sep 24, 2020  | unlocked   |
| t             | hiclemakenet  |               | n             | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| VehicleTypeNe | nvidia/tlt_ve | unpruned_v1.0 | Classificatio | Transfer  | FP32      | Sep 24, 2020  | unlocked   |
| t             | hicletypenet  |               | n             | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| DashCamNet    | nvidia/tlt_da | unpruned_v1.0 | Object        | Transfer  | FP32      | Sep 24, 2020  | unlocked   |
|               | shcamnet      |               | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| TrafficCamNet | nvidia/tlt_tr | unpruned_v1.0 | Object        | Transfer  | FP32      | Sep 24, 2020  | unlocked   |
|               | afficcamnet   |               | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| PeopleNet     | nvidia/tlt_pe | unpruned_v2.1 | Object        | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
|               | oplenet       |               | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| FaceDetectIR  | nvidia/tlt_fa | unpruned_v1.0 | Object        | Transfer  | FP32      | Sep 24, 2020  | unlocked   |
|               | cedetectir    |               | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| TLT           | nvidia/tlt_pr | vgg16         | Classificatio | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
| Pretrained Cl | etrained_clas |               | n             | Learning  |           |               |            |
| assification  | sification    |               |               | Toolkit   |           |               |            |
| TLT           | nvidia/tlt_pr | vgg16         | Object        | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
| Pretrained    | etrained_obje |               | Detection     | Learning  |           |               |            |
| Object        | ct_detection  |               |               | Toolkit   |           |               |            |
| Detection     |               |               |               |           |           |               |            |
| TLT           | nvidia/tlt_pr | resnet34      | Object        | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
| Pretrained    | etrained_dete |               | Detection     | Learning  |           |               |            |
| DetectNet V2  | ctnet_v2      |               |               | Toolkit   |           |               |            |
| TLT           | nvidia/tlt_in | resnet50      | Instance      | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
| Pretrained    | stance_segmen |               | Segmentation  | Learning  |           |               |            |
| Instance      | tation        |               |               | Toolkit   |           |               |            |
| Segmentation  |               |               |               |           |           |               |            |
| TLT           | nvidia/tlt_se | resnet101     | Semantic      | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
| Pretrained    | mantic_segmen |               | Segmentation  | Learning  |           |               |            |
| Semantic      | tation        |               |               | Toolkit   |           |               |            |
| Segmentation  |               |               |               |           |           |               |            |
| PeopleSegNet  | nvidia/tlt_pe | trainable_v1. | Instance      | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
|               | oplesegnet    | 0             | Segmentation  | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| HeartRateNet  | nvidia/tlt_he | trainable_v1. | HeartRateNet  | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
|               | artratenet    | 0             | Estimation    | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| FaceDetect    | nvidia/tlt_fa | trainable_v1. | Object        | Transfer  | FP32      | Feb 10, 2021  | unlocked   |
|               | cenet         | 0             | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| License Plate | nvidia/tlt_lp | unpruned_v1.0 | Object        | Transfer  | FP32      | Feb 10, 2021  | unlocked   |
| Detection     | dnet          |               | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| License Plate | nvidia/tlt_lp | trainable_v1. | Character     | Transfer  | FP32      | Feb 10, 2021  | unlocked   |
| Recognition   | rnet          | 0             | Recognition   | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| Facial        | nvidia/tlt_fp | trainable_v1. | Fiducial      | Transfer  | FP32      | Feb 10, 2021  | unlocked   |
| Landmarks     | enet          | 0             | Landmarks     | Learning  |           |               |            |
| Estimation    |               |               |               | Toolkit   |           |               |            |
| GestureNet    | nvidia/tlt_ge | trainable_v1. | Gesture Class | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
|               | sturenet      | 0             | ification     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| Gaze          | nvidia/tlt_ga | trainable_v1. | Gaze          | Transfer  | FP32      | Feb 10, 2021  | unlocked   |
| Estimation    | zenet         | 0             | Detection     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
| EmotionNet    | nvidia/tlt_em | trainable_v1. | Emotion Class | Transfer  | FP32      | Feb 09, 2021  | unlocked   |
|               | otionnet      | 0             | ification     | Learning  |           |               |            |
|               |               |               |               | Toolkit   |           |               |            |
+---------------+---------------+---------------+---------------+-----------+-----------+---------------+------------+
root@1594d9b196fc:/workspace/tlt-experiments# ngc registry model download-version nvidia/tlt_peoplenet:unpruned_v2.1 --dest ./model 
Downloaded 85.32 MB in 4m 29s, Download speed: 324.34 KB/s
----------------------------------------------------
Transfer id: tlt_peoplenet_vunpruned_v2.1 Download status: Completed.
Downloaded local path: /workspace/tlt-experiments/model/tlt_peoplenet_vunpruned_v2.1
Total files downloaded: 1 
Total downloaded size: 85.32 MB
Started at: 2021-02-22 06:48:01.030555
Completed at: 2021-02-22 06:52:30.399340
Duration taken: 4m 29s
----------------------------------------------------
```
# 6. preparing some picutures from google etc
```
root@e5b1a4fcdf06:/workspace# pwd
/workspace
root@e5b1a4fcdf06:/workspace# ls -l tlt-experiments/input/
total 1164
-rw-rw-r-- 1 1000 1000 234137 Feb 23 06:12 24176012-collage-of-many-different-human-faces.jpg
-rw-rw-r-- 1 1000 1000 112381 Feb 23 02:04 800px-Bon_odori_at_Hanazono_Shrine.jpg
-rw-r--r-- 1 root root 276650 Jan  2 04:29 9937f27d49d4adf2432969222cb12fc004b5f75e.jpeg
-rw-rw-r-- 1 1000 1000 557374 Feb 23 02:12 zombie1.jpg
```

# 7. preparing config file for detectnet_v2
```
root@e5b1a4fcdf06:/workspace/tlt-experiments# pwd
/workspace/tlt-experiments
root@e5b1a4fcdf06:/workspace/tlt-experiments# cat kitti/detectnet_v2_inference_kitti_tlt.txt
inferencer_config{
target_classes: "Person"
target_classes: "Bag"
target_classes: "Face"

image_width: 960
image_height: 544

image_channels: 3
batch_size: 16
gpu_index: 0

tlt_config{
model: "/workspace/tlt-experiments/model/tlt_peoplenet_vunpruned_v2.1/resnet34_peoplenet.tlt"
}
}
bbox_handler_config{
kitti_dump: true
disable_overlay: false
overlay_linewidth: 2
classwise_bbox_handler_config{
key:"Person"
value: {
confidence_model: "aggregate_cov"
output_map: "Person"
bbox_color{
R: 0
G: 255
B: 0
}
clustering_config{
coverage_threshold: 0.00
dbscan_eps: 0.3
dbscan_min_samples: 0.05
minimum_bounding_box_height: 4
}
}
}
classwise_bbox_handler_config{
key:"Bag"
value: {
confidence_model: "aggregate_cov"
output_map: "Bag"
bbox_color{
R: 0
G: 255
B: 255
}
clustering_config{
coverage_threshold: 0.00
dbscan_eps: 0.3
dbscan_min_samples: 0.05
minimum_bounding_box_height: 4
}
}
}
classwise_bbox_handler_config{
key:"Face"
value: {
confidence_model: "aggregate_cov"
output_map: "Face"
bbox_color{
R: 255
G: 0
B: 0
}
clustering_config{
dbscan_eps: 0.3
dbscan_min_samples: 0.05
minimum_bounding_box_height: 4
}
}
}
classwise_bbox_handler_config{
key:"default"
value: {
confidence_model: "aggregate_cov"
bbox_color{
R: 255
G: 0
B: 0
}
clustering_config{
coverage_threshold: 0.00
dbscan_eps: 0.3
dbscan_min_samples: 0.05
minimum_bounding_box_height: 4
}
}
}
}
```


# 7. inference
```
root@1594d9b196fc:/workspace/tlt-experiments# mkdir output
root@1594d9b196fc:/workspace/tlt-experiments# tlt-infer detectnet_v2 -e kitti/detectnet_v2_inference_kitti_tlt.txt -o output -i input/ -k tlt_encode

....
2021-02-25 05:26:25.197299: W tensorflow/core/common_runtime/bfc_allocator.cc:424] ________________________________________________________________________________*****_____**********
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:18<00:00, 78.73s/it]
2021-02-25 05:26:27,794 [INFO] iva.detectnet_v2.scripts.inference: Inference complete
```

# 8. check results
```
root@e5b1a4fcdf06:/workspace/tlt-experiments# ls -l output/images_annotated/
total 904
-rw-r--r-- 1 root root 255501 Feb 25 05:26 24176012-collage-of-many-different-human-faces.jpg
-rw-r--r-- 1 root root  96005 Feb 25 05:26 800px-Bon_odori_at_Hanazono_Shrine.jpg
-rw-r--r-- 1 root root 171206 Feb 25 05:26 9937f27d49d4adf2432969222cb12fc004b5f75e.jpeg
-rw-r--r-- 1 root root 397153 Feb 25 05:26 zombie1.jpg
root@e5b1a4fcdf06:/workspace/tlt-experiments# ls -l output/labels/
total 52
-rw-r--r-- 1 root root  2179 Feb 25 05:26 24176012-collage-of-many-different-human-faces.txt
-rw-r--r-- 1 root root 18016 Feb 25 05:26 800px-Bon_odori_at_Hanazono_Shrine.txt
-rw-r--r-- 1 root root 17805 Feb 25 05:26 9937f27d49d4adf2432969222cb12fc004b5f75e.txt
-rw-r--r-- 1 root root  4756 Feb 25 05:26 zombie1.txt
```

