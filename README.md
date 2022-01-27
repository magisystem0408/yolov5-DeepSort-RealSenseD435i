# yolov5+Realsence+DeepSense D435i

## Create Environment
```shell
# create conda environment
conda crate -n <Any name> python=3.8
conda activate <Any name>

# change directory
cd personalize_hand

# clone yolov5
git clone https://github.com/ultralytics/yolov5/tree/aa1859909c96d5e1fc839b2746b45038ee8465c9

# install requirements
pip install -r requirements.txt

# change yolov5 directory
cd yolov5

$ install requirements
pip install -r requirements.txt
```


## Module used
### yolov5
> https://github.com/ultralytics/yolov5

### Yolov5 + Deep Sort with Pytorch
> https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

### DeepSense D435i
> https://www.intelrealsense.com/depth-camera-d435i/

## program introduction
### realsence_track.py
We have introduced yolov5 and deepsort from the information acquired by deepsense.

The following parameters are output from the console
```shell
frame_idx, id, c, names[c], bbox_left, bbox_top, bbox_w, bbox_h, center_x, center_y,depth
```
> Image displayed
![realsence](https://user-images.githubusercontent.com/61937077/151387782-fc5056f3-0dac-4fe5-afc4-07e6146d045d.png)
![depth](https://user-images.githubusercontent.com/61937077/151387943-a14d4be1-f3cc-4815-bfa0-0a84cbf324c1.png)

> Console Image
![スクリーンショット 2022-01-28 002628](https://user-images.githubusercontent.com/61937077/151389521-550f1f2f-a187-4599-a617-0f8db8cfbad8.png)

#### Execution method
```python
python realsence_track.py
```


### realsence_track_person.py
Only detect people and output to the console

The following parameters are output from the console
```shell
frame_idx, id, c, 'person', bbox_left, bbox_top, bbox_w, bbox_h, center_x, center_y,depth
```

![スクリーンショット 2022-01-28 005053](https://user-images.githubusercontent.com/61937077/151394068-1c2c3dda-d1cb-401b-bb3a-de3f1473071d.png)


#### Execution method
```python
python realsence_track_person.py
```
