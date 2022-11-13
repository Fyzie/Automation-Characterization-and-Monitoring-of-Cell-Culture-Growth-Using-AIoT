# Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT
> Project Continuation from [This Repo](https://github.com/Fyzie/Autonomous-Cell-Counting-and-Handling)   
#### Table of Contents
- [Version 0](https://github.com/Fyzie/Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT#version-0---cell-segmentation--firebase-realtime-database) (Cell Segmentation + Firebase Realtime Database)
- [Version 1](https://github.com/Fyzie/Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT#version-1---cell-segmentation--gui) (Cell Segmentation + GUI)
- [Version 2](https://github.com/Fyzie/Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT#version-2---deployment-on-raspberry-pi-4--mobile-app) (Deployment on Raspberry Pi + Mobile App)
- [Sensors](https://github.com/Fyzie/Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT#sensors---donutboard-circuit--3d-printed-casing) (Circuit Design)
- [Demo Video](https://github.com/Fyzie/Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT#demo-video)\
## Version 0 - Cell Segmentation + Firebase Realtime Database
> Refer [here](https://github.com/Fyzie/Cell-segmentation-using-U-Net-based-networks) for model training using Tensorflow   

#### How to convert tensorflow model (eg. hdf5) to pb format:
```
import tensorflow as tf
import keras

model = tf.keras.models.load_model('model.hdf5')
tf.saved_model.save(model,'model.pb')
```
- Download 2021 OpenVINO developer package [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/)  
> For Windows: w_openvino_toolkit_dev_p_2021.4.582.zip

#### How to convert pb format to ir format:
- using Windows command prompt
```
cd ...\openvino_2021\bin
setupvars.bat
cd ...\openvino_2021\deployment_tools\model_optimizer
python mo.py --saved_model_dir <model_pb_folder> --output_dir IR --input_shape [1,<input_shape>,<input_shape>,1]
```
> Three files created:   
(1) saved_model.bin   
(2) saved_model.mapping   
(3) saved_model.xml   
#### How to run OpenVINO program:
- using Windows command prompt/ Anaconda prompt   
- make sure to initialize OpenVINO environment before running program
```
conda activate <anaconda_env> # if using Anaconda
cd ...\openvino_2021\bin
setupvars.bat
cd ...\Version 1\OpenVINO
python openvino_segmentation.py
```
## Version 1 - Cell Segmentation + GUI
<details open>
<summary>Local GUI</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200532356-0d42cbda-1155-4b6e-af7b-5ab82a9d6e45.png" width="500">
</details>
<details open>
  
<summary>Trials on multiple platforms</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200540621-be7e1822-1c31-4694-bec0-870499e48a5d.png" width="500">
</details>

## Version 2 - Deployment on Raspberry Pi 4 + Mobile App
#### Requirements:
```
Inference environment : Intel OpenVINO   
Processor unit        : Intel Neural Compute Stick 2 (VPU)
Local GUI             : PyQt5
Mobile App            : Android Studio
Real-time database    : Google Firebase
```
- Download 2021 OpenVINO runtime package [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/)  
> For Raspi: l_openvino_toolkit_runtime_raspbian_p_2021.4.582.tgz   
#### Run program on Raspberry Pi terminal:
- make sure to eliminate 'space' on the folder name
```
source .../openvino_2021/bin/setupvars.sh
cd .../Version_2
python3 openvino_gui.py
```
<details open>
<summary>Overall System and Setup</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200306093-427eb8bd-43b6-4e2d-aa3d-1f17e04d9063.png" width="500">
<img src="https://user-images.githubusercontent.com/76240694/200543466-df7e8343-8f1a-4e34-b7ce-387369d50290.png" width="500">
</details>

<details open>
<summary>Raspbian GUI + Mobile App + Firebase Elements</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200542264-eef30518-0e61-4869-b37f-88d629df3bbf.png" width="500">
<img src="https://user-images.githubusercontent.com/76240694/200544152-b0e2e736-9de5-4f81-8c4b-178c752ab7c8.png" width="140">
<img src="https://user-images.githubusercontent.com/76240694/200969609-bf025d7c-265c-4be7-8fce-e1b027d91e90.png" width="198">
</details>

## Sensors - Donutboard Circuit + 3D Printed Casing
```
1. Adafruit VCNL4040 Proximity and Lux Sensor - STEMMA QT / Qwiic  
    - White   
    - Light   
    - Proximity (as door exposure)   
2. DHT22 Sensor Module Breakout 
    - Temperature
    - Humidity   
```   
<img src="https://user-images.githubusercontent.com/76240694/200702980-925c5412-9b37-48d3-a359-f673c5788fb0.jpg" width="300"> <img src="https://user-images.githubusercontent.com/76240694/200703016-f23e2ebd-0b62-4545-9853-ba982f6c1fff.jpg" width="300">

## Demo Video
<img src="https://user-images.githubusercontent.com/76240694/200549257-d36fa798-5be6-4d9f-aca4-09458fb15d02.mp4" width="500">
