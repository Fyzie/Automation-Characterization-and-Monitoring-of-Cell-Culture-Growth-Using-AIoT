# Automation-Characterization-and-Monitoring-of-Cell-Culture-Growth-Using-AIoT
Project Continuation from [This Repo](https://github.com/Fyzie/Autonomous-Cell-Counting-and-Handling)
## Version 1 - Cell Segmentation + GUI
How to convert tensorflow model (eg. hdf5) to pb format:
```
import tensorflow as tf
import keras

model = tf.keras.models.load_model('model.hdf5')
tf.saved_model.save(model,'model.pb')
```
- Download 2021 OpenVINO developer package [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/)  
> For Windows: w_openvino_toolkit_dev_p_2021.4.582.zip

How to convert pb format to ir format:
```
cd ...\openvino_2021\bin
setupvars.bat
cd ...\openvino_2021\deployment_tools\model_optimizer
python mo.py --saved_model_dir <model_pb_folder> --output_dir IR --input_shape [1,<input_shape>,<input_shape>,1]
```
<details open>
<summary>Local GUI</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200532356-0d42cbda-1155-4b6e-af7b-5ab82a9d6e45.png" width="600">
</details>
<details open>
  
<summary>Trials on multiple platforms</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200540621-be7e1822-1c31-4694-bec0-870499e48a5d.png" width="600">
</details>


## Version 2 - Deployment on Raspberry Pi 4
```
Inference environment : Intel OpenVINO   
Processor unit        : Intel Neural Compute Stick 2 (VPU)
```
<details open>
<summary>Overall System</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200306093-427eb8bd-43b6-4e2d-aa3d-1f17e04d9063.png" width="600">
</details>

<details open>
<summary>Overall Setup</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200543466-df7e8343-8f1a-4e34-b7ce-387369d50290.png" width="600">
</details>

<details open>
<summary>Raspbian GUI + Mobile App</summary>
<br>
<img src="https://user-images.githubusercontent.com/76240694/200542264-eef30518-0e61-4869-b37f-88d629df3bbf.png" width="600">
<img src="https://user-images.githubusercontent.com/76240694/200544152-b0e2e736-9de5-4f81-8c4b-178c752ab7c8.png" width="160">

</details>

- Download 2021 OpenVINO runtime package [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/)  
> For Raspi: l_openvino_toolkit_runtime_raspbian_p_2021.4.582.tgz
