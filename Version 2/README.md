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
