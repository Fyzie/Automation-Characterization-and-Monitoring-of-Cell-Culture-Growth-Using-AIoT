## Version 0 - Cell Segmentation + Firebase Realtime Database
> Refer [here](https://github.com/Fyzie/Cell-segmentation-using-U-Net-based-networks) for model training using Tensorflow   
> Reference to use GPU for Tensorflow [[link](https://www.youtube.com/watch?v=hHWkvEcDBO0&t=335s)]

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

Reference for Firebase Realtime Database [[link](https://www.youtube.com/watch?v=8IWalfRUk1M)]   
