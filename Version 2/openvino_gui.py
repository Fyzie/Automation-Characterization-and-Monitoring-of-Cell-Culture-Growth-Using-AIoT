from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import os

from openvino.inference_engine import IECore
import cv2
import numpy as np
from patchify import patchify, unpatchify
import os
from PIL import Image

import adafruit_dht
import adafruit_vcnl4040
import pyrebase
import board
import time
import threading

class Ui_Dialog(object):
    def fileBrowser(self):
        try:
            folder_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Open File', directory='.')
            if folder_directory:
                self.previous_directory = folder_directory
            else:
                folder_directory = self.previous_directory
            self.lineEdit.setText(folder_directory)
            files = os.listdir(folder_directory)
            files.sort()
            self.listWidget.clear()
            image_format = ['.tif', '.tiff','.jpg', '.png']
            for file in files:
                for i in image_format:
                    if file.endswith(i):    
                        self.listWidget.addItem(file)
        except AttributeError as e:
            self.lineEdit.setText("Choose a directory!")
                
    def showImage(self):
#         print(self.image_path)
        self.image = QImage(self.image_path)
        pic = QtWidgets.QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(self.image))
        pic = self.scene.addItem(pic)
        resize = self.scene.sceneRect()
        self.graphicsView.fitInView(resize, QtCore.Qt.KeepAspectRatio)
            
    def itemClicked(self, item):
        # print("Selected items: ", item.text())
        directory = self.lineEdit.text()
        self.image_path = directory + '/' + item.text()
#         print(self.image_path)
        self.showImage()
    
    def normalize(self,x, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
        l2[l2 == 0] = 1
        return x / np.expand_dims(l2, axis)
    
    ################################################################
    # draw mask of predictions
    def segmentation_map_to_image(self,
        result: np.ndarray, colormap: np.ndarray, remove_holes=False
    ) -> np.ndarray:

        if len(result.shape) != 2 and result.shape[0] != 1:
            raise ValueError(
                f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
            )

        if len(np.unique(result)) > colormap.shape[0]:
            raise ValueError(
                f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
                "different output values. Please make sure to convert the network output to "
                "pixel values before calling this function."
            )
        elif result.shape[0] == 1:
            result = result.squeeze(0)

        result = result.astype(np.uint8)

        contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
        mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label_index, color in enumerate(colormap):
            label_index_map = result == label_index
            label_index_map = label_index_map.astype(np.uint8) * 255
            contours, hierarchies = cv2.findContours(
                label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                mask,
                contours,
                contourIdx=-1,
                color=color.tolist(),
                thickness=cv2.FILLED,
            )

        return mask
    
    def segmentCell(self):
        try:
            put()
            start = time.time()
            _translate = QtWidgets.QApplication.translate
            self.pushButton_2.setText(_translate("Dialog", "Segmenting",None))
            QtWidgets.QApplication.processEvents()
            self.indicator = 6
            self.notify()
            
            ie = IECore()
            net = ie.read_network(
                model="model_msunet/saved_model.xml")
            exec_net = ie.load_network(net, "MYRIAD")
            img_size = 256
    
            output_layer_ir = next(iter(exec_net.outputs))
            input_layer_ir = next(iter(exec_net.input_info))
            
            if self.image_path == 'segmented_image/segmented_image.jpg':
                self.image_path = self.previous_image
            large_image = cv2.imread(self.image_path)
            lab_img = cv2.cvtColor(large_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_img)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl_img = clahe.apply(l)
            updated_lab_img2 = cv2.merge((cl_img, a, b))
            bgr_image = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            large_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            
            #############################################################################
            # image normalization
            N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims
    
            image_norm = np.expand_dims(self.normalize(np.array(large_image), axis=1),2)
            image_norm = image_norm[:,:,0][:,:,None]
            input_image = np.expand_dims(image_norm.transpose(2, 0, 1), 0)
    
            #############################################################################
            # predict by patches
            # large image to small patches
            patches = patchify(large_image, (img_size, img_size), step=img_size)
    
            predicted_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    # print(i, j)
                    
                    # Run the infernece
                    single_patch = patches[i,j,:,:]
                    image_norm = np.expand_dims(self.normalize(np.array(single_patch), axis=1),2)
                    image_norm = image_norm[:,:,0][:,:,None]
                    input_image = np.expand_dims(image_norm.transpose(2, 0, 1), 0)
                    result = exec_net.infer(inputs={input_layer_ir: input_image})
                    result_ir = result[output_layer_ir]
    
                    #Predict and threshold for values above 0.5 probability
                    single_patch_prediction = (result_ir[0,0,:,:] > 0.5).astype(np.uint8)
                    predicted_patches.append(single_patch_prediction)
    
            predicted_patches = np.array(predicted_patches)
            predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], img_size, img_size))
    
            reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
            
            #############################################################################
            # confluency calculation
            num_white = np.sum(reconstructed_image == 1)
            num_black = np.sum(reconstructed_image == 0)
            self.confluency = (num_white/(num_white+num_black))*100
            # print('Confluency: {}'.format(confluency))
            
            #############################################################################
            # segmentation masking
            height, width = reconstructed_image.shape
            
            # Define colormap, each color represents a class
            colormap = np.array([[0, 0, 0], [0, 255, 0]])
    
            # Define the transparency of the segmentation mask on the photo
            alpha = 0.3
    
            # Use function from notebook_utils.py to transform mask to an RGB image
            mask = self.segmentation_map_to_image(reconstructed_image, colormap)
    
            resized_mask = cv2.resize(mask, (width, height))
    
#             rgb_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
    
            # Create image with mask put on
            image_with_mask = cv2.addWeighted(bgr_image, 1-alpha, resized_mask, alpha, 0)
            
            self.image_path = 'segmented_image/segmented_image.jpg'
            cv2.imwrite(self.image_path, image_with_mask)
            
            self.showImage()
            self.label_13.setText('{:.2f} %'.format(self.confluency))
            QtWidgets.QApplication.processEvents()
            
            end = time.time()
            print('Time taken: {:.2f} seconds'.format(end - start))
            
            if int(self.confluency) >= int(self.lineEdit_2.text()):
                self.indicator = 1 # reach confluency
            else:
                self.indicator = 2 # not reach confluency
            
            self.notify()
            
            if self.indicator == 1:
                self.pushButton_2.setText(_translate("Dialog", "Transferring",None))
                QtWidgets.QApplication.processEvents()
                self.indicator = 3
                self.notify()
                motion_2()
                
            else:
                self.pushButton_2.setText(_translate("Dialog", "Storing",None))
                QtWidgets.QApplication.processEvents()
                self.indicator = 5
                self.notify()
                motion_1()
                
            self.indicator = 4
            self.notify()
            idle()
            self.pushButton_2.setText(_translate("Dialog", "Segment",None))
            QtWidgets.QApplication.processEvents()
        except AttributeError as e:
            self.label_13.setText('No image')
            
    def adafruitSensors(self):
        try:
            dht_pin = board.D4
            i2c = board.I2C()

            dht = adafruit_dht.DHT22(dht_pin, use_pulseio=False)
            vcnl = adafruit_vcnl4040.VCNL4040(i2c)
            self.temperature = dht.temperature
            self.humidity = dht.humidity
            self.proximity = vcnl.proximity
            self.light = vcnl.light
            self.white = vcnl.white
            if self.proximity >= 6:
                # self.label_12.setText('{:.2f}'.format(self.proximity))
                self.door = "Closed"
            else:
                self.door = "Opened"
            self.door_position = self.door
            if self.humidity is not None and self.temperature is not None:
                self.label_8.setText('{:.2f}*C'.format(self.temperature))
                self.label_9.setText('{:.2f}%'.format(self.humidity))
                self.label_10.setText('{:.2f}'.format(self.white))
                self.label_11.setText('{:.2f}'.format(self.light))
                self.door = self.label_12.setText(self.door)
            self.indicator = 0 # sensors
            self.notify()
        
        except RuntimeError as e:
            pass
        
    def notify(self):
        config = {
          "apiKey": "xxx",
          "authDomain": "xxx",
          "databaseURL": "xxx",
          "storageBucket": "xxx"
        }

        firebase = pyrebase.initialize_app(config)

        db = firebase.database()
        storage = firebase.storage()
        
        if self.indicator == 0: 
            sensor_data = {
            "Temperature" : '{:.2f} ℃'.format(self.temperature),
            "Humidity" : '{:.2f} %'.format(self.humidity),
            "White" : '{:.2f}'.format(self.white),
            "Light" : '{:.2f}'.format(self.light),
            "Door" : self.door_position,
            "Indicator" : self.indicator
            }
            
            db.child("Sensors").push(sensor_data)

            db.update(sensor_data)
        
        elif self.indicator == 1 or self.indicator == 2: 
            storage.child('segmented_image.jpg').put(self.image_path)
            cell_data = {
            "Confluency" : '{:.2f} %'.format(self.confluency),
            "Indicator" : self.indicator
            }
            db.child("Cells").push(cell_data)
            db.update(cell_data)
            if self.indicator == 2 and int(self.confluency) >= 0.5*int(self.lineEdit_2.text()):
                _translate = QtWidgets.QApplication.translate
                self.pushButton_2.setText(_translate("Dialog", "Waiting",None))
                QtWidgets.QApplication.processEvents()
                for x in range (10):
                    countdown = 10 - x
                    time.sleep(1)
                    automation_data = {
                    "Countdown" : countdown
                    }
                    db.child("Automation").push(automation_data)
                    db.update(automation_data)
#                     print(countdown)
                    self.indicator = db.child("Indicator").get().val()
#                     print('Indicator: ', self.indicator)
                    if self.indicator == 1 or self.indicator == 3:
                        break;
                        # 1 - force transfer
                        # 3 - deny force transfer
        
        elif self.indicator == 3 or self.indicator == 4 or self.indicator == 5 or self.indicator == 6:
            if self.indicator == 3:
                automation_data = {
                "Countdown" : "Transferring",
                "Indicator" : self.indicator
                }
            
            elif self.indicator == 4:
                automation_data = {
                "Countdown" : "Idle",
                "Indicator" : self.indicator
                }
            elif self.indicator == 5:
                automation_data = {
                "Countdown" : "Storing",
                "Indicator" : self.indicator
                }
            else:
                automation_data = {
                "Countdown" : "Segmenting",
                "Indicator" : self.indicator
                }
            db.child("Automation").push(automation_data)
            db.update(automation_data)
        
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(798, 384)
        Dialog.setMouseTracking(False)
        
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(20, 40, 400, 300))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.cell_image = QtWidgets.QLabel(Dialog)
        self.cell_image.setGeometry(QtCore.QRect(170, 10, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cell_image.setFont(font)
        self.cell_image.setTextFormat(QtCore.Qt.AutoText)
        self.cell_image.setAlignment(QtCore.Qt.AlignCenter)
        self.cell_image.setWordWrap(False)
        self.cell_image.setObjectName("cell_image")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(490, 200, 81, 16))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(660, 200, 81, 16))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(580, 320, 81, 16))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(490, 260, 81, 16))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(660, 260, 81, 16))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(480, 220, 111, 21))
        self.label_8.setAutoFillBackground(False)
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_8.setLineWidth(2)
        self.label_8.setText("")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(640, 220, 111, 21))
        self.label_9.setAutoFillBackground(False)
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_9.setLineWidth(2)
        self.label_9.setText("")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(640, 280, 111, 21))
        self.label_10.setAutoFillBackground(False)
        self.label_10.setFrameShape(QtWidgets.QFrame.Box)
        self.label_10.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_10.setLineWidth(2)
        self.label_10.setText("")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setGeometry(QtCore.QRect(480, 280, 111, 21))
        self.label_11.setAutoFillBackground(False)
        self.label_11.setFrameShape(QtWidgets.QFrame.Box)
        self.label_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_11.setLineWidth(2)
        self.label_11.setText("")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setGeometry(QtCore.QRect(570, 340, 111, 21))
        self.label_12.setAutoFillBackground(False)
        self.label_12.setFrameShape(QtWidgets.QFrame.Box)
        self.label_12.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_12.setLineWidth(2)
        self.label_12.setText("")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(450, 180, 331, 191))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(710, 10, 71, 23))
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(550, 10, 151, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(450, 40, 331, 131))
        self.listWidget.setObjectName("listWidget")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(450, 10, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(250, 350, 71, 21))
        self.label_13.setAutoFillBackground(False)
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_13.setLineWidth(2)
        self.label_13.setText("")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(Dialog)
        self.label_14.setGeometry(QtCore.QRect(170, 350, 81, 16))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_17 = QtWidgets.QLabel(Dialog)
        self.label_17.setGeometry(QtCore.QRect(0, 350, 121, 16))
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(100, 350, 71, 20))
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.max_conf = "80"
        self.lineEdit_2.setText(self.max_conf)
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 350, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox.raise_()
        self.graphicsView.raise_()
        self.cell_image.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.pushButton.raise_()
        self.lineEdit.raise_()
        self.listWidget.raise_()
        self.label_2.raise_()
        self.label_13.raise_()
        self.label_14.raise_()
        self.pushButton_2.raise_()
        self.lineEdit_2.raise_()
        self.label_17.raise_()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.adafruitSensors)
        self.timer.start()

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(lambda: self.fileBrowser())
        self.pushButton_2.clicked.connect(self.segmentCell)
        self.listWidget.itemClicked.connect(self.itemClicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Cell Segmentation"))
        self.cell_image.setText(_translate("Dialog", "Cell Image"))
        self.label_3.setText(_translate("Dialog", "Temperature"))
        self.label_4.setText(_translate("Dialog", "Humidity"))
        self.label_5.setText(_translate("Dialog", "Door"))
        self.label_6.setText(_translate("Dialog", "Light"))
        self.label_7.setText(_translate("Dialog", "White"))
        self.groupBox.setTitle(_translate("Dialog", "Sensors"))
        self.pushButton.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "Image Folder"))
        self.label_14.setText(_translate("Dialog", "Confluency"))
        self.label_17.setText(_translate("Dialog", "Max Conf."))
        self.pushButton_2.setText(_translate("Dialog", "Segment"))
        
class Dialog(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)     

if __name__ == "__main__":
    import sys
    from servo_arm import *
    init() # initialize robotic arm position 
    app = QtWidgets.QApplication(sys.argv)
    w = Dialog()
    w.show()
    sys.exit(app.exec_())
