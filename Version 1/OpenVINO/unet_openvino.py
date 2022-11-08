from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import os

from openvino.inference_engine import IECore
import cv2
import numpy as np
from patchify import patchify, unpatchify

import time 

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
            self.listWidget.clear()
            image_format = ['.jpg', '.png', '.tif', '.tiff']
            for file in files:
                for i in image_format:
                    if file.endswith(i):
                        self.listWidget.addItem(file)
        except AttributeError as e:
            self.lineEdit.setText("Choose another directory!")
                
    def showImage(self):
        self.image = QImage(self.image_path)
        pic = QtWidgets.QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(self.image))
        pic = self.scene.addItem(pic)
        resize = self.scene.sceneRect()
        self.graphicsView.fitInView(resize, QtCore.Qt.KeepAspectRatio)
        
    def itemClicked(self, item):
        # print("Selected items: ", item.text())
        directory = self.lineEdit.text()
        self.current_image = item.text()
        self.image_path = directory + '/' + item.text()
        self.previous_image = self.image_path
        # print(image_path)
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
            start = time.time()
            _translate = QtWidgets.QApplication.translate
            self.pushButton_2.setText(_translate("Dialog", "Segmenting",None))
            QtWidgets.QApplication.processEvents()
            
            ie = IECore()
            net = ie.read_network(model="model_unet/saved_model.xml")
            exec_net = ie.load_network(net, "CPU") # "CPU" - Intel CPU | "GPU" - Intel GPU | "MYRIAD" - Intel NCS VPU
            img_size = 512
    
            output_layer_ir = next(iter(exec_net.outputs))
            input_layer_ir = next(iter(exec_net.input_info))
            
            if self.image_path == 'unet_' + self.current_image + '.jpg':
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
            confluency = (num_white/(num_white+num_black))*100
            confluency = '{:.2f} %'.format(confluency)
            # print('Confluency: {}'.format(confluency))
            
            #############################################################################
            # segmentation masking
            height, width = reconstructed_image.shape
            
            # Define colormap, each color represents a class
            # colormap = np.array([[68, 1, 84], [255, 216, 52]])
            colormap = np.array([[0, 0, 0], [0, 255, 0]])
    
            # Define the transparency of the segmentation mask on the photo
            alpha = 0.3
    
            # Use function from notebook_utils.py to transform mask to an RGB image
            mask = self.segmentation_map_to_image(reconstructed_image, colormap)
    
            resized_mask = cv2.resize(mask, (width, height))
    
            # rgb_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
    
            # Create image with mask put on
            image_with_mask = cv2.addWeighted(bgr_image, 1-alpha, resized_mask, alpha, 0)
            
            self.image_path = 'unet_' + self.current_image + '.jpg'
            cv2.imwrite(self.image_path, image_with_mask)
            
            end = time.time()
            print('Time taken: {:.2f} seconds'.format(end - start))
            
            self.pushButton_2.setText(_translate("Dialog", "Segment",None))
            QtWidgets.QApplication.processEvents()
            self.showImage()
            self.label_13.setText(confluency)
        except AttributeError as e:
            self.label_13.setText("No image")
            self.pushButton_2.setText(_translate("Dialog", "Segment",None))
            QtWidgets.QApplication.processEvents()
        
        
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
        self.label_13.setGeometry(QtCore.QRect(90, 350, 221, 21))
        self.label_13.setAutoFillBackground(False)
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_13.setLineWidth(2)
        self.label_13.setText("")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(Dialog)
        self.label_14.setGeometry(QtCore.QRect(10, 350, 81, 16))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 350, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.graphicsView.raise_()
        self.cell_image.raise_()
        self.pushButton.raise_()
        self.lineEdit.raise_()
        self.listWidget.raise_()
        self.label_2.raise_()
        self.label_13.raise_()
        self.label_14.raise_()
        self.pushButton_2.raise_()

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(lambda: self.fileBrowser())
        self.pushButton_2.clicked.connect(self.segmentCell)
        self.listWidget.itemClicked.connect(self.itemClicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Cell Segmentation"))
        self.cell_image.setText(_translate("Dialog", "Cell Image"))
        self.pushButton.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "Image Folder"))
        self.label_14.setText(_translate("Dialog", "Confluency"))
        self.pushButton_2.setText(_translate("Dialog", "Segment"))
        
class Dialog(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)     

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Dialog()
    w.show()
    sys.exit(app.exec_())
