# -*- coding: utf-8 -*-
"""
Created in 2018

@author: Tabea Viviane Riepe
"""

import os, inspect
from PyQt4 import QtCore, QtGui

global sfwPath
sfwPath=os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))[:-4]

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

#==============================================================================
# Form for inserting deep learning information such as network name, address...
#==============================================================================
class Ui_settings(object):
    
    def setupUi(self, settings,controller=None):
        self.controller=controller
        settings.setObjectName(_fromUtf8("settings"))
        settings.resize(610, 500)
        
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(settings.sizePolicy().hasHeightForWidth())
        settings.setMinimumSize(QtCore.QSize(610, 500))
        settings.setSizePolicy(sizePolicy)
        
        self.network = QtGui.QLabel('Network')
        networkb = QtGui.QPushButton("Choose Folder")
        self.path_chosen = QtGui.QLineEdit()
        networkb.clicked.connect(self.getfile) 
        
        
        self.caffePath = QtGui.QLabel('Caffe Path')
        caffePathb= QtGui.QPushButton("Choose Folder")
        self.path_chosen2 = QtGui.QLineEdit()
        caffePathb.clicked.connect(self.getfile2) 
        
        #Combobox:
        self.processor = QtGui.QLabel('Processor')
        self.processor_edit = QtGui.QComboBox()
        self.processor_edit.addItems(['gpu','cpu'])
        
        #LineEdit:
        self.processor_ID = QtGui.QLabel('Processor ID')
        self.processor_ID_edit = QtGui.QLineEdit()
        self.processor_ID_edit.setValidator(QtGui.QIntValidator())
        
        self.trainModelFile = QtGui.QLabel('Name of the training model file')
        self.trainModelFile_edit = QtGui.QLineEdit()
        
        self.modelFile = QtGui.QLabel('Name of the model file')
        self.modelFile_edit = QtGui.QLineEdit()
        
        #Checkbox:
        self.normImage = QtGui.QLabel('Normalize Image')
        self.normImage_edit = QtGui.QCheckBox()
        
        self.zeroCenter = QtGui.QLabel('Zero Center')
        self.zeroCenter_edit = QtGui.QCheckBox()
        
        #Spinbox
        self.numOfTiles = QtGui.QLabel('Number of Tiles')
        self.numOfTiles_edit = QtGui.QSpinBox()
        self.numOfTiles_edit.setRange(1,10)
        self.numOfTiles_edit.setValue(2)
        
        self.downSampleFactor = QtGui.QLabel('Down Sample Factor')
        self.downSampleFactor_edit = QtGui.QSpinBox()
        self.downSampleFactor_edit.setRange(1,200)
        
        #Create the Ok and Cancel buttons
        button_ok = QtGui.QPushButton('Ok')
        button_cancel = QtGui.QPushButton('Cancel')
        
        #connect the buttons to its function 
        button_ok.clicked.connect(self.on_accept_clicked)
        button_cancel.clicked.connect(self.cancel_action)
        
        
        #Now the layout is created and the widgets and buttons are added:
        
        #First the vertical layouts for network and caffe path are defined:
        net_caf = QtGui.QVBoxLayout()
        net_caf.addWidget(self.network)
        net_caf.addWidget(self.caffePath)
        
        net_caf2 = QtGui.QVBoxLayout()
        net_caf2.addWidget(self.path_chosen)
        net_caf2.addWidget(self.path_chosen2) 
           
        net_caf3 = QtGui.QVBoxLayout()
        net_caf3.addWidget(networkb)
        net_caf3.addWidget(caffePathb)
        
        #Now the columns are combined:
        hbox_net_cafe = QtGui.QHBoxLayout()
        hbox_net_cafe.addLayout(net_caf)
        hbox_net_cafe.addLayout(net_caf2)
        hbox_net_cafe.addLayout(net_caf3)
        
        #First the vertical layouts for processor and processorID are defined:
        pro_pid = QtGui.QVBoxLayout()
        pro_pid.addWidget(self.processor)
        pro_pid.addWidget(self.processor_ID)
        
        pro_pid2 = QtGui.QVBoxLayout()
        pro_pid2.addWidget(self.processor_edit)
        pro_pid2.addWidget(self.processor_ID_edit)

        #Now the columns are combined:
        hbox_pro_pid = QtGui.QHBoxLayout()
        hbox_pro_pid.addLayout(pro_pid)
        hbox_pro_pid.addLayout(pro_pid2)
        
        #First the vertical layout for trainings model file and model file is defined:
        tmf_mf = QtGui.QVBoxLayout()
        tmf_mf.addWidget(self.trainModelFile)
        tmf_mf.addWidget(self.modelFile)
        
        tmf_mf2 = QtGui.QVBoxLayout()
        tmf_mf2.addWidget(self.trainModelFile_edit)
        tmf_mf2.addWidget(self.modelFile_edit)
        
        #Now the columns are combined:
        hbox_tmf_mf = QtGui.QHBoxLayout()
        hbox_tmf_mf.addLayout(tmf_mf)
        hbox_tmf_mf.addLayout(tmf_mf2)
        
        
        # horizontal box for norm image
        hbox_ni = QtGui.QHBoxLayout()
        hbox_ni.addWidget(self.normImage)
        hbox_ni.addWidget(self.normImage_edit) 
        
        # horizontal box for zero center
        hbox_zc = QtGui.QHBoxLayout()
        hbox_zc.addWidget(self.zeroCenter)
        hbox_zc.addWidget(self.zeroCenter_edit) 
        
        # horizontal box for number of tiles
        hbox_not = QtGui.QHBoxLayout()
        hbox_not.addWidget(self.numOfTiles)
        hbox_not.addWidget(self.numOfTiles_edit) 

        # horizontal box for number of down sample factor
        hbox_dsf = QtGui.QHBoxLayout()
        hbox_dsf.addWidget(self.downSampleFactor)
        hbox_dsf.addWidget(self.downSampleFactor_edit)
        
        #horizontal box for the buttons
        hbox_but = QtGui.QHBoxLayout()
        hbox_but.addWidget(button_ok)
        hbox_but.addWidget(button_cancel)
        
        # vertical layout to combine all the horizontal layouts
        final = QtGui.QVBoxLayout(settings)
        final.addLayout(hbox_net_cafe)
        final.addLayout(hbox_pro_pid)
        final.addLayout(hbox_tmf_mf)
        final.addLayout(hbox_ni)        
        final.addLayout(hbox_zc)
        final.addLayout(hbox_not)
        final.addLayout(hbox_dsf)
        final.addLayout(hbox_but)
        
        
        settings.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        settings.setWindowTitle(_translate("settings", "Setting", None))
        
        self.settings=settings
        
        
        #If the log file already exists, the settings are imported into the window:
        if os.path.isfile(os.path.join(sfwPath,'controller','settings.csv')):
            with open(os.path.join(sfwPath,'controller','settings.csv')) as f:
                content = f.read().split('\n')
            mod_content = []
            for element in content:
                mod_content.append(element[element.find(',')+1:])
            self.path_chosen.setText(mod_content[0])
            self.path_chosen2.setText(mod_content[1])
            
            if mod_content[2] == 'gpu':
                self.processor_edit.setCurrentIndex(0)
            else:
                self.processor_edit.setCurrentIndex(1)
                
            self.processor_ID_edit.setText(mod_content[3])
            self.trainModelFile_edit.setText(mod_content[4])
            self.modelFile_edit.setText(mod_content[5])
            
            if mod_content[6] == 'yes':
                self.normImage_edit.setChecked(True)
            if mod_content[7] == 'yes':
                self.zeroCenter_edit.setChecked(True)
                
            self.numOfTiles_edit.setValue(int(mod_content[8]))

            self.downSampleFactor_edit.setValue(int(mod_content[9]))
            

    def getfile(self):
      fdir=QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:',\
          '', QtGui.QFileDialog.ShowDirsOnly)
      self.path_chosen.setText(fdir)
     
    def getfile2(self):
      fdir=QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:',\
          '', QtGui.QFileDialog.ShowDirsOnly)
      self.path_chosen2.setText(fdir)
      
    def on_accept_clicked(self):
        '''This function writes the inserted data into a csv file.'''
        with open(os.path.join(sfwPath,'controller','settings.csv'),'w') as f:
            f.write(self.network.text() + ',' + self.path_chosen.text() + '\n')
            f.write(self.caffePath.text() + ',' + self.path_chosen2.text() + '\n')
            f.write(self.processor.text() + ',' + self.processor_edit.currentText() + '\n')
            f.write(self.processor_ID.text() + ',' + self.processor_ID_edit.text() + '\n')
            f.write(self.trainModelFile.text() + ',' + self.trainModelFile_edit.text() + '\n')
            f.write(self.modelFile.text() +',' + self.modelFile_edit.text() + '\n')
            
            if self.normImage_edit.isChecked():
                f.write(self.normImage.text() + ',yes\n')
            else:
                f.write(self.normImage.text() + ',no\n')
            
            if self.zeroCenter_edit.isChecked():
                f.write(self.zeroCenter.text() + ',yes\n')
            else:
                f.write(self.zeroCenter.text() + ',no\n')
                
            f.write(self.numOfTiles.text() + ',' +\
                str(self.numOfTiles_edit.value()) + '\n')
            f.write(self.downSampleFactor.text() +\
                ',' + str(self.downSampleFactor_edit.value()))
         
        self.settings.close()
        
    def get_network_info(self):
        sett=dict()
        sett['netPath']=str(self.path_chosen.text())
        sett['caffePath']=str(self.path_chosen2.text())
        sett['processor']=str(self.processor_edit.currentText())
        sett['processorId']=int(self.processor_ID_edit.text())
        sett['trainModelFile']=str(self.trainModelFile_edit.text())
        sett['modelFile']=str(self.modelFile_edit.text())
        sett['normImage']=self.normImage_edit.isChecked()
        sett['scaleImage']=1
        sett['zeroCenter']=self.zeroCenter_edit.isChecked()
        sett['numOfTiles']=self.numOfTiles_edit.value()
        sett['downSampleFactor']=self.downSampleFactor_edit.value()  
        sett['d4a_size']=0
        return sett
        
    def cancel_action(self):
        self.settings.close()


