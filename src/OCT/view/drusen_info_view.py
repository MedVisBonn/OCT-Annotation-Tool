# -*- coding: utf-8 -*-

"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import numpy as np
from PyQt4 import QtCore, QtGui

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
# Viewer for drusen information such as size, number and ...
#==============================================================================
class Ui_drusenInfoTable(object):
    
    def setupUi(self, drusenInfoTable,controller=None):
        self.controller=controller
        drusenInfoTable.setObjectName(_fromUtf8("drusenInfoTable"))
        drusenInfoTable.resize(610, 500)
        
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(drusenInfoTable.sizePolicy().hasHeightForWidth())
        drusenInfoTable.setMinimumSize(QtCore.QSize(610, 500))
        drusenInfoTable.setSizePolicy(sizePolicy)
        
        self.verticalLayout_4 = QtGui.QVBoxLayout(drusenInfoTable)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.groupBox = QtGui.QGroupBox(drusenInfoTable)
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.groupBox_2 = QtGui.QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.radioButton = QtGui.QRadioButton(self.groupBox_2)
        self.radioButton.setObjectName(_fromUtf8("radioButton"))
        self.radioButton.setChecked(True)
        self.horizontalLayout_2.addWidget(self.radioButton)
        self.radioButton_2 = QtGui.QRadioButton(self.groupBox_2)
        self.radioButton_2.setObjectName(_fromUtf8("radioButton_2"))
        self.horizontalLayout_2.addWidget(self.radioButton_2)
        spacerItem = QtGui.QSpacerItem(122, 20, QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_3.addWidget(self.label)
        self.drusenCount = QtGui.QLabel(self.groupBox)
        self.drusenCount.setObjectName(_fromUtf8("drusenCount"))
        self.horizontalLayout_3.addWidget(self.drusenCount)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_4.addWidget(self.label_2)
        self.drusenCenter = QtGui.QLabel(self.groupBox)
        self.drusenCenter.setObjectName(_fromUtf8("drusenCenter"))
        self.horizontalLayout_4.addWidget(self.drusenCenter)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_5.addWidget(self.label_3)
        self.drusenArea = QtGui.QLabel(self.groupBox)
        self.drusenArea.setObjectName(_fromUtf8("drusenArea"))
        self.horizontalLayout_5.addWidget(self.drusenArea)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        spacerItem1 = QtGui.QSpacerItem(38, 20, QtGui.QSizePolicy.Minimum,\
            QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_6.addWidget(self.label_4)
        self.drusenHeight = QtGui.QLabel(self.groupBox)
        self.drusenHeight.setObjectName(_fromUtf8("drusenHeight"))
        self.horizontalLayout_6.addWidget(self.drusenHeight)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_5 = QtGui.QLabel(self.groupBox)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_7.addWidget(self.label_5)
        self.drusenVolume = QtGui.QLabel(self.groupBox)
        self.drusenVolume.setObjectName(_fromUtf8("drusenVolume"))
        self.horizontalLayout_7.addWidget(self.drusenVolume)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        
        self.label_12 = QtGui.QLabel(self.groupBox)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.horizontalLayout_8.addWidget(self.label_12)
        
        self.drusenDiameter = QtGui.QLabel(self.groupBox)
        self.drusenDiameter.setObjectName(_fromUtf8("drusenDiameter"))
        self.horizontalLayout_8.addWidget(self.drusenDiameter)
        
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.scrollArea = QtGui.QScrollArea(self.groupBox)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 541, 347))
        self.scrollAreaWidgetContents.setObjectName(\
            _fromUtf8("scrollAreaWidgetContents"))
        self.gridLayout = QtGui.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.tableWidget = QtGui.QTableWidget(self.scrollAreaWidgetContents)
        self.tableWidget.setObjectName(_fromUtf8("tableWidget"))
        self.tableWidget.setColumnCount(5)
        self.gridLayout.addWidget(self.tableWidget, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_3.addWidget(self.scrollArea)
        self.scrollArea.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.radioButton.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.label_12.raise_()
        self.drusenDiameter.raise_()
        self.drusenCount.raise_()
        self.drusenCenter.raise_()
        self.drusenArea.raise_()
        self.drusenHeight.raise_()
        self.drusenVolume.raise_()
        
        self.groupBox_2.raise_()
        self.verticalLayout_4.addWidget(self.groupBox)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        spacerItem2 = QtGui.QSpacerItem(300, 20,\
            QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton = QtGui.QPushButton(drusenInfoTable)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtGui.QPushButton(drusenInfoTable)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.horizontalLayout_10.addLayout(self.horizontalLayout)
        self.verticalLayout_4.addLayout(self.horizontalLayout_10)

        self.retranslateUi(drusenInfoTable)
        QtCore.QMetaObject.connectSlotsByName(drusenInfoTable)
        self.tableWidget.setHorizontalHeaderLabels(["Center","Area","Height",\
            "Volume","Diameter"])
        
        self.radioButton.clicked.connect(self.pixel_unit_selected)
        self.radioButton_2.clicked.connect(self.micrometer_unit_selected)
        
        self.pushButton.clicked.connect(self.export_info)
        self.pushButton_2.clicked.connect(self.update_info)
        
    def retranslateUi(self, drusenInfoTable):
        drusenInfoTable.setWindowTitle(_translate("drusenInfoTable", "Drusen Analysis", None))
        self.groupBox_2.setTitle(_translate("drusenInfoTable", "Measurment unit", None))
        self.radioButton.setText(_translate("drusenInfoTable", "Pixel", None))
        self.radioButton_2.setText(_translate("drusenInfoTable", "Micrometer", None))
        self.label.setText(_translate("drusenInfoTable", "Count:", None))
        self.drusenCount.setText(_translate("drusenInfoTable", "13", None))
        self.label_2.setText(_translate("drusenInfoTable", "Centeral position:", None))
        self.drusenCenter.setText(_translate("drusenInfoTable", "(1,2)", None))
        self.label_3.setText(_translate("drusenInfoTable", "Average area:", None))
        self.drusenArea.setText(_translate("drusenInfoTable", "36", None))
        self.label_4.setText(_translate("drusenInfoTable", "Average height:", None))
        self.drusenHeight.setText(_translate("drusenInfoTable", "4", None))
        self.label_5.setText(_translate("drusenInfoTable", "Average volume:", None))
        self.drusenVolume.setText(_translate("drusenInfoTable", "59", None))
        self.label_12.setText(_translate("drusenInfoTable", "(Large, Small) diameter:", None))
        self.drusenDiameter.setText(_translate("drusenInfoTable", "(7,4)", None))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)

        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.pushButton.setText(_translate("drusenInfoTable", "&Export", None))
        self.pushButton_2.setText(_translate("drusenInfoTable", "&Update", None))
        
    def set_data(self,cx, cy, area, height, volume, largeR, smallR, theta):

        numDrusen=len(cx)
        self.drusenCount.setText(str(numDrusen))
        self.drusenCenter.setText("("+str(round(np.average(cx),1))+","+\
            str(round(np.average(cy),1))+")")
        self.drusenArea.setText(str(round(np.average(area),1)))
        self.drusenHeight.setText(str(round(np.average(height),1)))
        self.drusenVolume.setText(str(round(np.average(volume),1)))
        self.drusenDiameter.setText("("+str(round(np.average(largeR),1))+\
            ","+str(round(np.average(smallR),1))+")")
        
        self.tableWidget.setRowCount(numDrusen)
        
        self.tableWidget.setHorizontalHeaderLabels(["Center","Area","Height",\
            "Volume","Diameter"])
        
        for i in range(numDrusen):
            item = QtGui.QTableWidgetItem()
            self.tableWidget.setVerticalHeaderItem(i, item)
            
            item = QtGui.QTableWidgetItem()
            self.tableWidget.setItem(i, 0, item)
            item = QtGui.QTableWidgetItem()
            self.tableWidget.setItem(i, 1, item)
            item = QtGui.QTableWidgetItem()
            self.tableWidget.setItem(i, 2, item)
            item = QtGui.QTableWidgetItem()
            self.tableWidget.setItem(i, 3, item)
            item = QtGui.QTableWidgetItem()
            self.tableWidget.setItem(i, 4, item)
            
            item = self.tableWidget.verticalHeaderItem(i)
            item.setText(_translate("drusenInfoTable", str(i+1), None))
        
            item = self.tableWidget.item(i, 0)
            item.setText(_translate("drusenInfoTable", "("+str(round(cx[i],1))+\
                ","+str(round(cy[i],1))+")", None))

            item = self.tableWidget.item(i, 1)
            item.setText(_translate("drusenInfoTable", str(round(area[i],1)), None))

            item = self.tableWidget.item(i, 2)
            item.setText(_translate("drusenInfoTable", str(round(height[i],1)), None))

            item = self.tableWidget.item(i, 3)
            item.setText(_translate("drusenInfoTable", str(round(volume[i],1)), None))
            
            item = self.tableWidget.item(i, 4)
            item.setText(_translate("drusenInfoTable", "("+str(round(largeR[i],\
                1))+","+str(round(smallR[i],1))+")", None))
       
    def enable_export(self):
        self.pushButton.setEnabled(True)
        
    def pixel_unit_selected(self):
        self.controller.pixel_unit_selected()
        
    def micrometer_unit_selected(self):
        self.controller.micrometer_unit_selected()
        
    def export_info(self):
        self.controller.export_drusen_analysis()
        self.pushButton.setEnabled(False)
    
    def update_info(self):
        self.controller.update_drusen_analysis()
        self.pushButton.setEnabled(True)

