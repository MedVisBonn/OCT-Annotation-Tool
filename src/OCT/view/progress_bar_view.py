"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

import os
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
# Viewer for a progress bar
#==============================================================================
class Ui_FormProgressBar(object):
    
    def setupUi(self, FormProgressBar):
        FormProgressBar.setObjectName(_fromUtf8("FormProgressBar"))
        FormProgressBar.resize(339, 102)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(FormProgressBar.sizePolicy().hasHeightForWidth())
        FormProgressBar.setSizePolicy(sizePolicy)
        FormProgressBar.setMinimumSize(QtCore.QSize(250, 102))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(os.path.join("icons",\
            "633563-basic-icons","png","233-empty.png"))), QtGui.QIcon.Normal,\
            QtGui.QIcon.Off)
        FormProgressBar.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(FormProgressBar)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(FormProgressBar)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.progressBar = QtGui.QProgressBar(FormProgressBar)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,\
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(228, 20, QtGui.QSizePolicy.Expanding,\
            QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.value=0
        self.verticalLayout.addLayout(self.horizontalLayout)
        FormProgressBar.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.retranslateUi(FormProgressBar)
        QtCore.QMetaObject.connectSlotsByName(FormProgressBar)

    def retranslateUi(self, FormProgressBar):
        FormProgressBar.setWindowTitle(_translate("FormProgressBar", "Progress",\
            None))
        self.label.setText(_translate("FormProgressBar", "Loading", None))

    def update_progress_using_step(self,step):
        self.value=step+self.value
        self.set_progress_bar_value(self.value)
        
    def set_progress_bar_value(self,value):
        self.progressBar.setValue(value)
        QtGui.QApplication.processEvents()
        
    def get_progress_bar_value(self):
        value=self.progressBar.value()
        return value
        
    def reset_value(self):
        self.value=0
        
    def set_text(self,text):
        self.label.setText(_translate("FormProgressBar", text, None))
        
