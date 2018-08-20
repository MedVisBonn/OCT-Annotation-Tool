"""
Created in 2018

@author: Shekoufeh Gorgi Zadeh
"""

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
# Open file dialog
#==============================================================================
def get_scan_path():
    # The QWidget widget is the base class of all user interface objects in PyQt4.
    w = QtGui.QWidget()
    # Set window size. 
    w.resize(320, 240)
    # Get file name
    filename=QtGui.QFileDialog.getOpenFileName(w,'Open File','','B-Scan Directory')
    return filename
