import PyQt4
from PyQt4 import QtGui, uic
import pkgutil
from pkg_resources import resource_filename, Requirement

def get_ui_file(uiname):
    return resource_filename(
        'peri', "gui/{}".format(uiname)
    )

def get_filename(path, parent=None):
    dialog = PyQt4.QtGui.QFileDialog(parent)
    return str(dialog.getOpenFilename(None, "Select image"))

class MyWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi(get_ui_file('main.ui'), self)
        self.show()

def launch_gui():
    import sys
    app = QtGui.QApplication(sys.argv)
    window = MyWindow()
    app.exec_()
