import PyQt4
from PyQt4 import QtGui, uic
import pkgutil
from pkg_resources import resource_filename, Requirement

def get_ui_file(uiname):
    return resource_filename(
        Requirement.parse("peri"), "peri/gui/{}".format(uiname)
    )

def get_filename(path, parent=None):
    dialog = PyQt4.QtGui.QFileDialog(parent)
    return str(dialog.getOpenFilename(None, "Select image"))

class MyWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('mywindow.ui', self)
        self.show()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
