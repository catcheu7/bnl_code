import sys
import gdspy,numpy
import matplotlib.pyplot as plt
import ipython
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton, QFileDialog

class GDS():
    def loadgds(setter):
        print(setter)
        test = gdspy.GdsLibrary(infile = setter)
        cell = test.top_level()[0]
        polys = cell.get_polygons(by_spec = True)
#polys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GDS Voxels v.0.01")
        wid = QWidget()

        gdsbutton = QPushButton("Load Design")
        gdsbutton.clicked.connect(self.button_clicked)
        self.setCentralWidget(gdsbutton)

    def button_clicked(self,s):
        print("clicked")
        lwin,setter = QFileDialog.getOpenFileName(self,'GDS Loader',filter = 'GDS (*.gds)')
        lwin.exec()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()

#Created: December 16, 2024 - new version
#Use for tutorial only