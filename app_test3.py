import sys
import gdspy,numpy
import matplotlib.pyplot as plt
import ipython
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton, QFileDialog

class GDS():
    def loadgds(setter):
        test = gdspy.GdsLibrary(infile = setter)
        cell = test.top_level()[0]
        polys = cell.get_polygons(by_spec = True)
        bound = cell.get_bounding_box()
        diff = bound[:,None] - bound[None,:]
        return bound, diff

    def graphingbound(diff,bound):
        plt.ioff()
        fig = plt.figure(figsize = (diff[1][0][0]/100,diff[1][0][1]/100),frameon = False)
        ax = fig.add_subplot()
        plt.axis('off')
        plt.xlim(bound[0][0],bound[1][0])
        plt.ylim(bound[0][1],bound[1][1])

    def layered():
        count = 0

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