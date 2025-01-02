import sys
import gdspy,numpy
import matplotlib.pyplot as plt
#import ipython
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton, QFileDialog, QLineEdit, QFormLayout, QLabel
from PySide6.QtGui import QDoubleValidator

class GDS():
    def loadgds(setter):
        test = gdspy.GdsLibrary(infile = setter)
        cell = test.top_level()[0]
        polys = cell.get_polygons(by_spec = True)
        bound = cell.get_bounding_box()
        diff = bound[:,None] - bound[None,:]
        print(diff)
        return bound, diff

    def graphingbound(diff,bound):
        plt.ioff()
        fig = plt.figure(figsize = (diff[1][0][0]/100,diff[1][0][1]/100),frameon = False)
        ax = fig.add_subplot()
        plt.axis('off')
        plt.xlim(bound[0][0],bound[1][0])
        plt.ylim(bound[0][1],bound[1][1])
        return fig

    def layered():
        count = 0
        for a,coords in polys.items():
            colors = ['black','red','blue','magenta','green','orange']
            for b in coords:
                m = Polygon(b)
                plt.fill(*m.exterior.xy,color = colors[count])
            count +=1
        plt.savefig('test.png',dpi = 100)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GDS Voxels v.0.01")
        wid = QWidget()

        gdsbutton = QPushButton("Load Design")
        gdsbutton.clicked.connect(self.button_clicked)
        #self.setCentralWidget(gdsbutton)

        self.bound_x1 = QLineEdit()
        x1 = QLabel('X: Starting')
        self.bound_x1.setValidator(QDoubleValidator())

        self.bound_x2 = QLineEdit()
        x2 = QLabel('X: Ending')
        self.bound_x2.setValidator(QDoubleValidator())

        self.bound_y1 = QLineEdit()
        y1 = QLabel('Y: Starting')
        self.bound_y1.setValidator(QDoubleValidator())

        self.bound_y2 = QLineEdit()
        y2 = QLabel('Y: Ending')
        self.bound_y2.setValidator(QDoubleValidator())

        inwidget = [gdsbutton,x1,bound_x1,x2,bound_x2,y1,bound_y1,y2,bound_y2]

        layoutin = QFormLayout()

        for widget in inwidget:
            layoutin.addRow(widget)

        wid.setLayout(layoutin)
        self.setCentralWidget(wid)

    def button_clicked(self,s):
        print("clicked")
        lwin,setter = QFileDialog.getOpenFileName(self,'GDS Loader',filter = 'GDS (*.gds)')
        bound,diff = GDS.loadgds(lwin)
        fig2 = GDS.graphingbound(diff,bound)
        xstart = self.bound_x1.text()
        xend = self.bound_x2.text()
        ystart = self.bound_y1.text()
        yend = self.bound_y2.text()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()

#Created: December 16, 2024 - new version
#Use for tutorial only