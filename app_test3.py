import sys
import gdspy,numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
#import ipython
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton, QFileDialog, QLineEdit, QFormLayout, QLabel
from PySide6.QtGui import QDoubleValidator
from imageio import imread as imreader

class GDS():
    def loadgds(setter):
        test = gdspy.GdsLibrary(infile = setter)
        cell = test.top_level()[0]
        polys = cell.get_polygons(by_spec = True)
        bound = cell.get_bounding_box()
        diff = bound[:,None] - bound[None,:]
        print(diff)
        return bound, diff,polys

    def graphingbound(diff,bound):
        plt.ioff()
        fig = plt.figure(figsize = (diff[0]/100,diff[1]/100),frameon = False)
        ax = fig.add_subplot()
        plt.axis('off')
        plt.xlim(0,diff[0])
        plt.ylim(0,diff[1])
        return fig

    def layered(bound,fig,polys):
        count = 0
        filenames = []
        for a,coords in polys.items():
            colors = ['black','red','blue','magenta','green','orange']
            for b in coords:
                m = Polygon(b)
                t = np.array(m.exterior.xy)
                for i in np.arange(0,t.shape[1]):
                    if bound[0][0] < t[0][i] < bound[1][0] and bound[0][1] < t[1][i] < bound[1][1]:
                        adjust = t - np.tile(np.array([[bound[0][0],bound[0][1]]]).transpose(),(1,t.shape[1]))
                        scaled = adjust//np.array([[0.5,0.5]]).transpose()
                        cor = Polygon(list(zip(scaled[0],scaled[1])))
                        print(cor)
                        plt.fill(*cor.exterior.xy,color = colors[count])
                        break
            plt.clf()
            count +=1
            layername = 'layer' + str(count) + '.png'
            filenames.append(layername)
            fig.savefig(layername,dpi = 100)
        return filenames

    def checkbounds(self,x1,x2,y1,y2,bound,diff):
        if x1 < bound[0][0] or x1 > bound[1][0]:
            self.errormes.setText('Error: Starting X out of bounds!')
            return False
        elif x2 < bound[0][0] or x2 > bound[1][0]:
            self.errormes.setText('Error: Ending X out of bounds!')
        elif x2 < x1 or x1 > x2:
            self.errormes.setText('Error: Invalid Range')
        elif y2 < y1 or y1 > y2:
            self.errormes.setText('Error: Invalid Range!')
            return False
        elif y1 < bound[0][1] or y1 > bound[1][1] or y2 < bound[0][1] or y2 > bound[1][1]:
            self.errormes.setText('Error: Y out of bounds!')
            return False
        else:
            self.errormes.setText('')
            return True
        
    def loadlayers(layers):
        matlist = ()
        for a in layers:
            matimg = imreader(a)/255
            matlist += (matimg,)
        return matlist
    
    """ def loadsample(layers):
        samplist = ()
        for b in layers:
            samp = np.broadcast(5,b.shape()[0],b.shape()[1])
            samplist += (samp,)
        sample = np.concatenate(samplist,axis = 0)
        return sample """
    
    def loadsample(layers):
        samplist = ()
        count = 0
        for b in layers:
            samp = np.broadcast(b,(5,b.shape[0],b.shape[1]))
            samplist += (samp,)
            print(count)
            count += 1
        sample = np.concatenate(samplist,axis = 0)
        return sample
    
    def graph(sample):
        ax = plt.figure().add_subplot()
        ax.voxels(sample)
        return ax
    
class voxelview(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('View')
        graph = MainWindow.button_clicked()
        ax = GDS.graph(graph)

class graphsample(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Graph')
        graph = MainWindow.button_clicked()
        ax = plt.figure().add_subplot()
        voxels = GDS.graph(graph)
        self.setCentralWidget(voxels)
        self.show()

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

        inwidget = [gdsbutton,x1,self.bound_x1,x2,self.bound_x2,y1,self.bound_y1,y2,self.bound_y2]

        self.layoutin = QFormLayout()

        for widget in inwidget:
            self.layoutin.addRow(widget)

        wid.setLayout(self.layoutin)
        self.setCentralWidget(wid)

    def button_clicked(self,s):
        print("clicked")
        lwin,setter = QFileDialog.getOpenFileName(self,'GDS Loader',filter = 'GDS (*.gds)')
        bound,diff,polys = GDS.loadgds(lwin)
        xstart = float(self.bound_x1.text())
        xend = float(self.bound_x2.text())
        ystart = float(self.bound_y1.text())
        yend = float(self.bound_y2.text())
        self.errormes = QLabel()
        self.layoutin.addRow(self.errormes)
        bound1 = [(xstart,ystart),(xend,yend)]
        diff1 = [(xend - xstart),(yend - ystart)]
        boolcheck = GDS.checkbounds(self,xstart,xend,ystart,yend,bound1,diff1)
        sample = None
        if boolcheck == False:
            print('Error')
        else:
            figcustom = GDS.graphingbound(diff1,bound1)
            layers = GDS.layered(bound1,figcustom,polys)
            matlist = GDS.loadlayers(layers)
            sample = GDS.loadsample(matlist)
            return sample
        
        if sample != None:
            win = graphsample()
            win.exec()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()

#Created: December 16, 2024 - new version
#Use for tutorial only