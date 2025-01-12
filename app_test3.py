import sys
import gdspy,numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.switch_backend('agg')
from PIL import Image
from shapely import Polygon
#import ipython
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton, QFileDialog, QLineEdit, QFormLayout, QLabel
from PySide6.QtGui import QDoubleValidator
from imageio import imread as imreader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

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

    def layered(bound,bound1,fig,polys):
        count = 0
        filenames = []
        matlist = []
        for a,coords in polys.items():
            plt.clf()
            fig.canvas.draw()
            colors = ['black','red','blue','magenta','green','orange']
            for b in coords:
                m = Polygon(b)
                t = np.array(m.exterior.xy)
                for i in np.arange(0,t.shape[1]):
                    if bound1[0][0] < t[0][i] < bound1[1][0] and bound1[0][1] < t[1][i] < bound1[1][1]:
                        adjust = t - np.tile(np.array([[bound[0][0],bound[0][1]]]).transpose(),(1,t.shape[1]))
                        scaled = adjust//np.array([[1,1]]).transpose()
                        cor = Polygon(list(zip(scaled[0],scaled[1])))
                        print(cor)
                        plt.fill(*cor.exterior.xy)#color = colors[count])
                        break
            count +=1
            col = fig.canvas.buffer_argb()
            mat = np.asarray(col)
            print(col)
            dim = fig.canvas.get_width_height()[::-1]
            mat = np.frombuffer(col,dtype = np.uint8).reshape(dim + (4,))
            matlist.append(col)#[:,:,0])
            layername = 'layer' + str(count) + '.png'
            filenames.append(layername)
            fig.savefig(layername,dpi = 100)

            plt.show()
        return filenames, matlist

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
            #matimg = imreader(a)/255
            boolmat = (a[:,1] != 1).astype(int)
            matlist += (boolmat,)
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
            samp = np.broadcast_to(b,(5,b.shape[0],b.shape[1]))
            samplist += (samp,)
            print(count)
            count += 1
        sample = np.concatenate(samplist,axis = 0)
        return sample
    
    def graph(sample):
        ax.voxels(sample)
        return ax
    
class voxelview(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('View')
        graph = MainWindow.button_clicked()
        ax = GDS.graph(graph)

class graphsample(FigureCanvasQTAgg):
    def __init__(self):
        
        ax = plt.figure()
        self.ax1 = ax.add_subplot(projection = '3d')
        super().__init__(ax)
        
class graphwin(QWidget):
    def __init__(self,sample):
        
        super().__init__()
        layform = QFormLayout(self)
        self.sample = sample
        self.canvas = graphsample()
        sample = self.sample
        self.setWindowTitle('Graph Sample')
        self.canvas.ax1.voxels(sample)
        layform.addWidget(self.canvas)
        self.setLayout(layform)

    

class MainWindow(QMainWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("GDS Voxels v.0.01")
        wid = QWidget()

        gdsbutton = QPushButton("Load Design")
        gdsbutton.clicked.connect(self.button_clicked)
        #self.setCentralWidget(gdsbutton)
        self.fname = QLineEdit()
        self.fname.setText('None')

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

        self.confirm = QPushButton('Confirm')
        self.confirm.clicked.connect(self.graphset)

        inwidget = [gdsbutton,self.fname,x1,self.bound_x1,x2,self.bound_x2,y1,self.bound_y1,y2,self.bound_y2,self.confirm]

        self.layoutin = QFormLayout()

        for widget in inwidget:
            self.layoutin.addRow(widget)

        wid.setLayout(self.layoutin)
        self.setCentralWidget(wid)

        

    def button_clicked(self):
        print("clicked")
        lwin,setter = QFileDialog.getOpenFileName(self,'GDS Loader',filter = 'GDS (*.gds)')
        self.fname.setText(lwin)

    def graphset(self):
        lwin = self.fname.text()
        if lwin != 'None':
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
            self.sample = None
            if boolcheck == False:
                print('Error')
            else:
                figcustom = GDS.graphingbound(diff1,bound1)
                layers, matcol = GDS.layered(bound,bound1,figcustom,polys)
                matlist = GDS.loadlayers(matcol)
                self.sample = GDS.loadsample(matcol)
                if self.sample.all() != None:
                    self.win = graphwin(self.samplereturn())
                    self.win.show()
            
        
    def samplereturn(self):
        return self.sample
        
    
        
        

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()

#Created: December 16, 2024 - new version
#Use for tutorial only