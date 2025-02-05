import sys
import random
import random
import OpenGL
import gdspy,numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon as poly
matplotlib.use('agg')
plt.switch_backend('Agg')
from PIL import Image
from shapely import Polygon
#import ipython
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton, QFileDialog, QLineEdit, QFormLayout, QLabel
from PySide6.QtGui import QDoubleValidator
from imageio import imread as imreader
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import threading
import multiprocessing as multi
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph
import pyqtgraph.opengl as gl
from pyqtgraph import functions as funcs
from pyqtgraph.opengl import GLScatterPlotItem as glscatter
from vispy import scene, io, app
app.use_app('PySide6')
import scipy
from scipy import ndimage
from scipy.spatial import ConvexHull
import skimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshlib
from meshlib import mrmeshnumpy as mrnumpy, mrmeshpy as mrpy
#import /simu/simu_011325_v3.py as simu

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
            ax = fig.add_subplot()
            fig.canvas.draw()
            colors = ['black','red','blue','magenta','green','orange']
            for b in coords:
                m = Polygon(b)
                t = np.array(m.exterior.xy)
                for i in np.arange(0,t.shape[1]):
                    if bound1[0][0] < t[0][i] < bound1[1][0] and bound1[0][1] < t[1][i] < bound1[1][1]:
                        adjust = t - np.tile(np.array([[bound[0][0],bound[0][1]]]).transpose(),(1,t.shape[1]))
                        scaled = adjust//np.array([[1,1]]).transpose()
                        cor = poly(list(zip(scaled[0],scaled[1])),edgecolor = 'r',alpha = 0)
                        print(cor)
                        #plt.fill(*cor.exterior.xy, color = 'none')#color = colors[count])
                        ax.add_patch(cor)
                        break
            count +=1
            col = fig.canvas.buffer_rgba()
            mat = np.asarray(col)
            print(col)
            dim = fig.canvas.get_width_height()[::-1]
            mat = np.frombuffer(col,dtype = np.uint8).reshape(dim + (4,))
            matlist.append(mat)#[:,:,0])
            print(mat.shape)
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
            boolmat = (a[:,:,0] != 1).astype(int)
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
        outline = ()
        count = 0
        for b in layers:
            d = ndimage.binary_fill_holes(b).astype(int)
            samp = np.broadcast_to(d,(5,d.shape[0],d.shape[1]))
            sampout = np.broadcast_to(b,(5,b.shape[0],b.shape[1]))
            samplist += (samp,)
            outline += (sampout,)
            print(count)
            count += 1
        sample = np.concatenate(samplist,axis = 0)
        sampleout = np.concatenate(outline,axis = 0)
        return sample, sampleout
    
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
        widgraph = pyqtgraph.PlotWidget()
        widpyqt = gl.GLViewWidget()
        widpyqt.show()
        self.sample = (1-sample).astype(np.float32)
        x,y,z = self.sample.shape
        
        colmat = np.zeros((self.sample.shape+(4,)),dtype = np.uint8)
        colmat[:,:,:,0] = 0
        colmat[:,:,:,1] = 0
        colmat[:,:,:,2] = 255*self.sample/1
        colmat[:,:,:,3] = 50
        #vert,face,norm,other = skimage.measure.marching_cubes(self.sample)
        """ widgl = gl.GLVolumeItem(colmat)
        widpyqt.addItem(widgl)
        viscan = scene.canvas.SceneCanvas('Voxel')
        gridview = viscan.central_widget.add_grid()
        viewer = gridview.add_view() """
        self.canvas = graphsample()
        a,b,c = np.nonzero(self.sample)
        ones = np.asarray(list(zip(a,b,c)))
        
        vertshull = ConvexHull(np.asarray(list(zip(a,b,c))))
        randind = random.sample(range(len(a)),int(len(a)/1000))
        anew,bnew,cnew = a[[randind]],b[[randind]],c[[randind]]
        xdim = np.arange(0,x)
        ydim = np.arange(0,y)
        zdim = np.arange(0,z)
        print((anew))
        #vol = scene.Volume(self.sample, parent = viewer.scene)
        """ vol = mrnumpy.simpleVolumeFrom3Darray(self.sample)
        vol2 = mrpy.simpleVolumeToDenseGrid(vol)
        iso = mrpy.gridToMesh(vol2)
        mesh = Poly3DCollection(iso)
        self.canvas.add_collection3d(mesh) """
        self.setWindowTitle('Graph Sample')
        """ for vert in vertshull.simplices:
         """    #self.canvas.ax1.plot3D(ones[vert,0],ones[vert,1],ones[vert,2])
        self.canvas.ax1.plot(xs = anew,ys = bnew,zs = cnew,s = 2)
        #self.canvas.view_init(elev = 90, azim = 0)
        layform.addWidget(self.canvas)
        self.setLayout(layform)

    

class MainWindow(QMainWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.threads = multi.Pool(processes=3)

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
                self.sample, self.outline = GDS.loadsample(matlist)
                if self.sample.all() != None:
                    self.win = graphwin(self.samplereturn())
                    self.win.show()
            
        
    def samplereturn(self):
        return self.outline
        
    
        
        

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()

#Created: December 16, 2024 - new version
#Use for tutorial only