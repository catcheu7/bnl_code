import sys, gdspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from shapely.geometry import Polygon as poly
from scipy import ndimage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QFileDialog, QVBoxLayout, QWidget, QDialog
)
from PySide6.QtCore import Qt


class GDS:
    @staticmethod
    def loadgds(file_path):
        # Simulate loading GDS data
        test = gdspy.GdsLibrary(infile=file_path)
        cell = test.top_level()[0]
        polys = cell.get_polygons(by_spec=True)
        bound = cell.get_bounding_box()
        diff = bound[:, None] - bound[None, :]
        return bound, diff, polys

    @staticmethod
    def layered(diff, bound, bound1, polys):
        """
        Generates 2D cross-sections for each layer, saves them as images, and returns the filenames.
        """
        count = 0
        filenames = []  # List to store filenames of saved images
        xsize = 1
        ysize = 1

        for layer_id, coords in polys.items():
            figa = plt.figure(figsize=(diff[1][0][0] / 100, diff[1][0][1] / 100), frameon=True)
            ax = figa.add_subplot()
            ax.set_xlim(bound1[0][0] - bound[0][0], bound1[1][0] - bound[0][0])
            ax.set_ylim(bound1[0][1] - bound[0][1], bound1[1][1] - bound[0][1])
            plt.autoscale(False)

            for polygon_coords in coords:
                polygon = poly(polygon_coords)
                t = np.array(polygon.exterior.xy)
                adjust = t - np.tile(np.array([[bound[0][0], bound[0][1]]]).transpose(), (1, t.shape[1]))
                scaled = np.round(adjust / np.array([[xsize, ysize]]).transpose())
                scaled_polygon = poly(list(zip(scaled[0], scaled[1])))
                x, y = scaled_polygon.exterior.xy
                plt.fill(x, y, color='red')

            count += 1
            filename = f"layer_{count}.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
            filenames.append(filename)
            plt.close(figa)

        return filenames

    @staticmethod
    def loadsample(layers):
        """
        Processes the layers into a sample and its outline.
        """
        samplist = ()
        outline = ()
        matlist = []
        for filename in layers:
            matimg = plt.imread(filename)[:, :, 0]  # Assuming grayscale image
            boolmat = (matimg != 1).astype(int)
            matlist.append(boolmat)
        for b in matlist:
            d = ndimage.binary_fill_holes(b).astype(int)
            samp = np.broadcast_to(d, (5, d.shape[0], d.shape[1]))
            sampout = np.broadcast_to(b, (5, b.shape[0], b.shape[1]))
            samplist += (samp,)
            outline += (sampout,)
        sample = np.concatenate(samplist, axis=0)
        sampleout = np.concatenate(outline, axis=0)
        return sample, sampleout


class GraphSample(FigureCanvas):
    def __init__(self, sample, parent=None):
        """
        Displays a 3D voxel graph of the sample.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(sample, facecolors='blue', edgecolors='gray')
        super().__init__(fig)


class GraphWin(QDialog):
    def __init__(self, sample, parent=None):
        """
        Creates a new window to display the 3D voxel graph.
        """
        super().__init__(parent)
        self.setWindowTitle("3D Voxel Graph")
        self.setGeometry(100, 100, 800, 600)

        # Add the 3D voxel graph
        layout = QVBoxLayout()
        canvas = GraphSample(sample, parent=self)
        layout.addWidget(canvas)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDS Viewer")
        self.setGeometry(100, 100, 400, 300)

        # Widgets
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.load_button = QPushButton("Load GDS File")
        self.load_button.clicked.connect(self.load_gds_file)
        self.layout.addWidget(self.load_button)

        self.file_label = QLabel("No file selected")
        self.layout.addWidget(self.file_label)

        self.x1_label = QLabel("X1:")
        self.layout.addWidget(self.x1_label)
        self.x1_entry = QLineEdit()
        self.layout.addWidget(self.x1_entry)

        self.x2_label = QLabel("X2:")
        self.layout.addWidget(self.x2_label)
        self.x2_entry = QLineEdit()
        self.layout.addWidget(self.x2_entry)

        self.y1_label = QLabel("Y1:")
        self.layout.addWidget(self.y1_label)
        self.y1_entry = QLineEdit()
        self.layout.addWidget(self.y1_entry)

        self.y2_label = QLabel("Y2:")
        self.layout.addWidget(self.y2_label)
        self.y2_entry = QLineEdit()
        self.layout.addWidget(self.y2_entry)

        self.plot_button = QPushButton("Plot Graph")
        self.plot_button.clicked.connect(self.plot_graph)
        self.layout.addWidget(self.plot_button)

    def load_gds_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open GDS File", "", "GDS Files (*.gds)")
        if file_path:
            self.file_label.setText(file_path)
        else:
            self.file_label.setText("No file selected")

    def plot_graph(self):
        file_path = self.file_label.text()
        if file_path == "No file selected":
            self.show_error("No file selected!")
            return

        try:
            x1 = float(self.x1_entry.text())
            x2 = float(self.x2_entry.text())
            y1 = float(self.y1_entry.text())
            y2 = float(self.y2_entry.text())
        except ValueError:
            self.show_error("Invalid input for bounds!")
            return

        bound, diff, polys = GDS.loadgds(file_path)
        bound1 = [(x1, y1), (x2, y2)]
        diff1 = [(x2 - x1), (y2 - y1)]

        # Call the layered function to generate and save the layers
        filenames = GDS.layered(diff, bound, bound1, polys)

        # Pass the filenames to loadsample and get the sample
        sample, _ = GDS.loadsample(filenames)

        # Open the 3D voxel graph in a new window
        graph_win = GraphWin(sample, parent=self)
        graph_win.exec()

    def show_error(self, message):
        error_dialog = QDialog(self)
        error_dialog.setWindowTitle("Error")
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        error_dialog.setLayout(layout)
        error_dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())