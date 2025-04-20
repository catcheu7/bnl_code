import sys, gdspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import Polygon as poly
from scipy import ndimage
from tkinter import Tk, Label, Button, Entry, filedialog, Frame, messagebox, Toplevel


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
            plt.savefig(filename, dpi=1, bbox_inches='tight', pad_inches=0.1)
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


class GraphSample(FigureCanvasTkAgg):
    def __init__(self, sample, master=None):
        """
        Displays a 3D voxel graph of the sample.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(sample, facecolors='blue', edgecolors='gray')
        super().__init__(fig, master)


class GraphWin(Toplevel):
    def __init__(self, sample, master=None):
        """
        Creates a new window to display the 3D voxel graph.
        """
        super().__init__(master)
        self.title("3D Voxel Graph")
        self.geometry("800x600")

        # Add the 3D voxel graph
        canvas = GraphSample(sample, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("GDS Viewer")

        # Widgets
        self.load_button = Button(root, text="Load GDS File", command=self.load_gds_file)
        self.load_button.pack()

        self.file_label = Label(root, text="No file selected")
        self.file_label.pack()

        self.x1_label = Label(root, text="X1:")
        self.x1_label.pack()
        self.x1_entry = Entry(root)
        self.x1_entry.pack()

        self.x2_label = Label(root, text="X2:")
        self.x2_label.pack()
        self.x2_entry = Entry(root)
        self.x2_entry.pack()

        self.y1_label = Label(root, text="Y1:")
        self.y1_label.pack()
        self.y1_entry = Entry(root)
        self.y1_entry.pack()

        self.y2_label = Label(root, text="Y2:")
        self.y2_label.pack()
        self.y2_entry = Entry(root)
        self.y2_entry.pack()

        self.plot_button = Button(root, text="Plot Graph", command=self.plot_graph)
        self.plot_button.pack()

    def load_gds_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("GDS Files", "*.gds")])
        if file_path:
            self.file_label.config(text=file_path)
        else:
            self.file_label.config(text="No file selected")

    def plot_graph(self):
        file_path = self.file_label.cget("text")
        if file_path == "No file selected":
            messagebox.showerror("Error", "No file selected!")
            return

        try:
            x1 = float(self.x1_entry.get())
            x2 = float(self.x2_entry.get())
            y1 = float(self.y1_entry.get())
            y2 = float(self.y2_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input for bounds!")
            return

        bound, diff, polys = GDS.loadgds(file_path)
        bound1 = [(x1, y1), (x2, y2)]
        diff1 = [(x2 - x1), (y2 - y1)]

        # Call the layered function to generate and save the layers
        filenames = GDS.layered(diff, bound, bound1, polys)

        # Pass the filenames to loadsample and get the sample
        sample, _ = GDS.loadsample(filenames)

        # Open the 3D voxel graph in a new window
        GraphWin(sample, master=self.root)


if __name__ == "__main__":
    root = Tk()
    app = MainWindow(root)
    root.mainloop()