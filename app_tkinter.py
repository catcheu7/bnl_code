import sys,gdspy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import Polygon as poly
from scipy import ndimage
from tkinter import Tk, Label, Button, Entry, filedialog, Frame, messagebox


class GDS:
    @staticmethod
    def loadgds(file_path):
        # Simulate loading GDS data
        test = gdspy.GdsLibrary(infile = file_path)
        cell = test.top_level()[0]
        polys = cell.get_polygons(by_spec = True)
        bound = cell.get_bounding_box()
        diff = bound[:,None] - bound[None,:]
        return bound, diff, polys

    @staticmethod
    def graphingbound(diff1, bound1, bound):
        plt.ioff()
        fig = plt.figure(figsize=(diff1[0] / 100, diff1[1] / 100), frameon=False)
        ax = fig.add_subplot()
        plt.axis('on')
        plt.xlim(bound1[0][0] - bound[0][0], bound1[1][0] - bound[0][0])
        plt.ylim(bound1[0][1] - bound[0][1], bound1[1][1] - bound[0][1])
        return fig

    @staticmethod
    def layered(diff, bound, bound1, polys):
        """
        Generates 2D cross-sections for each layer, saves them as images, and returns the filenames.

        Args:
            diff (list): Difference in bounds for scaling.
            bound (list): Original bounding box coordinates.
            bound1 (list): Adjusted bounding box coordinates.
            polys (dict): Dictionary of polygons representing the layers.

        Returns:
            list: Filenames of the saved images.
        """
        count = 0
        filenames = []  # List to store filenames of saved images
        xsize = 1
        ysize = 1

        for layer_id, coords in polys.items():
            # Create a new Matplotlib figure for the layer
            figa = plt.figure(figsize=(diff[1][0][0] / 100, diff[1][0][1] / 100), frameon=True)
            plt.axis('on')
            plt.xlim(bound1[0][0] - bound[0][0], bound1[1][0] - bound[0][0])
            plt.ylim(bound1[0][1] - bound[0][1], bound1[1][1] - bound[0][1])

            ax = figa.add_subplot()
            for polygon_coords in coords:
                # Create a Shapely polygon and scale it
                polygon = poly(polygon_coords)
                t = np.array(polygon.exterior.xy)
                adjust = t - np.tile(np.array([[bound[0][0], bound[0][1]]]).transpose(), (1, t.shape[1]))
                scaled = np.round(adjust / np.array([[xsize, ysize]]).transpose())

                # Create a new polygon with scaled coordinates
                scaled_polygon = poly(list(zip(scaled[0], scaled[1])))
                if scaled_polygon.is_valid:
                    x, y = scaled_polygon.exterior.xy
                    plt.fill(x, y, color='red')  # Plot the polygon
                else:
                    print(f"Invalid Polygon in Layer {layer_id}: {scaled_polygon}")

            # Save the figure as an image
            count += 1
            filename = f"layer_{count}.png"
            plt.savefig(filename, dpi=100)
            filenames.append(filename)  # Add the filename to the list
            plt.close(figa)  # Close the figure to free memory

        return filenames

    @staticmethod
    def loadsample(layers):
        samplist = ()
        outline = ()
        for b in layers:
            d = ndimage.binary_fill_holes(b).astype(int)
            samp = np.broadcast_to(d, (5, d.shape[0], d.shape[1]))
            sampout = np.broadcast_to(b, (5, b.shape[0], b.shape[1]))
            samplist += (samp,)
            outline += (sampout,)
        sample = np.concatenate(samplist, axis=0)
        sampleout = np.concatenate(outline, axis=0)
        return sample, sampleout


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

        self.canvas_frame = Frame(root)
        self.canvas_frame.pack()

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

        # Display a success message with the saved filenames
        messagebox.showinfo("Success", f"Layers saved as:\n{', '.join(filenames)}")


if __name__ == "__main__":
    root = Tk()
    app = MainWindow(root)
    root.mainloop()