import sys
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
        bound = np.array([[0, 0], [100, 100]])
        diff = bound[:, None] - bound[None, :]
        polys = {1: [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]}
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
        count = 0
        filenames = []
        matlist = []
        xsize = 1
        ysize = 1

        for a, coords in polys.items():
            figa = plt.figure(figsize=(diff[1][0][0] / 100, diff[1][0][1] / 100), frameon=True)
            plt.axis('on')
            plt.xlim(bound1[0][0] - bound[0][0], bound1[1][0] - bound[0][0])
            plt.ylim(bound1[0][1] - bound[0][1], bound1[1][1] - bound[0][1])

            ax = figa.add_subplot()
            for b in coords:
                m = poly(b)
                t = np.array(m.exterior.xy)
                adjust = t - np.tile(np.array([[bound[0][0], bound[0][1]]]).transpose(), (1, t.shape[1]))
                scaled = np.round(adjust / np.array([[xsize, ysize]]).transpose())

                cor = poly(list(zip(scaled[0], scaled[1])))
                if cor.is_valid:
                    x, y = cor.exterior.xy
                    plt.fill(x, y, color='red')  # Plot the polygon
                else:
                    print("Invalid Polygon:", cor)

            count += 1
            filenames.append(f"layer{count}.png")
            matlist.append(f"layer{count}.png")

        return filenames, matlist

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

        fig = GDS.graphingbound(diff1, bound1, bound)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = Tk()
    app = MainWindow(root)
    root.mainloop()