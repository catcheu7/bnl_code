import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Plot")
        canvas = FigureCanvasQTAgg(plt.figure())
        ax = canvas.figure.add_subplot()
        ax.plot([0, 1, 2], [0, 1, 4])  # Simple plot
        self.setCentralWidget(canvas)

app = QApplication([])
window = TestWindow()
window.show()
app.exec_()