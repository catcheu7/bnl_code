import sys
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.windowTitle("testing window")
        wid = QWidget()

        gdsbutton = QPushButton("Load Design")
        gdsbutton.clicked.connect(self.button_clicked)
        self.setCentralWidget(gdsbutton)

    def button_clicked(self,s):
        print("clicked")
        lwin = QDialog(self)
        lwin.setWindowTitle("Loading")
        lwin.exec()

app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()

#Created: December 16, 2024 - new version
#Use for tutorial only