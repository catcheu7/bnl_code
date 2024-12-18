import sys
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super.__init__

        gdsbutton = QPushButton("Load Design")
        gdsbutton.clicked.event

app = QApplication(sys.argv)
window = QWidget()
window.show()

app.exec()

#Created: December 16, 2024 - new version
#Use for tutorial only