import PySide6
from PySde6.QWidgets import QApplication, QWidget

import sys

app = QApplication(sys.argv)
window = QWidget()
window.show()

app.exec()