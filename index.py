from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer

import pyqtgraph as pg
import numpy as np
import pandas as pd
import sys
from pathlib import Path
# import res_rc # Import the resource module

from PyQt5.uic import loadUiType
import urllib.request

ui, _ = loadUiType('main.ui')


class MainApp(QMainWindow, ui):

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1450, 900)

    # def list_signals_to_plot2(self):
    #     checked_signals_indices = []
    #     for i in range(self.list_widget2.count()):
    #         item = self.list_widget2.item(i)
    #         if item.checkState() == Qt.Checked:
    #             checked_signals_indices.append(self.list_widget2.row(item))
    #     return checked_signals_indices

    # list_item = QListWidgetItem(f"{self.signal_viewer2.signals[i].title.toPlainText()}")
    # list_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
    # list_item.setCheckState(Qt.Checked)
    # self.list_widget2.addItem(list_item)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(Path('qss/darkStyle.qss').read_text())
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
