from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from time import sleep
# import res_rc # Import the resource module

from PyQt5.uic import loadUiType
import urllib.request

from Security_Voice_code_Access import Sound, Model

ui, _ = loadUiType('main.ui')


def plot_spectrogram(widget, sxx):
    # Plot Spectrogram
    img = pg.ImageItem()
    img.setImage(np.rot90(sxx))
    widget.addItem(img)
    widget.setYRange(0, 5 * np.log10(np.max(sxx)))

    # Set colormap
    colormap = pg.colormap.get('viridis')
    img.setColorMap(colormap)


def get_checked(lst, result):
    checked_indices = []
    for j in range(lst.count()):
        item = lst.item(j)
        if item.checkState() == Qt.Checked:
            checked_indices.append(lst.row(item))

    print("checked_indices:", checked_indices)
    if result in checked_indices:
        return True

    return False


def create_table(table, header_titles, num_rows, num_cols):
    table.setColumnCount(num_cols)
    table.setHorizontalHeaderLabels(header_titles)
    table.setColumnWidth(0, 200)
    table.setColumnWidth(1, 200)
    table.setRowCount(num_rows)


def fill_stats_tables(dict_map, probs_values, stats_table):
    print(probs_values)
    for row, (person, percent) in enumerate(zip(list(dict_map.values()), probs_values)):
        item1 = QTableWidgetItem(str(person))
        item2 = QTableWidgetItem(str(np.around(percent * 100, 2)) + " %")

        stats_table.setItem(row, 0, item1)
        stats_table.setItem(row, 1, item2)


def fill_list_widget_from_dict(dictionary, lst):
    for i in range(len(dictionary)):
        list_item = QListWidgetItem(f"{dictionary[i]}")
        list_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        list_item.setCheckState(Qt.Unchecked)
        lst.addItem(list_item)


class MainApp(QMainWindow, ui):

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1450, 900)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.graphics_view_layout2 = QHBoxLayout(self.spectrogram_graphics_view)
        self.graphics_view_layout2.addWidget(self.canvas)
        self.spectrogram_graphics_view.setLayout(self.graphics_view_layout2)

        self.sound = Sound()
        self.word_model = Model("w_model_svc_z.pkl")
        self.word_model.p_w_state = 'w'

        self.person_model = Model("p_model_svc.pkl")
        self.person_model.p_w_state = 'p'

        self.p_accessed = False
        self.w_accessed = False

        self.persons_map = {0: 'Abdallah', 1: 'Alaa',
                            2: 'Ali', 3: 'Amgad',
                            4: "Omar", 5: 'Mahmoud',
                            6: 'Marwan', 7: 'Kamal',
                            8: 'Shawky'}

        self.w_probs_values = []
        self.p_probs_values = []
        self.w_result = ""
        self.p_result = ""

        self.access_persons = []
        self.access_words = []

        self.words_map = {
            0: "Open the door",
            1: "Grant me access",
            2: "Unlock middle gate"
        }

        fill_list_widget_from_dict(self.persons_map, self.persons_list)
        fill_list_widget_from_dict(self.words_map, self.words_list)

        self.record_btn.clicked.connect(self.record_sound)

        create_table(self.persons_stats, ('Name', 'Percentage'), len(self.persons_map), 2)
        create_table(self.words_stats, ('Password', 'Percentage'), len(self.words_map), 2)

    def record_sound(self):
        self.result_label.setText("Recording...")
        QCoreApplication.processEvents()  # Force GUI update

        self.sound.record()

        self.result_label.setText("Loading...")
        QCoreApplication.processEvents()  # Force GUI update

        self.sound.extract_features()

        self.plot_spectro_plt(self.sound.spectro)

        self.predict_result()
        print("Result: ", self.p_result, ", ", self.w_result)
        # ------------------------
        self.p_accessed = get_checked(self.persons_list, self.p_result)
        self.w_accessed = get_checked(self.words_list, self.w_result)

        if self.p_accessed and self.w_accessed:
            self.result_label.setText("The Door is opened")
        else:
            self.result_label.setText("The Door is closed")

        # ------------------------
        fill_stats_tables(self.persons_map, self.p_probs_values, self.persons_stats)
        fill_stats_tables(self.words_map, self.w_probs_values, self.words_stats)

    def predict_result(self):
        self.word_model.sound = self.sound
        self.person_model.sound = self.sound
        self.p_result, self.p_probs_values = self.person_model.make_prediction()
        print("p: ", self.p_result)
        self.w_result, self.w_probs_values = self.word_model.make_prediction()
        print("w: ", self.w_result)

    def plot_spectro_plt(self, spectro):
        # Clear previous plot and plot the new spectrogram
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Use pcolormesh to plot the spectrogram
        ax.pcolormesh(spectro[1], spectro[0], 10 * np.log10(np.abs(spectro[2])), shading='auto', cmap='magma')

        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Spectrogram', fontsize=20)

        # Add colorbar to the plot
        cbar = self.figure.colorbar(
            ax.pcolormesh(spectro[1], spectro[0], 10 * np.log10(np.abs(spectro[2])), shading='auto', cmap='magma'),
            ax=ax, format='%+2.0f dB')
        cbar.set_label('Intensity (dB)')

        # Draw the plot
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(Path('qss/darkStyle.qss').read_text())
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
