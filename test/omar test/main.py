from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit
from recorder import record_audio


class VoiceRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.record_button = QPushButton("Record Voice")
        self.record_button.clicked.connect(self.record_voice)
        self.layout.addWidget(self.record_button)

        self.filename_label = QLabel("Filename:")
        self.filename_lineedit = QLineEdit("recording.wav")
        self.layout.addWidget(self.filename_label)
        self.layout.addWidget(self.filename_lineedit)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def record_voice(self):
        filename = self.filename_lineedit.text()
        try:
            record_audio(filename=filename, duration=3)
            self.status_label.setText(f"Voice recorded and saved as '{filename}'")
        except Exception as e:
            self.status_label.setText(f"Error recording: {e}")


if __name__ == "__main__":
    app = QApplication([])
    window = VoiceRecorder()
    window.show()
    app.exec_()
