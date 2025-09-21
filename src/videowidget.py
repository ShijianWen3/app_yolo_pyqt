import sys
import cv2
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget, QApplication

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        
        # OpenCV VideoCapture
        self.cap = None

        # Timer to update the widget and refresh the video frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)

        # Do not auto-start the timer here. The MainWindow will control
        # when playback starts/stops by calling start()/stop().

        # Try opening default camera (index 0) lazily when start() is called.

    def paintEvent(self, event):
        # Read a frame from the video
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage and scale to widget size while preserving aspect
        height, width, channels = frame.shape
        bytes_per_line = 3 * width
        qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Scale the image to the widget's size
        target = self.size()
        qimage_scaled = qimage.scaled(QSize(target.width(), target.height()), Qt.KeepAspectRatio)

        # Paint the frame centered
        painter = QPainter(self)
        x = (self.width() - qimage_scaled.width()) // 2
        y = (self.height() - qimage_scaled.height()) // 2
        painter.drawImage(x, y, qimage_scaled)
        painter.end()

    def closeEvent(self, event):
        # Release the video capture when the window is closed
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        event.accept()

    def open_camera(self, index=0):
        """Open a camera or video file. index can be camera index (int) or a file path."""
        # If a previous capture exists, release it
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        # Support int indices and string file paths
        try:
            self.cap = cv2.VideoCapture(index)
        except Exception as e:
            print('Failed to open capture:', e)
            self.cap = None

        if self.cap is None or not self.cap.isOpened():
            print(f'Warning: unable to open video source: {index}')
            self.cap = None
            return False
        return True

    def start(self, index=0):
        """Start updating frames. If index is provided, try to open that source."""
        if self.cap is None:
            ok = self.open_camera(index)
            if not ok:
                return False
        self.timer.start(30)
        return True

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = VideoWidget()
    widget.resize(640, 480)
    widget.show()
    widget.start(1)  # Open default camera
    sys.exit(app.exec_())