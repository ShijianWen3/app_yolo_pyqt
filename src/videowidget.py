import sys
import cv2
import queue
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPainter, QPaintEvent
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox
MAX_QUEUE_SIZE = 5
class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  
        # 使样式表中的背景颜色和边框生效
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        # OpenCV VideoCapture
        self.capIndex = -1
        self.capPath = ""
        self.isCapFromFile = False
        self.cap = None
        self.current_frame = None
        # Timer to update the widget and refresh the video frame
        self.fps = 60
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loadframe)
        # yolo model queue
        self.picQueue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.isReason = False
    def paintEvent(self, event: QPaintEvent):
        
        """绘制事件"""
        painter = QPainter(self)
        
        try:
            # 首先绘制背景（让样式表生效）
            super().paintEvent(event)
            
            # 计算缩放尺寸，保持宽高比
            widget_rect = self.rect()
            pad = 3  # 为边框留出空间
            available_rect = widget_rect.adjusted(pad, pad, -pad, -pad)

            # 如果没有当前帧，只绘制背景
            if self.current_frame is None:
                painter.fillRect(available_rect, Qt.black)
                painter.setPen(Qt.white)
                painter.drawText(self.rect(), Qt.AlignCenter, "No Video Signal")
                return

            # 转换帧格式
            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = frame.shape
            bytes_per_line = 3 * width
            qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            
            scaled_image = qimage.scaled(
                available_rect.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )

            # 计算居中位置
            x = available_rect.x() + (available_rect.width() - scaled_image.width()) // 2
            y = available_rect.y() + (available_rect.height() - scaled_image.height()) // 2

            # 绘制图像
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(x, y, scaled_image)

        except Exception as e:
            print(f"Paint error: {e}")
            painter.fillRect(self.rect(), Qt.red)
        finally:
            painter.end()

        

    def closeEvent(self, event):
        """窗口关闭事件"""
        print("VideoWidget closeEvent called")
        self.stop()
        event.accept()

    def open_camera(self, index=0):
        """打开摄像头或视频文件"""
        
        if self.isCapFromFile:
            index = self.capPath
            if index == '':
                reply = QMessageBox.warning(self, '警告', '请先选择视频文件!', QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)

                # 根据用户的选择执行操作
                if reply == QMessageBox.Ok:
                    print("用户点击了 OK")
                else:
                    print("用户点击了 Cancel")
                return
        else:
            index = self.capIndex
        # 如果已有capture，先释放
        if self.cap is not None:
            try:
                self.cap.release()
                print("Previous capture released")
            except Exception as e:
                print(f"Error releasing previous capture: {e}")

        # 打开新的capture
        try:
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)#避免缓冲延迟
            if self.cap is None or not self.cap.isOpened():
                self.cap = None
                return False
            # 设置一些基本属性
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
            # 获取并打印摄像头信息
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera opened successfully: {width}x{height} @ {fps} FPS")
            return True
        except Exception as e:
            print(f'Exception opening capture: {e}')
            self.cap = None
            return False
    def loadframe(self):
        
        """加载一帧图像"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.picQueue.full():
                    try:
                        self.picQueue.get_nowait()  # 丢弃最旧的一帧
                    except queue.Empty:
                        pass
                self.picQueue.put(frame)
                if self.isReason:
                    pass
                else:
                    self.current_frame = frame
                    self.update()  # 触发重绘
                return
        return

    def start(self):
        """开始视频播放"""
        index = self.capIndex
        print(f"VideoWidget start() called with index: {index}")
        
        # 如果没有capture或需要更换源，打开新的
        if self.cap is None:
            success = self.open_camera(index)
            if not success:
                print("Failed to open camera")
                return False
        
        # 开始定时器
        if not self.timer.isActive():
            self.timer.start(1000//self.fps)  # 约30 FPS
            print("Timer started")
        
        return True

    def stop(self):
        """停止视频播放"""
        print("VideoWidget stop() called")
        
        # 停止定时器
        if self.timer.isActive():
            self.timer.stop()
            print("Timer stopped")
        
        # 释放capture
        if self.cap is not None:
            try:
                self.cap.release()
                print("Capture released")
            except Exception as e:
                print(f"Error releasing capture: {e}")
            finally:
                self.cap = None
        
        # 清除当前帧
        self.current_frame = None
        self.update()  # 重绘以显示"No Video Signal"

    def __del__(self):
        """析构函数"""
        print("VideoWidget destructor called")
        self.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = VideoWidget()
    widget.resize(640, 480)
    widget.setStyleSheet("border: 3px dashed black;")
    widget.show()
    widget.start(0)  # 打开默认摄像头
    sys.exit(app.exec_())