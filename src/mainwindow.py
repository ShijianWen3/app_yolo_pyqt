from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import uic
import sys
import os
from reason import YOLOThread
from synthesis import Trajectory3DVisualizerThread, Auto3DCanvas
def compile_qrc_file():
    import subprocess
    """编译qrc文件为Python模块"""
    qrc_file = Path(__file__).parent /'..'/'qrc'/'qrc.qrc'  # qrc文件路径
    py_file = Path(__file__).parent /'..'/'src'/'qrc_rc.py'  # 输出的Python文件
    
    if qrc_file.exists():
        try:
            # 使用pyrcc5编译qrc文件
            result = subprocess.run([
                'pyrcc5', '-o', str(py_file), str(qrc_file)
            ], check=True, capture_output=True, text=True)
            print(f"QRC文件编译成功: {py_file}")
        except subprocess.CalledProcessError as e:
            print(f"编译QRC文件失败: {e}")
        except FileNotFoundError:
            print("pyrcc5未找到，请确保PyQt5正确安装")
    else:
        print(f"QRC文件不存在: {qrc_file}")

def compile_qrc_file_pyqt5():
    from PyQt5.pyrcc_main import main as pyrcc_main
    """编译qrc文件为Python模块"""
    qrc_file = Path(__file__).parent /'..'/'qrc'/'qrc.qrc'  # qrc文件路径
    py_file = Path(__file__).parent /'..'/'src'/'qrc_rc.py'  # 输出的Python文件
    if qrc_file.exists():
        original_argv = sys.argv
        sys.argv = ['pyrcc5', '-o', str(py_file), str(qrc_file)]
        try:
            pyrcc_main()
            print("QRC编译成功")
        except Exception as e:
            print(f"编译失败: {e}")
        finally:
            sys.argv = original_argv


# 在导入UI之前编译QRC文件
# compile_qrc_file()
compile_qrc_file_pyqt5()


# 确保导入路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from videowidget import VideoWidget

# 使用pathlib加载UI文件
ui_file = Path(__file__).parent / '..' / 'mainwindow.ui'  # 修正了文件名
# 解析为绝对路径（自动处理..等相对路径符号）
ui_file = ui_file.resolve()

if ui_file.exists():
    with open('./src/ui_mainwindow.py', 'w', encoding='utf-8') as f:
        uic.compileUi(str(ui_file), f)
    from ui_mainwindow import Ui_MainWindow  # 导入生成的UI类
else:
    print(f"UI文件不存在: {ui_file}")
    # 如果UI文件不存在，直接使用已有的UI类
    from ui_mainwindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)  # 由于是转换成py源码进行类继承, 所以要调用setupUi方法
           
        # self.widget_videoFront.setAttribute(Qt.WA_StyledBackground, True)
        # self.widget_videoFront.setAutoFillBackground(True)

        self.signalconnect()
        self.canvas_3d = Auto3DCanvas(self.widget_3Dplot)
        self.path_model = ''
        self.inference_thread_front = None  # 推理线程初始化为None
        self.inference_thread_side = None  # 推理线程初始化为None
        self.synthesis_thread = None  # 3D可视化线程初始化为None

        self.isSynthesising = False
        self.detFront = None
        self.detSide = None

    def signalconnect(self):
        self.pushButton_chooseFile_model.clicked.connect(self.chooseFile_model)
        self.pushButton_startPlay.clicked.connect(self.startPlay)
        self.pushButton_pausePlay.clicked.connect(self.pausePlay)
        self.pushButton_chooseFile_videoFront.clicked.connect(self.chooseFile_video)
        self.pushButton_chooseFile_videoSide.clicked.connect(self.chooseFile_video)
        self.pushButton_reason_start.clicked.connect(self.startReason)
        self.pushButton_reason_end.clicked.connect(self.EndReason)
        self.pushButton_synthesis.clicked.connect(self.synthesis)
        # 添加调试连接
        # print("Signal connections established")

    def chooseFile_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "PyTorch权重文件 (*.pt)")
        
        # 如果选择了文件，打印文件路径
        if file_path:
            # print(f"选中的文件路径: {file_path}")
            self.path_model = file_path
            self.pushButton_chooseFile_model.setText(os.path.basename(file_path))
        else:
            # print("没有选择文件")
            pass

    def startPlay(self):
        
        if self.comboBox_front.currentIndex() != 4:
            self.widget_videoFront.capIndex = self.comboBox_front.currentIndex()
        else:
            self.widget_videoFront.isCapFromFile = True
        if self.comboBox_side.currentIndex() != 4:
            self.widget_videoSide.capIndex = self.comboBox_side.currentIndex()
        else:
            self.widget_videoSide.isCapFromFile = True
        self.widget_videoFront.fps = self.spinBox_fpsFront.value()
        self.widget_videoSide.fps = self.spinBox_fpsSide.value()
        self.widget_videoSide.start()  # 尝试打开摄像头
        self.widget_videoFront.start()  # 尝试打开摄像头
        self.pushButton_reason_start.setEnabled(True)
        self.pushButton_reason_end.setEnabled(True)

    def chooseFile_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)")
        # 如果选择了文件，打印文件路径
        pushButton = self.sender()
        if file_path:
            print(f"选中的文件路径: {file_path}")
            if pushButton == self.pushButton_chooseFile_videoSide:
                self.widget_videoSide.capPath = file_path
            elif pushButton == self.pushButton_chooseFile_videoFront:
                self.widget_videoFront.capPath = file_path
            file_path = os.path.basename(file_path)
            pushButton.setText(file_path)#可以直接进行pushbotton方法的调用
        else:
            # print("没有选择文件")\
            pass
    
    def pausePlay(self):
        # print("pausePlay called")
        # if hasattr(self, 'widget_videoFront') and self.widget_videoFront:
        #     self.widget_videoFront.stop()
        #     print("Video playback stopped")
        self.widget_videoFront.stop()
        self.widget_videoSide.stop()

        self.EndReason()

        self.pushButton_reason_start.setEnabled(False)
        self.pushButton_reason_end.setEnabled(False)
    def startReason(self):
        
        try:
            # 创建推理线程，传入追踪器和VideoWidget的队列
            if self.path_model == '':
                reply = QMessageBox.warning(self, '警告', '请先选择模型文件!', QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)

                # 根据用户的选择执行操作
                if reply == QMessageBox.Ok:
                    # print("用户点击了 OK")
                    self.chooseFile_model()
                else:
                    # print("用户点击了 Cancel")
                    pass
                return
            self.inference_thread_front = YOLOThread(
                path_model=self.path_model,
                queue_frame=self.widget_videoFront.picQueue
            )

            self.inference_thread_side = YOLOThread(
                path_model=self.path_model,
                queue_frame=self.widget_videoSide.picQueue
            )
            
            # 连接信号
            self.inference_thread_front.inference_result.connect(self.on_inference_result_front)
            self.inference_thread_front.inference_stats.connect(self.on_inference_stats_front)
            self.inference_thread_front.error_occurred.connect(self.on_inference_error_front)

            self.inference_thread_side.inference_result.connect(self.on_inference_result_side)
            self.inference_thread_side.inference_stats.connect(self.on_inference_stats_side)
            self.inference_thread_side.error_occurred.connect(self.on_inference_error_side)
            
            # 设置推理参数
            self.inference_thread_front.set_inference_params(
                fps_limit=30,
                enable_skip_frames=True,
                max_queue_size=5
            )

            self.inference_thread_side.set_inference_params(
                fps_limit=30,
                enable_skip_frames=True,
                max_queue_size=5
            )
            
            print("✓ 推理线程初始化完成")
        except Exception as e:
            print(f"❌ 推理线程初始化失败: {e}")
            
        self.widget_videoFront.isReason = True
        self.widget_videoSide.isReason = True
        self.inference_thread_front.start_inference()
        self.inference_thread_side.start_inference()

    def EndReason(self):
        if self.inference_thread_front is not None:
            self.inference_thread_front.stop_inference()
        if self.inference_thread_side is not None:
            self.inference_thread_side.stop_inference()
        self.widget_videoFront.isReason = False
        self.widget_videoSide.isReason = False
    

    def on_inference_result_front(self, processed_frame, detections):
        """处理推理结果"""
        try:
            # 更新推理结果显示
            self.widget_videoFront.current_frame = processed_frame
            self.widget_videoFront.update()            
            # 更新检测结果显示
            self.detFront = detections
            self.load_det()
            
        except Exception as e:
            print(f"处理推理结果时出错: {e}")
    def on_inference_stats_front(self, stats):
        print(f"推理统计: {stats}")
        # 这里可以更新UI或进行其他处理
    def on_inference_error_front(self, error_message):
        print(f"推理错误: {error_message}")
        # 这里可以更新UI或进行其他处理

    def on_inference_result_side(self, processed_frame, detections):
        """处理推理结果"""
        try:
            # 更新推理结果显示
            self.widget_videoSide.current_frame = processed_frame
            self.widget_videoSide.update()            
            # 更新检测结果显示
            self.detSide = detections
            self.load_det()            
        except Exception as e:
            print(f"处理推理结果时出错: {e}")

    def on_inference_stats_side(self, stats):
        print(f"推理统计: {stats}")
        # 这里可以更新UI或进行其他处理
    def on_inference_error_side(self, error_message):
        print(f"推理错误: {error_message}")
        # 这里可以更新UI或进行其他处理

    def synthesis(self):
        if not self.isSynthesising:
            try:
                if self.synthesis_thread is None:
                    self.isSynthesising = True
                    # 创建3D可视化线程
                    self.synthesis_thread = Trajectory3DVisualizerThread()
                    self.synthesis_thread.update_plot_signal.connect(self.canvas_3d.update_plot)
                    self.synthesis_thread.start_tracking()

                    self.pushButton_synthesis.setStyleSheet("QPushButton { border: 2px solid black;font: 75 14pt 'Arial';border-radius: 8px; /* 设置圆角 */background-color: rgb(0, 255, 0);}")

                    print("✓ 3D可视化线程初始化完成")
                else:
                    print("3D可视化线程已在运行")
            except Exception as e:
                print(f"❌ 3D可视化线程初始化失败: {e}")
        else:
            self.isSynthesising = False
            if self.synthesis_thread is not None:
                self.synthesis_thread.stop()
                self.synthesis_thread = None
                self.pushButton_synthesis.setStyleSheet("QPushButton { border: 2px solid black;font: 75 14pt 'Arial';border-radius: 8px; /* 设置圆角 */background-color: rgb(255, 0, 0);}")

            print("3D可视化线程已停止")

    def load_det(self):
        if self.synthesis_thread is not None and self.isSynthesising:
            if self.detFront is not None and self.detSide is not None:
                self.synthesis_thread.add_detection_results(self.detFront, self.detSide)
                self.detFront = None
                self.detSide = None
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyleSheet("""
        QWidget {
            font-family: Arial;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    print("Application started")
    sys.exit(app.exec_())