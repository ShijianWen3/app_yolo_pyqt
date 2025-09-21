from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5 import uic
from videowidget import VideoWidget

# 使用pathlib加载UI文件
ui_file = Path(__file__).parent / '..'/'mainwindow.ui'  # 修正了文件名
# 解析为绝对路径（自动处理..等相对路径符号）
ui_file = ui_file.resolve()
if ui_file.exists():
    with open('./src/ui_mainwindow.py', 'w', encoding='utf-8') as f:
            uic.compileUi(str(ui_file), f)
    from ui_mainwindow import Ui_MainWindow  # 导入生成的UI类
else:
    print(f"UI文件不存在: {ui_file}")



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

        
        self.setupUi(self)#由于是转换成py源码进行类继承, 所以要调用setupUi方法
        # self.widget_videoFront = self.findChild(VideoWidget, 'widget_videoFront')
        self.widget_videoFront.setAttribute(Qt.WA_StyledBackground, True)

        self.signalconnect()


    def signalconnect(self):
        self.pushButton_chooseFile_model.clicked.connect(self.chooseFile_model)
        self.pushButton_startPlay.clicked.connect(self.startPlay)
    def chooseFile_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "PyTorcht权重文件 (*.pt)")
        
        # 如果选择了文件，打印文件路径
        if file_path:
            print(f"选中的文件路径: {file_path}")
        else:
            print("没有选择文件")
    def startPlay(self):
        self.widget_videoFront.start(1)  # Update every 30 ms for ~33 fps

        