import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5 import uic
from mainwindow import MainWindow

# 设置DPI策略
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


        
       
if __name__ == '__main__':
    app = QApplication(sys.argv)
 
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())