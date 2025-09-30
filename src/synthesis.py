from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class Trajectory3DVisualizerThread(QThread):
    """3D轨迹可视化线程 - 接收双摄像头推理结果"""
    update_plot_signal = pyqtSignal(dict)  # 发送轨迹数据信号
    
    def __init__(self):
        super().__init__()
        self.tracks_3d = {"red_ball": [], "green_ball": [], "blue_ball": []}
        self.tracking_enabled = False
        self.running = True
        self.mutex = QMutex()
        self.TRAJ_BUF_LEN = 200
        
        # 存储双摄像头的最新检测结果
        self.latest_det0 = []
        self.latest_det1 = []
        self.has_new_data = False
        
    def add_detection_results(self, det0, det1):
        """接收双摄像头的推理结果"""
        self.mutex.lock()
        self.latest_det0 = det0
        self.latest_det1 = det1
        self.has_new_data = True
        self.mutex.unlock()
    
    def merge_3d_tracks_frame(self, det0, det1):
        """合成三维轨迹点"""
        d3 = {}
        for cname in ["red_ball", "green_ball", "blue_ball"]:
            c0 = next((d for d in det0 if d['class_name'] == cname), None)
            c1 = next((d for d in det1 if d['class_name'] == cname), None)
            if c0 and c1:
                d3[cname] = (c0['center'][0], c0['center'][1], c1['center'][0])
        return d3
    
    def set_tracking_enabled(self, enabled):
        """设置追踪状态"""
        self.mutex.lock()
        self.tracking_enabled = enabled
        self.mutex.unlock()
    
    def clear_tracks(self):
        """清空轨迹"""
        self.mutex.lock()
        for cname in self.tracks_3d:
            self.tracks_3d[cname].clear()
        self.mutex.unlock()
    
    def run(self):
        """线程主循环"""
        while self.running:
            try:
                self.mutex.lock()
                if self.has_new_data and self.tracking_enabled:
                    det0_copy = self.latest_det0.copy()
                    det1_copy = self.latest_det1.copy()
                    self.has_new_data = False
                    self.mutex.unlock()
                    
                    d3 = self.merge_3d_tracks_frame(det0_copy, det1_copy)
                    
                    if d3:
                        self.mutex.lock()
                        for cname, pt in d3.items():
                            self.tracks_3d[cname].append(pt)
                            if len(self.tracks_3d[cname]) > self.TRAJ_BUF_LEN:
                                self.tracks_3d[cname].pop(0)
                        
                        tracks_copy = {k: list(v) for k, v in self.tracks_3d.items()}
                        self.mutex.unlock()
                        self.update_plot_signal.emit(tracks_copy)
                else:
                    self.mutex.unlock()
                
                self.msleep(20)
                
            except Exception as e:
                print(f"❌ 3D可视化线程错误: {e}")
                break
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait(1000)

    def start_tracking(self):
        """开始追踪"""
        self.set_tracking_enabled(True)
        self.start()


class Auto3DCanvas(FigureCanvas):
    """自适应320x320 QWidget的3D画布"""
    
    def __init__(self, parent_widget):
        # 创建figure，使用tight_layout自动调整
        self.fig = Figure(facecolor='white')
        super().__init__(self.fig)
        
        # 自动填充父Widget
        self.setParent(parent_widget)
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(8, 8, 8, 8)  # 在320x320中留8px padding
        layout.addWidget(self)
        
        # 创建3D子图
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 颜色映射
        self.color_map = {"red_ball": "r", "green_ball": "g", "blue_ball": "b"}
        
        # 使用tight_layout自动调整边距，在减去padding后的304x304区域中显示
        self.fig.tight_layout(pad=0.5)
        
        self.setup_axes()
        
    def setup_axes(self):
        """设置坐标轴样式"""
        self.ax.set_xlabel('X (cam0)', fontsize=8)
        self.ax.set_ylabel('Y (cam0)', fontsize=8) 
        self.ax.set_zlabel('X (cam1)', fontsize=8)
        self.ax.tick_params(labelsize=6)
        self.ax.set_xlim(0, 640)
        self.ax.set_ylim(0, 480)
        self.ax.set_zlim(0, 640)
        self.ax.set_title("3D Ball Trajectories", fontsize=9)
    
    def update_plot(self, tracks_3d):
        """更新3D轨迹图，自动缩放到最佳显示效果"""
        self.ax.clear()
        self.setup_axes()
        
        all_points = []
        plot_data = {}
        
        # 处理轨迹数据
        for cname, points in tracks_3d.items():
            if len(points) > 0:
                xs, ys, zs = zip(*points)
                all_points.extend(points)
                plot_data[cname] = (xs, ys, zs)
        
        # 绘制轨迹点
        for cname, (xs, ys, zs) in plot_data.items():
            color = self.color_map.get(cname, "k")
            n_points = len(xs)
            
            # 渐变效果
            alphas = np.linspace(0.3, 1.0, n_points)
            sizes = np.linspace(10, 30, n_points)
            
            for i in range(n_points):
                self.ax.scatter(xs[i], ys[i], zs[i], 
                              c=color, alpha=alphas[i], s=sizes[i],
                              label=cname if i == n_points-1 else "")
            
            # 连接线
            if n_points > 1:
                self.ax.plot(xs, ys, zs, color=color, alpha=0.4, linewidth=1)
        
        # 自动缩放 - 关键部分：让内容自适应304x304的显示区域
        if all_points:
            xs, ys, zs = zip(*all_points)
            
            # 计算数据范围
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys) 
            z_min, z_max = min(zs), max(zs)
            
            # 添加15%的数据padding，让轨迹在显示区域中居中且留有空白
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            padding_x = max(x_range * 0.15, 30)
            padding_y = max(y_range * 0.15, 30)
            padding_z = max(z_range * 0.15, 30)
            
            self.ax.set_xlim(x_min - padding_x, x_max + padding_x)
            self.ax.set_ylim(y_min - padding_y, y_max + padding_y)
            self.ax.set_zlim(z_min - padding_z, z_max + padding_z)
        
        # 图例
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), 
                         loc='upper right', fontsize=7)
        
        # 自动调整布局以适应320x320减去padding的区域
        self.fig.tight_layout(pad=0.3)
        self.draw()