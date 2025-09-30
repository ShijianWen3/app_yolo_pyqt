import torch
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from PyQt5.QtCore import QThread, pyqtSignal
import queue

class RGBBallTracker:
    def __init__(self, model_path, confidence_threshold=0.55, device='auto'):
        """
        初始化RGB球追踪器
        
        Args:
            model_path: YOLO模型文件路径
            confidence_threshold: 置信度阈值
            device: 设备选择 ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)
        """
        # 自动选择最佳设备
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"✓ 自动选择GPU: {torch.cuda.get_device_name(0)}")
                print(f"✓ GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device = 'cpu'
                print("⚠ GPU不可用，使用CPU")
        else:
            if device.startswith('cuda') and not torch.cuda.is_available():
                print("⚠ 指定GPU不可用，切换到CPU")
                device = 'cpu'
            elif device.startswith('cuda'):
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                if gpu_id < torch.cuda.device_count():
                    print(f"✓ 使用指定GPU: {torch.cuda.get_device_name(gpu_id)}")
                else:
                    print(f"⚠ GPU {gpu_id} 不存在，使用GPU 0")
                    device = 'cuda:0'
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        print(f"📦 正在加载模型...")
        self.model = YOLO(model_path)
        
        # 将模型移动到指定设备
        if device != 'cpu':
            print(f"🔄 将模型移动到 {device}...")
            self.model.to(device)
        
        print(f"✓ 模型已加载到 {device}")
        
        # GPU优化设置
        if device.startswith('cuda'):
            self._setup_gpu_optimization()
        
        # 定义颜色映射 (BGR格式)
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0)
        }
        
        # 类别名称映射（根据你的模型训练时的类别顺序调整）
        self.class_names = {
            0: 'red_ball',
            1: 'green_ball', 
            2: 'blue_ball'
        }
        
        # 追踪历史记录
        self.tracking_history = defaultdict(list)
        self.frame_count = 0
        
        # 性能监控
        self.inference_times = []
        self.total_inference_time = 0
        
        # GPU预热
        if device.startswith('cuda'):
            self._warmup_model()
    
    def _setup_gpu_optimization(self):
        """设置GPU优化"""
        try:
            # 启用CUDNN基准模式（固定输入尺寸时有效）
            torch.backends.cudnn.benchmark = True
            print("✓ CUDNN基准模式已启用")
        except Exception as e:
            print(f"⚠ GPU优化设置失败: {e}")
    
    def _warmup_model(self, warmup_frames=5):
        """GPU模型预热"""
        print("🔥 GPU模型预热中...")
        dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        for i in range(warmup_frames):
            with torch.no_grad():
                _ = self.model(dummy_frame, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        print("✓ GPU预热完成")
        
    def process_frame(self, frame, isPrintInfo=True):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            isPrintInfo: 是否显示统计信息
            
        Returns:
            processed_frame: 处理后的图像帧
            detections: 检测结果
        """


        # 记录推理开始时间
        inference_start = time.perf_counter()
        
        # 运行YOLO推理 - GPU优化推理
        with torch.no_grad():  # 禁用梯度计算节省显存
            if self.device.startswith('cuda'):
                # GPU推理时使用混合精度加速
                with torch.cuda.amp.autocast():
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                # 确保GPU操作完成
                torch.cuda.synchronize()
            else:
                # CPU推理
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # 记录推理时间
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        self.inference_times.append(inference_time)
        self.total_inference_time += inference_time
        
        # 复制帧用于绘制
        processed_frame = frame.copy()
        detections = []
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    # 获取类别名称
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 记录检测结果
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    }
                    detections.append(detection)

        # 只保留每种球类型置信度最高的一个
        best_detections = {}
        for det in detections:
            cname = det['class_name']
            if cname not in best_detections or det['confidence'] > best_detections[cname]['confidence']:
                best_detections[cname] = det
        detections = list(best_detections.values())

        # 绘制和追踪仅对筛选后的检测进行
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = self.get_color_for_class(class_name)
            # 更新追踪历史
            self.tracking_history[class_name].append((center_x, center_y))
            if len(self.tracking_history[class_name]) > 50:
                self.tracking_history[class_name].pop(0)
            # 绘制边界框
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            # 绘制标签
            label = f'{class_name}: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(processed_frame, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        color, -1)
            cv2.putText(processed_frame, label, 
                      (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # 绘制中心点
            cv2.circle(processed_frame, (center_x, center_y), 5, color, -1)

        # 绘制追踪轨迹
        self.draw_tracking_trails(processed_frame)

        # 绘制统计信息
        if isPrintInfo:
            self.draw_statistics(processed_frame, detections, inference_time)

        return processed_frame, detections
    
    def get_color_for_class(self, class_name):
        """根据类别名称获取颜色"""
        if 'red' in class_name.lower():
            return self.colors['red']
        elif 'green' in class_name.lower():
            return self.colors['green']
        elif 'blue' in class_name.lower():
            return self.colors['blue']
        else:
            return (128, 128, 128)  # 灰色作为默认颜色
    
    def draw_tracking_trails(self, frame):
        """绘制追踪轨迹"""
        for class_name, points in self.tracking_history.items():
            if len(points) > 1:
                color = self.get_color_for_class(class_name)
                # 绘制轨迹线
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 1)
    
    def draw_statistics(self, frame, detections, inference_time):
        """绘制统计信息"""
        h, w = frame.shape[:2]
        
        # 统计各类球的数量
        stats = defaultdict(int)
        for det in detections:
            stats[det['class_name']] += 1
        
        # 计算性能指标
        avg_inference_time = np.mean(self.inference_times[-30:]) if self.inference_times else 0
        theoretical_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 绘制背景 - 扩大以显示更多信息
        info_height = 180 if self.device.startswith('cuda') else 140
        cv2.rectangle(frame, (10, 10), (350, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, info_height), (255, 255, 255), 2)
        
        # 显示基本信息
        y_pos = 30
        cv2.putText(frame, f'Frame: {self.frame_count}', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        y_pos += 20
        cv2.putText(frame, f'Device: {self.device}', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示性能信息
        y_pos += 20
        cv2.putText(frame, f'Inference: {inference_time*1000:.1f}ms', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        y_pos += 20
        cv2.putText(frame, f'Avg FPS: {theoretical_fps:.1f}', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 显示GPU信息
        if self.device.startswith('cuda') and torch.cuda.is_available():
            y_pos += 20
            memory_used = torch.cuda.memory_allocated() / 1024**3
            cv2.putText(frame, f'GPU Mem: {memory_used:.2f}GB', 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 显示检测统计
        y_pos += 25
        for class_name in ['red_ball', 'green_ball', 'blue_ball']:
            count = stats.get(class_name, 0)
            color = self.get_color_for_class(class_name)
            cv2.putText(frame, f'{class_name}: {count}', 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_pos += 20
        
        self.frame_count += 1
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return None
        
        return {
            'avg_inference_ms': np.mean(self.inference_times) * 1000,
            'min_inference_ms': np.min(self.inference_times) * 1000,
            'max_inference_ms': np.max(self.inference_times) * 1000,
            'theoretical_fps': 1 / np.mean(self.inference_times),
            'total_frames': len(self.inference_times)
        }
    
# 继承自 QThread，用于处理 YOLO 推理

class YOLOThread(QThread):
    # 定义信号，用于向主线程发送推理结果
    inference_result = pyqtSignal(np.ndarray, list)  # (processed_frame, detections)
    inference_stats = pyqtSignal(dict)  # 性能统计信息
    error_occurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self, path_model, queue_frame, parent=None):
        super().__init__(parent)
        
        
        self.rgb_tracker = RGBBallTracker(path_model, confidence_threshold=0.55, device='auto')
        self.queue_frame = queue_frame
        self.running = False
        
        # 推理控制参数
        self.max_queue_size = 5  # 队列最大大小，防止积压
        self.inference_interval = 0.033  # 推理间隔，约30 FPS
        self.skip_frames = False  # 是否跳帧处理
        
        # 统计信息
        self.processed_frames = 0
        self.skipped_frames = 0
        self.last_stats_time = time.time()
        
    def run(self):
        """线程主循环"""
        print("🚀 YOLO推理线程启动")
        self.running = True
        last_inference_time = time.time()
        
        while self.running:
            # print('i am reasoning')
            try:
               
                # 控制推理频率
                current_time = time.time()
                if current_time - last_inference_time < self.inference_interval:
                    time.sleep(0.01)
                    continue
                
                # 获取帧进行推理
                frame = self._get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 执行推理
                processed_frame, detections = self.rgb_tracker.process_frame(
                    frame, isPrintInfo=True
                )
                
                # 发送推理结果到主线程
                self.inference_result.emit(processed_frame, detections)
                
                # 更新统计信息
                self.processed_frames += 1
                last_inference_time = current_time
                
                # 定期发送性能统计
                if current_time - self.last_stats_time > 1.0:  # 每秒更新一次
                    self._emit_stats()
                    self.last_stats_time = current_time
                
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                error_msg = f"推理线程错误: {str(e)}"
                print(f"❌ {error_msg}")
                self.error_occurred.emit(error_msg)
                self.running = False
                break
        
        print("🛑 YOLO推理线程结束")
    
    def _get_latest_frame(self):
        """
        从队列获取最新帧，可选择性跳帧
        
        Returns:
            frame: 获取到的帧，如果队列为空返回None
        """
        frame = None
        frames_in_queue = 0
        
        # try:
        #     # 如果启用跳帧，获取队列中最新的帧
        #     if self.skip_frames:
        #         # 清空旧帧，只保留最新的
        #         while not self.frame_queue.empty():
        #             frame = self.frame_queue.get_nowait()
        #             frames_in_queue += 1
                
        #         # 统计跳过的帧数
        #         if frames_in_queue > 1:
        #             self.skipped_frames += frames_in_queue - 1
                    
        #     else:
        #         # 不跳帧，按顺序处理
        #         if not self.frame_queue.empty():
        #             frame = self.frame_queue.get_nowait()
        #             frames_in_queue = 1
            
        #     # 如果队列积压过多，自动启用跳帧
        #     if self.frame_queue.qsize() > self.max_queue_size:
        #         self.skip_frames = True
        #         print(f"⚠️ 队列积压({self.frame_queue.qsize()})，启用跳帧模式")
        #     elif self.frame_queue.qsize() < 2:
        #         self.skip_frames = False
                
        # except queue.Empty:
        #     pass
        # except Exception as e:
        #     print(f"获取帧时出错: {e}")
        frame = self.queue_frame.get()
        
        return frame
    
    def _emit_stats(self):
        """发送统计信息"""
        try:
            # 获取RGB追踪器的性能统计
            tracker_stats = self.rgb_tracker.get_performance_stats()
            
            # 综合统计信息
            stats = {
                'processed_frames': self.processed_frames,
                'skipped_frames': self.skipped_frames,
                'queue_size': self.queue_frame.qsize(),
                'skip_mode': self.skip_frames,
                'tracker_stats': tracker_stats
            }
            
            self.inference_stats.emit(stats)
            
        except Exception as e:
            print(f"发送统计信息时出错: {e}")
    
    def start_inference(self):
        """启动推理"""
        if not self.running:
            print("▶️ 启动YOLO推理线程")
            self.start()
        else:
            print("▶️ YOLO已在推理")
    

    
    def stop_inference(self):
        """停止推理线程"""
        print("⏹️ 停止YOLO推理线程")
        self.running = False
        
        # 等待线程结束，但设置超时
        if self.isRunning():
            self.wait(1000)  # 最多等待3秒
            if self.isRunning():
                print("⚠️ 推理线程未能正常结束，强制终止")
                self.terminate()
    
    def set_inference_params(self, fps_limit=30, enable_skip_frames=None, max_queue_size=None):
        """
        设置推理参数
        
        Args:
            fps_limit: FPS限制
            enable_skip_frames: 是否启用跳帧
            max_queue_size: 最大队列大小
        """
        self.inference_interval = 1.0 / fps_limit if fps_limit > 0 else 0.033
        
        if enable_skip_frames is not None:
            self.skip_frames = enable_skip_frames
            
        if max_queue_size is not None:
            self.max_queue_size = max_queue_size
        
        print(f"🔧 推理参数已更新: FPS限制={fps_limit}, 跳帧={self.skip_frames}, 最大队列={self.max_queue_size}")
    
    def clear_queue(self):
        """清空帧队列"""
        try:
            cleared_count = 0
            while not self.queue_frame.empty():
                self.queue_frame.get_nowait()
                cleared_count += 1
            
            if cleared_count > 0:
                print(f"🗑️ 清空了 {cleared_count} 帧")
                
        except Exception as e:
            print(f"清空队列时出错: {e}")
    
    def get_status(self):
        """获取线程状态"""
        return {
            'running': self.running,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.skipped_frames,
            'queue_size': self.queue_frame.qsize(),
            'skip_mode': self.skip_frames
        }

def merge_3d_tracks_frame(det0, det1):
	# det0, det1: list of detection dicts
	# 返回: {class_name: (x0, y0, x1)}
	d3 = {}
	for cname in ["red_ball", "green_ball", "blue_ball"]:
		c0 = next((d for d in det0 if d['class_name']==cname), None)
		c1 = next((d for d in det1 if d['class_name']==cname), None)
		if c0 and c1:
			d3[cname] = (c0['center'][0], c0['center'][1], c1['center'][0])
	return d3

