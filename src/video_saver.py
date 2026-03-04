import cv2
import os
import queue
import threading
import time
from datetime import datetime


class VideoSaver:
    """
    视频保存类，配合 VideoWidget 使用。

    使用方式：
        saver = VideoSaver(video_widget, save_dir="./recordings", fps=30.0)
        start_btn.clicked.connect(saver.SaverStart)
        stop_btn.clicked.connect(saver.SaverEnd)
    """

    _SENTINEL = object()

    def __init__(
        self,
        video_widget,
        save_dir: str = ".",
        filename: str = "",
        fps: float = 30.0,
        fourcc: str = "XVID",
        file_ext: str = ".avi",
    ):
        """
        :param video_widget: VideoWidget 实例
        :param save_dir:     保存目录，不存在会自动创建
        :param filename:     文件名（不含扩展名），空则自动用时间戳
        :param fps:          录制帧率，建议与 VideoWidget.fps 保持一致
        :param fourcc:       编码格式：
                               'XVID' + '.avi'  →  最稳定（默认）
                               'MJPG' + '.avi'  →  文件较大，兼容性强
                               'avc1' + '.mp4'  →  H.264 MP4（需系统支持）
        :param file_ext:     文件扩展名
        """
        self._widget = video_widget
        self.save_dir = save_dir
        self.filename = filename
        self.fps = fps
        self.fourcc = fourcc
        self.file_ext = file_ext

        self._write_thread: threading.Thread | None = None
        self._save_queue: queue.Queue = queue.Queue()
        self.is_recording: bool = False
        self._current_output_path: str = ""

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #

    def SaverStart(self) -> bool:
        if self.is_recording:
            print("[VideoSaver] 已在录制中，请勿重复调用 SaverStart()")
            return False

        frame_size = self._get_frame_size()
        # ★ 若当前还没有帧（刚开始播放），等待最多 2 秒
        if frame_size is None:
            print("[VideoSaver] 等待视频帧...")
            for _ in range(20):
                time.sleep(0.1)
                frame_size = self._get_frame_size()
                if frame_size is not None:
                    break

        if frame_size is None:
            print("[VideoSaver] 无法获取视频分辨率，请确认摄像头已开启并有帧输出")
            return False

        output_path = self._build_output_path()
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)

        writer = cv2.VideoWriter(output_path, fourcc_code, self.fps, frame_size)
        if not writer.isOpened():
            print(f"[VideoSaver] VideoWriter 初始化失败，路径: {output_path}")
            writer.release()
            return False

        # 清空队列残留
        while not self._save_queue.empty():
            try:
                self._save_queue.get_nowait()
            except queue.Empty:
                break

        # ★ 通过回调钩子注册帧接收，不再 monkey-patch loadframe
        self._register_callback()

        # 启动写线程（writer 的 write/release 只在此线程内调用）
        self._write_thread = threading.Thread(
            target=self._write_loop,
            args=(writer,),
            daemon=True,
            name="VideoSaverThread",
        )
        self._write_thread.start()

        self.is_recording = True
        print(f"[VideoSaver] 开始录制 → {output_path}  {frame_size} @ {self.fps} FPS")
        return True

    def SaverEnd(self) -> str | None:
        if not self.is_recording:
            print("[VideoSaver] 当前未在录制，SaverEnd() 无效")
            return None

        self.is_recording = False

        # ★ 注销回调，停止往队列里放新帧
        self._unregister_callback()

        # 哨兵通知写线程：排空队列后 release 并退出
        self._save_queue.put(self._SENTINEL)

        if self._write_thread is not None:
            self._write_thread.join(timeout=30)
            self._write_thread = None

        saved_path = self._current_output_path
        print(f"[VideoSaver] 录制结束，文件已保存: {saved_path}")
        return saved_path

    # ------------------------------------------------------------------ #
    #  内部方法                                                             #
    # ------------------------------------------------------------------ #

    def _build_output_path(self) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        name = self.filename if self.filename else datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_output_path = os.path.join(self.save_dir, name + self.file_ext)
        return self._current_output_path

    def _get_frame_size(self):
        if self._widget.current_frame is not None:
            h, w = self._widget.current_frame.shape[:2]
            return (w, h)
        if self._widget.cap is not None and self._widget.cap.isOpened():
            w = int(self._widget.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._widget.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return None

    def _on_frame(self, frame):
        """★ 帧回调：由 VideoWidget.loadframe 在主线程中调用"""
        if self.is_recording:
            self._save_queue.put(frame.copy())

    def _register_callback(self):
        """★ 向 VideoWidget 注册帧回调"""
        if self._on_frame not in self._widget._frame_callbacks:
            self._widget._frame_callbacks.append(self._on_frame)

    def _unregister_callback(self):
        """★ 从 VideoWidget 注销帧回调"""
        try:
            self._widget._frame_callbacks.remove(self._on_frame)
        except ValueError:
            pass  # 已经不在列表中，忽略

    def _write_loop(self, writer: cv2.VideoWriter):
        """
        写线程：writer 的 write/release 均在此线程内，无跨线程竞争。
        """
        try:
            while True:
                item = self._save_queue.get()
                if item is self._SENTINEL:
                    # 排空剩余帧后退出
                    while True:
                        try:
                            remaining = self._save_queue.get_nowait()
                            if remaining is not self._SENTINEL:
                                writer.write(remaining)
                        except queue.Empty:
                            break
                    break
                writer.write(item)
        except Exception as e:
            print(f"[VideoSaver] 写帧异常: {e}")
        finally:
            writer.release()
            print("[VideoSaver] VideoWriter 已释放")
