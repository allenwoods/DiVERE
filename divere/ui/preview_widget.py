"""
预览组件
用于显示图像预览
"""

import numpy as np
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QCursor

from divere.core.data_types import ImageData


class PreviewWidget(QWidget):
    """图像预览组件"""
    
    # 发送图像旋转信号，参数为旋转方向：1=左旋，-1=右旋
    image_rotated = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        
        self.current_image: Optional[ImageData] = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # 拖动相关状态
        self.dragging = False
        self.last_mouse_pos = None
        self.drag_start_pos = None
        self.original_pan_pos = None
        
        # 平滑拖动相关
        self.smooth_drag_timer = QTimer()
        self.smooth_drag_timer.timeout.connect(self._smooth_drag_update)
        self.smooth_drag_timer.setInterval(16)  # ~60fps
        
        self._create_ui()
        self._setup_mouse_controls()
        self._setup_keyboard_controls()
    
    def _create_ui(self):
        """创建用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        # 旋转按钮
        self.rotate_left_btn = QPushButton("← 左旋")
        self.rotate_right_btn = QPushButton("右旋 →")
        self.rotate_left_btn.setMaximumWidth(80)
        self.rotate_right_btn.setMaximumWidth(80)
        self.rotate_left_btn.clicked.connect(self.rotate_left)
        self.rotate_right_btn.clicked.connect(self.rotate_right)
        
        # 视图控制按钮
        self.fit_window_btn = QPushButton("适应窗口")
        self.center_btn = QPushButton("居中")
        self.fit_window_btn.setMaximumWidth(80)
        self.center_btn.setMaximumWidth(80)
        self.fit_window_btn.clicked.connect(self.fit_to_window)
        self.center_btn.clicked.connect(self.center_image)
        
        # 添加按钮到布局
        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        button_layout.addWidget(self.fit_window_btn)
        button_layout.addWidget(self.center_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setText("请加载图像")
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; color: #ffffff; }")
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
    
    def _setup_mouse_controls(self):
        """设置鼠标控制"""
        # 将鼠标事件绑定到image_label而不是scroll_area
        self.image_label.wheelEvent = self._wheel_event
        self.image_label.mousePressEvent = self._mouse_press_event
        self.image_label.mouseMoveEvent = self._mouse_move_event
        self.image_label.mouseReleaseEvent = self._mouse_release_event
        
        # 设置鼠标样式
        self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
    def _setup_keyboard_controls(self):
        """设置键盘控制"""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def set_image(self, image_data: ImageData):
        """设置显示的图像"""
        self.current_image = image_data
        self._update_display()

    def get_current_image_data(self) -> Optional[ImageData]:
        """返回当前显示的ImageData对象"""
        return self.current_image
    
    def _update_display(self):
        """更新显示"""
        if not self.current_image or self.current_image.array is None:
            self.image_label.setText("请加载图像")
            return
        
        try:
            pixmap = self._array_to_pixmap(self.current_image.array)
            if self.zoom_factor != 1.0:
                scaled_size = pixmap.size() * self.zoom_factor
                pixmap = pixmap.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            # 应用平移偏移
            if self.pan_x != 0 or self.pan_y != 0:
                # 创建一个更大的画布来容纳平移后的图像
                original_size = pixmap.size()
                canvas_width = original_size.width() + abs(self.pan_x)
                canvas_height = original_size.height() + abs(self.pan_y)
                canvas = QPixmap(canvas_width, canvas_height)
                canvas.fill(QColor(0, 0, 0, 0))  # 透明背景
                
                painter = QPainter(canvas)
                painter.drawPixmap(self.pan_x, self.pan_y, pixmap)
                painter.end()
                
                pixmap = canvas
            
            self.image_label.setPixmap(pixmap)
            
        except Exception as e:
            print(f"更新显示失败: {e}")
            self.image_label.setText(f"显示错误: {str(e)}")
    
    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """将numpy数组转换为QPixmap"""
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        
        # 确保数组是连续的内存布局（旋转后可能不连续）
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        
        height, width = array.shape[:2]
        
        if len(array.shape) == 3:
            channels = array.shape[2]
            if channels == 3:
                qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
            elif channels == 4:
                qimage = QImage(array.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
            else: # Fallback for other channel counts
                array = array[:, :, :3]
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
        
        # 为DisplayP3图像应用Qt内置色彩管理
        if hasattr(self, 'current_image') and self.current_image and self.current_image.color_space == "Rec2020":
            from PyQt6.QtGui import QColorSpace
            # 创建色彩空间（DisplayP3）
            displayp3_space = QColorSpace(QColorSpace.NamedColorSpace.DisplayP3)
            # 应用色彩空间到QImage
            qimage.setColorSpace(displayp3_space)
        
        return QPixmap.fromImage(qimage)
    
    def _wheel_event(self, event):
        """鼠标滚轮事件 - 缩放"""
        if not self.current_image: 
            event.accept()
            return
            
        delta = event.angleDelta().y()
        
        # 获取鼠标在图像上的位置
        mouse_pos = event.position()
        
        # 执行缩放
        zoom_factor_change = 1.05 if delta > 0 else 1/1.05
        old_zoom = self.zoom_factor
        self.zoom_factor *= zoom_factor_change
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        
        # 更新显示
        self._update_display()
        
        event.accept()
    
    def _mouse_press_event(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.drag_start_pos = event.pos()
            
            # 记录开始拖动时的平移位置
            self.original_pan_pos = QPoint(self.pan_x, self.pan_y)
            
            # 改变鼠标样式
            self.image_label.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            
        event.accept()
    
    def _mouse_move_event(self, event):
        """鼠标移动事件"""
        if self.dragging and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            
            # 直接更新平移偏移
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            
            # 更新显示
            self._update_display()
            
            self.last_mouse_pos = event.pos()
            
        event.accept()
    
    def _mouse_release_event(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_mouse_pos = None
            self.drag_start_pos = None
            self.original_pan_pos = None
            
            # 恢复鼠标样式
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            
        event.accept()
    
    def _smooth_drag_update(self):
        """平滑拖动更新（预留功能）"""
        # 这里可以实现更平滑的拖动效果
        pass
    
    def reset_view(self):
        """重置视图"""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self._update_display()

    def fit_to_window(self):
        """适应窗口大小"""
        if not self.current_image: 
            return
            
        widget_size = self.scroll_area.size()
        image_size = self.current_image.array.shape[:2][::-1]
        scale_x = widget_size.width() / image_size[0]
        scale_y = widget_size.height() / image_size[1]
        self.zoom_factor = min(scale_x, scale_y, 1.0)
        
        # 重置平移
        self.pan_x = 0
        self.pan_y = 0
        
        self._update_display()
    
    def center_image(self):
        """居中显示图像"""
        if not self.current_image:
            return
            
        # 重置平移以居中显示
        self.pan_x = 0
        self.pan_y = 0
        
        self._update_display()
        
    def rotate_left(self):
        """左旋90度"""
        if self.current_image and self.current_image.array is not None:
            # 发送信号通知主窗口执行旋转
            self.image_rotated.emit(1)  # 1表示左旋
            
    def rotate_right(self):
        """右旋90度"""
        if self.current_image and self.current_image.array is not None:
            # 发送信号通知主窗口执行旋转
            self.image_rotated.emit(-1)  # -1表示右旋
            
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key.Key_Left:
            self.rotate_left()
        elif event.key() == Qt.Key.Key_Right:
            self.rotate_right()
        elif event.key() == Qt.Key.Key_0:  # 数字0键重置视图
            self.reset_view()
        elif event.key() == Qt.Key.Key_F:  # F键适应窗口
            self.fit_to_window()
        elif event.key() == Qt.Key.Key_C:  # C键居中
            self.center_image()
        else:
            super().keyPressEvent(event)
