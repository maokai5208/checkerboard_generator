"""
棋盘格生成工具
支持生成棋盘格、嵌入DM码（Data Matrix码）、关联物理坐标系
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Tuple, Optional, Dict
import os


class CheckerboardGenerator:
    """棋盘格生成器"""
    
    def __init__(self, 
                 rows: int = 9,
                 cols: int = 6,
                 square_size: float = 20.0,
                 image_size: Tuple[int, int] = (2000, 3000),
                 margin: int = 100):
        """
        初始化棋盘格生成器
        
        参数:
            rows: 内部角点行数（棋盘格内部交叉点）
            cols: 内部角点列数
            square_size: 每个方格的物理尺寸（单位：mm）
            image_size: 输出图像尺寸 (width, height)
            margin: 边距（像素）
        """
        self.rows = rows
        self.cols = cols
        self.square_size = square_size  # 物理尺寸（mm）
        self.image_size = image_size
        self.margin = margin
        
        # 计算棋盘格实际尺寸
        self.checkerboard_size = (
            cols * square_size,
            rows * square_size
        )
        
        # 物理坐标系原点（默认在左上角）
        self.origin = np.array([0.0, 0.0])
        # 物理坐标系轴方向（默认：x向右，y向下）
        self.axis_x = np.array([1.0, 0.0])
        self.axis_y = np.array([0.0, 1.0])
        
    def generate_checkerboard(self) -> np.ndarray:
        """
        生成棋盘格图像
        
        返回:
            棋盘格图像（numpy数组，BGR格式）
        """
        width, height = self.image_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 计算棋盘格在图像中的位置（居中）
        board_width = width - 2 * self.margin
        board_height = height - 2 * self.margin
        
        # 计算每个方格在图像中的像素尺寸
        square_pixel_w = board_width / self.cols
        square_pixel_h = board_height / self.rows
        
        # 绘制棋盘格
        for i in range(self.rows):
            for j in range(self.cols):
                x = int(self.margin + j * square_pixel_w)
                y = int(self.margin + i * square_pixel_h)
                w = int(square_pixel_w)
                h = int(square_pixel_h)
                
                # 交替绘制黑白方格（第一个方格为黑色）
                if (i + j) % 2 == 0:
                    color = (0, 0, 0)  # 黑色
                else:
                    color = (255, 255, 255)  # 白色
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        
        return img
    
    def get_square_pixel_size(self) -> Tuple[float, float]:
        """
        计算棋盘格每个方格在图像中的像素尺寸
        
        返回:
            (宽度, 高度) 像素尺寸
        """
        width, height = self.image_size
        board_width = width - 2 * self.margin
        board_height = height - 2 * self.margin
        square_pixel_w = board_width / self.cols
        square_pixel_h = board_height / self.rows
        return square_pixel_w, square_pixel_h
    
    def calculate_dm_border_size(self, dm_size: int, data: str = None) -> int:
        """
        根据DM码尺寸计算边框大小（留1个内部特征方块大小）
        
        参数:
            dm_size: DM码尺寸（像素）
            data: DM码数据（可选，如果提供则获取实际模块数量）
        
        返回:
            边框大小（像素）
        """
        # 如果提供了数据，尝试获取实际的模块数量
        if data is not None:
            try:
                from pylibdmtx.pylibdmtx import encode
                encoded = encode(data.encode('utf-8'))
                # 获取原始模块数量（Data Matrix码是方形的，width和height应该相同）
                original_modules = encoded.width  # 使用width，因为DM码是方形的
                
                # 计算缩放后的模块大小
                # 原始尺寸 -> 目标尺寸的缩放比例
                scale_factor = dm_size / original_modules
                
                # 1个模块的大小 = 1 * 缩放后的模块大小（强制使用，不限制）
                border_size = 1 * scale_factor
                
                # 四舍五入到最近的整数（更准确）
                border_size = int(np.round(border_size))
                
                # 强制使用1个模块大小，不进行限制调整
                # 确保至少为1像素（避免为0）
                if border_size < 1:
                    border_size = 1
                
                return border_size
            except Exception as e:
                # 如果获取失败，使用估算方法
                pass
        
        # 估算方法：根据DM码尺寸估算模块数量
        # Data Matrix码的常见模块数量：10x10, 12x12, 14x14, 16x16, 18x18, 20x20, 22x22, 24x24等
        # 对于我们的数据格式（ID:X:Y），通常使用10x10或12x12
        
        # 根据DM码尺寸估算模块数量
        if dm_size <= 60:
            estimated_modules = 10
        elif dm_size <= 120:
            estimated_modules = 12
        elif dm_size <= 180:
            estimated_modules = 14
        elif dm_size <= 240:
            estimated_modules = 16
        elif dm_size <= 300:
            estimated_modules = 18
        else:
            estimated_modules = 20
        
        # 计算模块大小
        module_size = dm_size / estimated_modules
        
        # 边框大小 = 1个模块大小（强制使用，不限制）
        border_size = module_size * 1
        
        # 四舍五入到最近的整数
        border_size = int(np.round(border_size))
        
        # 强制使用1个模块大小，不进行限制调整
        # 确保至少为1像素（避免为0）
        if border_size < 1:
            border_size = 1
        
        return border_size
    
    def generate_dm_code(self, data: str, size: int = 200) -> np.ndarray:
        """
        生成真正的Data Matrix码（DM码）
        
        Data Matrix码的特征：边缘有两条实体线（L型定位图案）
        
        参数:
            data: 要编码的数据
            size: DM码尺寸（像素）
        
        返回:
            DM码图像（numpy数组，BGR格式）
        """
        try:
            # 方案1: 使用pylibdmtx生成真正的Data Matrix码
            try:
                from pylibdmtx.pylibdmtx import encode
                
                # 编码数据为Data Matrix码
                encoded = encode(data.encode('utf-8'))
                
                # 将编码结果转换为PIL图像
                from PIL import Image
                dm_img_pil = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
                
                # 转换为numpy数组
                dm_img = np.array(dm_img_pil)
                
                # 转换为BGR格式
                dm_img = cv2.cvtColor(dm_img, cv2.COLOR_RGB2BGR)
                
                # 调整尺寸（使用最近邻插值保持清晰度）
                dm_img = cv2.resize(dm_img, (size, size), interpolation=cv2.INTER_NEAREST)
                
                return dm_img
            except ImportError:
                print("警告: pylibdmtx未安装，尝试使用备用方案")
                raise ImportError("pylibdmtx not installed")
            except Exception as e:
                print(f"使用pylibdmtx生成DM码失败: {e}")
                raise
            
        except (ImportError, Exception) as e:
            # 备用方案：使用datamatrix库（如果可用）
            try:
                import datamatrix
                from PIL import Image, ImageDraw
                
                # 创建Data Matrix码
                dm = datamatrix.DataMatrix(data)
                dm.save("temp_dm.png")
                
                # 读取图像
                dm_img = cv2.imread("temp_dm.png")
                if dm_img is not None:
                    dm_img = cv2.resize(dm_img, (size, size), interpolation=cv2.INTER_NEAREST)
                    os.remove("temp_dm.png")
                    return dm_img
            except ImportError:
                pass
            except Exception as e2:
                print(f"使用datamatrix库失败: {e2}")
            
            # 如果都失败，生成一个带提示的占位符
            placeholder = np.ones((size, size, 3), dtype=np.uint8) * 255
            # 绘制L型边框（模拟DM码的特征）
            border_width = max(3, size // 20)
            cv2.rectangle(placeholder, (0, 0), (size//3, border_width), (0, 0, 0), -1)  # 上边
            cv2.rectangle(placeholder, (0, 0), (border_width, size//3), (0, 0, 0), -1)  # 左边
            cv2.putText(placeholder, "DM", (size//4, size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            print(f"警告: 无法生成真正的DM码 ({e})，使用占位符。请安装: pip install pylibdmtx")
            return placeholder
    
    def generate_coordinate_dm_data(self, 
                                    row: int,
                                    col: int,
                                    spacing: int,
                                    square_size: float) -> str:
        """
        生成包含坐标信息的DM码数据
        格式: "行,列,G间距S棋盘格大小"
        例如: "5,5,G5S10.0" 表示第5行第5列，条码间距为5，单个棋盘格尺寸为10.0mm
        
        参数:
            row: 行索引（从0开始）
            col: 列索引（从0开始）
            spacing: 条码间距（每隔多少个方格放置一个DM码）
            square_size: 单个棋盘格尺寸（mm）
        
        返回:
            格式化的坐标信息字符串
        """
        # 格式: "行,列,G间距S棋盘格大小"
        # 注意：行、列从1开始计数（用户友好）
        return f"{row+1},{col+1},G{spacing}S{square_size:.1f}"
    
    def embed_dm_code(self, 
                      img: np.ndarray,
                      dm_id: int,
                      x: float,
                      y: float,
                      dm_size: int = None,
                      position: str = "top-left",
                      square_row: int = None,
                      square_col: int = None,
                      dm_square_size: int = 2,
                      spacing: int = None) -> np.ndarray:
        """
        在棋盘格图像中嵌入DM码
        
        参数:
            img: 棋盘格图像
            dm_id: DM码的ID编号
            x: DM码位置的物理x坐标（mm）
            y: DM码位置的物理y坐标（mm）
            dm_size: DM码尺寸（像素），如果为None则自动根据方格大小计算
            position: DM码在图像中的位置 ("top-left", "top-right", "bottom-left", "bottom-right")
            square_row: 方格行索引（从0开始），如果指定则从该方格开始嵌入
            square_col: 方格列索引（从0开始），如果指定则从该方格开始嵌入
            dm_square_size: DM码占据的方格数量（默认2，即2x2=4个方格）
        
        返回:
            嵌入DM码后的图像
        """
        # 计算方格大小
        square_pixel_w, square_pixel_h = self.get_square_pixel_size()
        
        # 如果指定了方格位置，从该方格开始嵌入，占据dm_square_size x dm_square_size个方格
        if square_row is not None and square_col is not None:
            # 计算起始方格位置
            square_x = int(self.margin + square_col * square_pixel_w)
            square_y = int(self.margin + square_row * square_pixel_h)
            square_w = int(square_pixel_w)
            square_h = int(square_pixel_h)
            
            # DM码占据的区域大小（2x2个方格）
            dm_area_w = int(square_w * dm_square_size)
            dm_area_h = int(square_h * dm_square_size)
            
            # 生成坐标信息（格式: "行,列,G间距S棋盘格大小"）
            # 如果spacing为None，则无法生成新格式，使用默认值
            if spacing is None:
                spacing = 5  # 默认间距
            dm_data = self.generate_coordinate_dm_data(square_row, square_col, spacing, self.square_size)
            
            # 先计算临时DM码尺寸，用于计算边框
            if dm_size is None:
                temp_dm_size = min(dm_area_w, dm_area_h) - 4  # 预留一些空间
            else:
                temp_dm_size = dm_size
            
            # 计算自适应边框大小（留2个内部特征方块大小）
            # 传入数据以便获取实际模块数量
            border_size = self.calculate_dm_border_size(temp_dm_size, data=dm_data)
            
            # DM码尺寸 = 占据区域大小 - 2倍边框大小
            if dm_size is None:
                dm_size = min(dm_area_w, dm_area_h) - 2 * border_size
            
            # 重新计算边框（基于最终DM码尺寸和实际数据）
            border_size = self.calculate_dm_border_size(dm_size, data=dm_data)
            
            # 生成DM码图像
            dm_img = self.generate_dm_code(dm_data, dm_size)
            
            # 在占据区域的中心嵌入DM码，留自适应边框
            pos_x = square_x + border_size
            pos_y = square_y + border_size
            
            # 将DM码嵌入到图像中
            result_img = img.copy()
            
            # 先清除DM码占据区域的棋盘格背景（填充白色）
            # 清除整个占据区域（包括边框）
            clear_area_x = square_x
            clear_area_y = square_y
            clear_area_w = dm_area_w
            clear_area_h = dm_area_h
            
            # 确保不超出图像边界
            img_h, img_w = result_img.shape[:2]
            clear_end_x = min(clear_area_x + clear_area_w, img_w)
            clear_end_y = min(clear_area_y + clear_area_h, img_h)
            
            # 填充白色背景
            result_img[clear_area_y:clear_end_y, clear_area_x:clear_end_x] = (255, 255, 255)
            
            # 然后嵌入DM码
            result_img[pos_y:pos_y+dm_size, pos_x:pos_x+dm_size] = dm_img
            
            return result_img
        
        # 否则使用原来的位置逻辑
        if dm_size is None:
            # 默认使用方格大小，先计算边框
            temp_dm_size = int(min(square_pixel_w, square_pixel_h)) - 4
            border_size = self.calculate_dm_border_size(temp_dm_size)
            dm_size = int(min(square_pixel_w, square_pixel_h)) - 2 * border_size
        else:
            # 计算边框大小
            border_size = self.calculate_dm_border_size(dm_size)
        
        # 生成坐标信息（格式: "行,列,G间距S棋盘格大小"）
        # 如果square_row和square_col为None，从物理坐标反推（近似）
        if square_row is None or square_col is None:
            if spacing is None:
                spacing = 5  # 默认间距
            approx_row = int(y / self.square_size) if self.square_size > 0 else 0
            approx_col = int(x / self.square_size) if self.square_size > 0 else 0
            dm_data = self.generate_coordinate_dm_data(approx_row, approx_col, spacing, self.square_size)
        else:
            if spacing is None:
                spacing = 5  # 默认间距
            dm_data = self.generate_coordinate_dm_data(square_row, square_col, spacing, self.square_size)
        dm_img = self.generate_dm_code(dm_data, dm_size)
        
        # 计算DM码在图像中的位置
        img_h, img_w = img.shape[:2]
        
        if position == "top-left":
            pos_x = self.margin // 2
            pos_y = self.margin // 2
        elif position == "top-right":
            pos_x = img_w - self.margin // 2 - dm_size - 2 * border_size
            pos_y = self.margin // 2
        elif position == "bottom-left":
            pos_x = self.margin // 2
            pos_y = img_h - self.margin // 2 - dm_size - 2 * border_size
        elif position == "bottom-right":
            pos_x = img_w - self.margin // 2 - dm_size - 2 * border_size
            pos_y = img_h - self.margin // 2 - dm_size - 2 * border_size
        else:
            pos_x = self.margin // 2
            pos_y = self.margin // 2
        
        # 将DM码嵌入到图像中（留自适应边框）
        result_img = img.copy()
        result_img[pos_y+border_size:pos_y+border_size+dm_size, 
                   pos_x+border_size:pos_x+border_size+dm_size] = dm_img
        
        return result_img
    
    def embed_multiple_dm_codes(self,
                                img: np.ndarray,
                                dm_positions: list,
                                dm_size: int = None,
                                dm_square_size: int = 2,
                                spacing: int = None) -> np.ndarray:
        """
        在棋盘格图像中嵌入多个DM码
        
        参数:
            img: 棋盘格图像
            dm_positions: DM码位置列表，每个元素为:
                         - (dm_id, x, y, position) - 使用位置字符串
                         - (dm_id, x, y, square_row, square_col) - 从指定方格开始嵌入，占据dm_square_size x dm_square_size个方格
                         - (dm_id, x, y, pos_x, pos_y, "custom") - 自定义像素位置
                         x, y为物理坐标（mm）
            dm_size: DM码尺寸（像素），如果为None则自动根据方格大小计算
            dm_square_size: DM码占据的方格数量（默认2，即2x2=4个方格）
        
        返回:
            嵌入DM码后的图像
        """
        result_img = img.copy()
        
        for dm_info in dm_positions:
            if len(dm_info) == 4:
                # 格式: (dm_id, x, y, position)
                dm_id, x, y, position = dm_info
                result_img = self.embed_dm_code(result_img, dm_id, x, y, dm_size, position, 
                                             dm_square_size=dm_square_size)
            elif len(dm_info) == 5:
                # 判断是方格位置还是自定义位置
                dm_id, x, y, arg1, arg2 = dm_info
                if isinstance(arg1, int) and isinstance(arg2, int):
                    # 格式: (dm_id, x, y, square_row, square_col) - 从指定方格开始嵌入
                    square_row, square_col = arg1, arg2
                    result_img = self.embed_dm_code(result_img, dm_id, x, y, dm_size, 
                                                   square_row=square_row, square_col=square_col,
                                                   dm_square_size=dm_square_size, spacing=spacing)
                elif isinstance(arg1, (int, float)) and isinstance(arg2, (int, float)):
                    # 格式: (dm_id, x, y, pos_x, pos_y) - 自定义像素位置（旧格式，兼容）
                    pos_x, pos_y = int(arg1), int(arg2)
                    square_pixel_w, square_pixel_h = self.get_square_pixel_size()
                    # 计算2x2方格的大小
                    dm_area_size = int(min(square_pixel_w, square_pixel_h) * dm_square_size)
                    
                    # 先计算临时DM码尺寸，用于计算边框
                    if dm_size is None:
                        temp_dm_size = dm_area_size - 4  # 预留一些空间
                    else:
                        temp_dm_size = dm_size
                    
                    # 计算自适应边框大小
                    border_size = self.calculate_dm_border_size(temp_dm_size)
                    
                    # DM码尺寸 = 占据区域大小 - 2倍边框大小
                    if dm_size is None:
                        dm_size = dm_area_size - 2 * border_size
                    
                    # 重新计算边框（基于最终DM码尺寸）
                    border_size = self.calculate_dm_border_size(dm_size)
                    
                    # 从像素位置反推行、列（近似）
                    # 如果无法确定，使用默认值
                    if spacing is None:
                        spacing = 5
                    # 从物理坐标反推行、列
                    approx_row = int(y / self.square_size) if self.square_size > 0 else 0
                    approx_col = int(x / self.square_size) if self.square_size > 0 else 0
                    dm_data = self.generate_coordinate_dm_data(approx_row, approx_col, spacing, self.square_size)
                    dm_img = self.generate_dm_code(dm_data, dm_size)
                    
                    # 先清除背景（填充白色）
                    img_h, img_w = result_img.shape[:2]
                    clear_area_x = pos_x
                    clear_area_y = pos_y
                    clear_area_w = dm_area_size
                    clear_area_h = dm_area_size
                    
                    clear_end_x = min(clear_area_x + clear_area_w, img_w)
                    clear_end_y = min(clear_area_y + clear_area_h, img_h)
                    
                    if clear_area_x >= 0 and clear_area_y >= 0:
                        result_img[clear_area_y:clear_end_y, clear_area_x:clear_end_x] = (255, 255, 255)
                    
                    # 然后嵌入DM码，留自适应边框
                    if pos_x >= 0 and pos_y >= 0 and pos_x + dm_size + 2*border_size <= img_w and pos_y + dm_size + 2*border_size <= img_h:
                        result_img[pos_y+border_size:pos_y+border_size+dm_size, 
                                   pos_x+border_size:pos_x+border_size+dm_size] = dm_img
            else:
                raise ValueError("dm_positions格式错误，应为 (dm_id, x, y, position) 或 (dm_id, x, y, square_row, square_col)")
        
        return result_img
    
    def set_coordinate_system(self,
                              origin: Tuple[float, float],
                              axis_x: Tuple[float, float] = (1.0, 0.0),
                              axis_y: Tuple[float, float] = (0.0, 1.0)):
        """
        设置物理坐标系
        
        参数:
            origin: 原点坐标 (x, y)
            axis_x: X轴方向向量 (x, y)
            axis_y: Y轴方向向量 (x, y)
        """
        self.origin = np.array(origin)
        self.axis_x = np.array(axis_x)
        self.axis_y = np.array(axis_y)
    
    def generate_dm_positions_by_spacing(self, 
                                        spacing: int = 5,
                                        start_id: int = 1,
                                        start_position: int = None) -> list:
        """
        根据间距自动生成DM码位置
        
        参数:
            spacing: DM码间距（每隔多少个方格放置一个DM码）
            start_id: 起始DM码ID
            start_position: 第一个DM码的起始位置（行和列索引，从0开始）
                          如果为None，则使用spacing作为起始位置
        
        返回:
            DM码位置列表，格式: [(dm_id, x, y, square_row, square_col), ...]
        """
        dm_positions = []
        dm_id = start_id
        
        # 确定第一个DM码的起始位置
        if start_position is None:
            start_pos = spacing
        else:
            start_pos = start_position
        
        # 第一个DM码从start_pos行start_pos列开始
        # 然后每隔spacing个方格放置一个DM码
        # 注意：DM码占据2x2个方格，需要检查边界
        for row in range(start_pos, self.rows, spacing):
            for col in range(start_pos, self.cols, spacing):
                # 检查DM码是否会超出棋盘格边界
                # DM码占据2x2个方格，所以需要检查 row+1 和 col+1 是否在范围内
                if row + 1 < self.rows and col + 1 < self.cols:
                    # 计算物理坐标（mm）
                    x = col * self.square_size
                    y = row * self.square_size
                    
                    dm_positions.append((dm_id, x, y, row, col))
                    dm_id += 1
        
        return dm_positions
    
    def save_checkerboard(self, 
                         output_path: str,
                         embed_dm: bool = True,
                         dm_positions: list = None,
                         dm_size: int = None,
                         dm_spacing: int = None,
                         dm_square_size: int = 2,
                         dm_start_position: int = None) -> str:
        """
        生成并保存棋盘格图像
        
        参数:
            output_path: 输出文件路径
            embed_dm: 是否嵌入DM码
            dm_positions: DM码位置列表，每个元素为:
                         - (dm_id, x, y, position) - 使用位置字符串
                         - (dm_id, x, y, square_row, square_col) - 从指定方格开始嵌入
                         如果为None且dm_spacing也为None，默认在左上角第一个方格嵌入一个DM码
            dm_size: DM码尺寸（像素），如果为None则自动根据方格大小计算（留1像素边框）
            dm_spacing: DM码间距（每隔多少个方格放置一个DM码），如果指定则自动生成位置
            dm_square_size: DM码占据的方格数量（默认2，即2x2=4个方格）
            dm_start_position: 第一个DM码的起始位置（行和列索引，从0开始）
                               如果为None，则使用dm_spacing作为起始位置
        
        返回:
            保存的文件路径
        """
        # 生成棋盘格
        img = self.generate_checkerboard()
        
        # 嵌入DM码
        if embed_dm:
            if dm_spacing is not None:
                # 根据间距自动生成DM码位置
                dm_positions = self.generate_dm_positions_by_spacing(
                    spacing=dm_spacing, 
                    start_position=dm_start_position
                )
            
            if dm_positions is None:
                # 默认在左上角第一个方格嵌入一个DM码（占据2x2个方格）
                img = self.embed_dm_code(img, dm_id=1, x=0.0, y=0.0, dm_size=dm_size, 
                                         square_row=0, square_col=0, dm_square_size=dm_square_size,
                                         spacing=dm_spacing)
            else:
                img = self.embed_multiple_dm_codes(img, dm_positions, dm_size, dm_square_size, spacing=dm_spacing)
        
        # 保存图像
        cv2.imwrite(output_path, img)
        
        # 保存配置信息
        config_path = output_path.replace('.png', '_config.json').replace('.jpg', '_config.json')
        self.save_config(config_path)
        
        return output_path
    
    def save_config(self, config_path: str):
        """保存配置信息到JSON文件"""
        config = {
            "checkerboard": {
                "rows": self.rows,
                "cols": self.cols,
                "square_size": self.square_size,
                "unit": "mm"
            },
            "image": {
                "width": self.image_size[0],
                "height": self.image_size[1],
                "margin": self.margin
            },
            "coordinate_system": {
                "origin": {
                    "x": float(self.origin[0]),
                    "y": float(self.origin[1])
                },
                "axis_x": {
                    "x": float(self.axis_x[0]),
                    "y": float(self.axis_x[1])
                },
                "axis_y": {
                    "x": float(self.axis_y[0]),
                    "y": float(self.axis_y[1])
                }
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)


def generate_checkerboard_simple(square_size: float,
                                  rows: int,
                                  cols: int,
                                  dm_spacing: int,
                                  dm_start_position: int = None,
                                  output_path: str = None) -> str:
    """
    简化的棋盘格生成接口，只需输入4个参数（可选第5个参数指定起始位置）
    
    参数:
        square_size: 棋盘格大小（每个方格的物理尺寸，单位：mm）
        rows: 行数
        cols: 列数
        dm_spacing: DM码间距（每隔多少个方格放置一个DM码）
        dm_start_position: 第一个DM码的起始位置（行和列索引，从0开始）
                           如果为None，则使用dm_spacing作为起始位置
                           例如：如果设置为5，则第一个DM码从第5行第5列开始
        output_path: 输出文件路径（可选，默认：output/checkerboard_with_dm.png）
    
    返回:
        保存的文件路径
    """
    # 创建output文件夹（如果不存在）
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 如果未指定输出路径，使用默认路径
    if output_path is None:
        output_path = os.path.join(output_dir, "checkerboard_with_dm.png")
    else:
        # 如果指定了路径但不包含output目录，则添加到output目录
        # 使用os.path处理跨平台路径
        output_path_normalized = os.path.normpath(output_path)
        output_dir_normalized = os.path.normpath(output_dir)
        if not output_path_normalized.startswith(output_dir_normalized + os.sep) and output_path_normalized != output_dir_normalized:
            filename = os.path.basename(output_path)
            output_path = os.path.join(output_dir, filename)
    # 自动计算图像尺寸
    # 每个方格至少100像素，加上边距
    pixels_per_mm = 10  # 每毫米10像素（可根据需要调整）
    square_pixel_size = int(square_size * pixels_per_mm)
    
    # 计算棋盘格区域大小
    board_width = cols * square_pixel_size
    board_height = rows * square_pixel_size
    
    # 设置边距（棋盘格区域大小的10%，最小100像素）
    margin = 100
    
    # 计算总图像尺寸
    image_width = board_width + 2 * margin
    image_height = board_height + 2 * margin
    
    # 创建棋盘格生成器
    generator = CheckerboardGenerator(
        rows=rows,
        cols=cols,
        square_size=square_size,
        image_size=(image_width, image_height),
        margin=margin
    )
    
    # 生成并保存棋盘格（每个DM码占据2x2个方格）
    output_path = generator.save_checkerboard(
        output_path=output_path,
        embed_dm=True,
        dm_spacing=dm_spacing,      # 每隔指定个方格放置一个DM码
        dm_size=None,                # 自动计算，占据2x2个方格，留1像素边框
        dm_square_size=2,           # DM码占据2x2个方格
        dm_start_position=dm_start_position  # 第一个DM码的起始位置
    )
    
    # 显示生成的DM码位置信息
    dm_positions = generator.generate_dm_positions_by_spacing(
        spacing=dm_spacing, 
        start_position=dm_start_position
    )
    print(f"棋盘格已生成并保存到: {output_path}")
    print(f"配置文件已保存到: {output_path.replace('.png', '_config.json')}")
    print(f"\n共生成 {len(dm_positions)} 个DM码:")
    for dm_id, x, y, row, col in dm_positions:
        # 生成条码内容
        dm_data = generator.generate_coordinate_dm_data(row, col, dm_spacing, square_size)
        print(f"  ID:{dm_id}, 坐标:({x:.1f}, {y:.1f})mm, 位置:第{row+1}行第{col+1}列 (占据2x2方格: 行{row+1}-{row+2}, 列{col+1}-{col+2}), 条码内容: {dm_data}")
    
    return output_path


def main():
    """主函数 - 示例用法"""
    # 简化的接口：只需输入4个参数（可选第5个参数指定起始位置）
    generate_checkerboard_simple(
        square_size=20.0,   # 棋盘格大小：20mm
        rows=25,            # 25行
        cols=25,            # 25列
        dm_spacing=5,       # DM码间距：每隔5个方格放置一个DM码
        dm_start_position=4  # 第一个DM码从第5行第5列开始（可选，如果不指定则使用dm_spacing）
        # output_path 未指定，将自动保存到 output/checkerboard_with_dm.png
    )


if __name__ == "__main__":
    main()

