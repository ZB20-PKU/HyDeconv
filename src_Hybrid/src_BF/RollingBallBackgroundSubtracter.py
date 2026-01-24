import numpy as np
import math
from typing import Optional, Tuple, List
import warnings
from numba import jit, njit, prange, float32, int32
import numba
import time

# =============== 修复版的背景减去器 ===============

class FixedBackgroundSubtracter:
    """修复版背景减去器 - 确保与原始版本结果一致"""
    
    def __init__(self, radius: float = 50.0, light_background: bool = True,
                 create_background: bool = False, use_paraboloid: bool = False,
                 do_presmooth: bool = True, correct_corners: bool = True):
        self.radius = radius
        self.light_background = light_background
        self.create_background = create_background
        self.use_paraboloid = use_paraboloid
        self.do_presmooth = do_presmooth
        self.correct_corners = correct_corners
        
        # 常量
        self.MAXIMUM = 0
        self.MEAN = 1
        self.X_DIRECTION = 0
        self.Y_DIRECTION = 1
        self.DIAGONAL_1A = 2
        self.DIAGONAL_1B = 3
        self.DIAGONAL_2A = 4
        self.DIAGONAL_2B = 5
        self.DIRECTION_PASSES = 9
        
        self.pass_count = 0
        self.n_passes = self.DIRECTION_PASSES
    
    def run(self, image: np.ndarray) -> np.ndarray:
        """运行背景减去算法"""
        if image.dtype != np.float32:
            img = image.astype(np.float32)
        else:
            img = image.copy()
        
        # 假设正常灰度图像，没有反转LUT
        inverted_lut = False
        invert = (inverted_lut and not self.light_background) or (not inverted_lut and self.light_background)
        
        if self.use_paraboloid:
            background = self.sliding_paraboloid_float_background_fixed(
                img, float(self.radius), invert, self.do_presmooth, self.correct_corners
            )
        else:
            # 构建滚动球数据
            ball_radius = self.radius
            if ball_radius <= 10:
                shrink_factor = 1
                arc_trim_per = 24
            elif ball_radius <= 30:
                shrink_factor = 2
                arc_trim_per = 24
            elif ball_radius <= 100:
                shrink_factor = 4
                arc_trim_per = 32
            else:
                shrink_factor = 8
                arc_trim_per = 40
                
            # 构建球体数据
            small_ball_radius = ball_radius / shrink_factor
            if small_ball_radius < 1:
                small_ball_radius = 1
            
            rsquare = small_ball_radius * small_ball_radius
            xtrim = int(arc_trim_per * small_ball_radius / 100)
            half_width = int(round(small_ball_radius - xtrim))
            ball_width = 2 * half_width + 1
            
            # 构建球体数据数组
            ball_data = np.zeros((ball_width, ball_width), dtype=np.float32)
            self.build_ball_data_fixed(ball_data, half_width, rsquare)
            
            background = self.rolling_ball_float_background_fixed(
                img, float(self.radius), invert, self.do_presmooth, 
                ball_data, ball_width, shrink_factor
            )
        
        if self.create_background:
            return background
        else:
            result = img - background
            # 防止NaN和无限值
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            # 确保值在合理范围内
            result = np.clip(result, -np.finfo(np.float32).max, np.finfo(np.float32).max)
            return result
    
    # =============== 滑动抛物面算法 ===============
    
    def sliding_paraboloid_float_background_fixed(self, fp: np.ndarray, radius: float,
                                                 invert: bool, do_presmooth: bool,
                                                 correct_corners: bool) -> np.ndarray:
        """滑动抛物面算法 - 保持与原始版本一致"""
        # 调用原始的正确版本（这部分的加速是正确的）
        height, width = fp.shape
        pixels = fp.ravel().copy()
        
        coeff2 = 0.5 / radius
        coeff2diag = 1.0 / radius
        
        # 反转图像
        if invert:
            pixels = self.invert_array_fixed(pixels)
        
        shift_by = 0.0
        # 预处理平滑
        if do_presmooth:
            fp_2d = pixels.reshape(height, width)
            shift_by = self.filter3x3_fixed(fp_2d, self.MAXIMUM)
            self.filter3x3_fixed(fp_2d, self.MEAN)
            pixels = fp_2d.ravel()
            self.pass_count += 1
        
        # 准备缓存数组
        cache_size = max(width, height)
        cache = np.zeros(cache_size, dtype=np.float32)
        next_point = np.zeros(cache_size, dtype=np.int32)
        
        # 角点校正
        if correct_corners:
            self.correct_corners_fixed(pixels, width, height, coeff2, cache, next_point)
        
        # 9个方向的滤波 - 与原始版本完全一致
        self.filter1D_fixed(pixels, width, height, self.X_DIRECTION, coeff2, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.Y_DIRECTION, coeff2, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.X_DIRECTION, coeff2, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.DIAGONAL_1A, coeff2diag, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.DIAGONAL_1B, coeff2diag, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.DIAGONAL_2A, coeff2diag, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.DIAGONAL_2B, coeff2diag, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.DIAGONAL_1A, coeff2diag, cache, next_point)
        self.filter1D_fixed(pixels, width, height, self.DIAGONAL_1B, coeff2diag, cache, next_point)
        
        # 恢复
        if invert:
            pixels = self.restore_inverted_fixed(pixels, shift_by)
        elif do_presmooth:
            pixels = self.restore_shift_fixed(pixels, shift_by)
        
        result = pixels.reshape(height, width)
        result = np.nan_to_num(result, nan=0.0)
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True, cache=False, boundscheck=False)
    def invert_array_fixed(pixels: np.ndarray) -> np.ndarray:
        """反转数组"""
        for i in prange(len(pixels)):
            pixels[i] = -pixels[i]
        return pixels
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True, cache=False, boundscheck=False)
    def restore_inverted_fixed(pixels: np.ndarray, shift_by: float) -> np.ndarray:
        """恢复反转数组"""
        for i in prange(len(pixels)):
            pixels[i] = -(pixels[i] - shift_by)
        return pixels
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True, cache=False, boundscheck=False)
    def restore_shift_fixed(pixels: np.ndarray, shift_by: float) -> np.ndarray:
        """恢复平移"""
        for i in prange(len(pixels)):
            pixels[i] -= shift_by
        return pixels
    
    def filter1D_fixed(self, pixels: np.ndarray, width: int, height: int,
                      direction: int, coeff2: float, cache: np.ndarray, next_point: np.ndarray):
        """1D滤波"""
        start_line = 0
        n_lines = 0
        line_inc = 0
        point_inc = 0
        
        if direction == self.X_DIRECTION:
            n_lines = height
            line_inc = width
            point_inc = 1
        elif direction == self.Y_DIRECTION:
            n_lines = width
            line_inc = 1
            point_inc = width
        elif direction == self.DIAGONAL_1A:
            n_lines = width - 2
            line_inc = 1
            point_inc = width + 1
        elif direction == self.DIAGONAL_1B:
            start_line = 1
            n_lines = height - 2
            line_inc = width
            point_inc = width + 1
        elif direction == self.DIAGONAL_2A:
            start_line = 2
            n_lines = width
            line_inc = 1
            point_inc = width - 1
        elif direction == self.DIAGONAL_2B:
            start_line = 0
            n_lines = height - 2
            line_inc = width
            point_inc = width - 1
        
        for i in range(start_line, n_lines):
            start_pixel = i * line_inc
            if direction == self.DIAGONAL_2B:
                start_pixel += width - 1
            
            # 确定线长度
            if direction == self.X_DIRECTION:
                length = width
            elif direction == self.Y_DIRECTION:
                length = height
            elif direction == self.DIAGONAL_1A:
                length = min(height, width - i)
            elif direction == self.DIAGONAL_1B:
                length = min(width, height - i)
            elif direction == self.DIAGONAL_2A:
                length = min(height, i + 1)
            elif direction == self.DIAGONAL_2B:
                length = min(width, height - i)
            
            self.line_slide_parabola_fixed(pixels, start_pixel, point_inc, length, coeff2, cache, next_point)
        
        self.pass_count += 1
    
    @staticmethod
    @jit(nopython=True, fastmath=True, cache=False, boundscheck=False)
    def line_slide_parabola_fixed(pixels: np.ndarray, start: int, inc: int, length: int,
                                 coeff2: float, cache: np.ndarray, next_point: np.ndarray):
        """滑动抛物线算法"""
        if length <= 0:
            return
        
        min_value = np.finfo(np.float32).max
        lastpoint = 0
        curvature_test = 1.999 * coeff2
        
        p = start
        v_previous1 = 0.0
        v_previous2 = 0.0
        
        for i in range(length):
            v = pixels[p]
            cache[i] = v
            if v < min_value:
                min_value = v
            
            if i >= 2 and v_previous1 + v_previous1 - v_previous2 - v < curvature_test:
                next_point[lastpoint] = i - 1
                lastpoint = i - 1
            
            v_previous2 = v_previous1
            v_previous1 = v
            p += inc
        
        next_point[lastpoint] = length - 1
        next_point[length - 1] = np.iinfo(np.int32).max
        
        # 主循环
        i1 = 0
        while i1 < length - 1:
            v1 = cache[i1]
            min_slope = np.finfo(np.float32).max
            i2 = i1 + 1
            search_to = length
            recalculate_limit_now = 0
            
            # 找到第二个接触点
            j = next_point[i1]
            while j < search_to:
                v2 = cache[j]
                slope = (v2 - v1) / (j - i1) + coeff2 * (j - i1)
                
                if slope < min_slope:
                    min_slope = slope
                    i2 = j
                    recalculate_limit_now = -3
                
                if recalculate_limit_now == 0:
                    b = 0.5 * min_slope / coeff2
                    b_sq = b * b
                    term = (v1 - min_value) / coeff2
                    if b_sq + term >= 0:
                        max_search = i1 + int(b + math.sqrt(b_sq + term) + 1)
                        if 0 < max_search < search_to:
                            search_to = max_search
                
                recalculate_limit_now += 1
                if j == length - 1:
                    break
                j = next_point[j]
            
            # 在两个接触点之间插值
            for j in range(i1 + 1, i2):
                cache[j] = v1 + (j - i1) * (min_slope - (j - i1) * coeff2)
            
            i1 = i2
        
        # 将结果写回原数组
        p = start
        for i in range(length):
            pixels[p] = cache[i]
            p += inc
    
    def correct_corners_fixed(self, pixels: np.ndarray, width: int, height: int,
                             coeff2: float, cache: np.ndarray, next_point: np.ndarray):
        """角点校正"""
        corners = np.zeros(4, dtype=np.float32)
        corrected_edges = np.zeros(2, dtype=np.float32)
        
        # 顶边
        self.line_slide_parabola_java_wrapper_fixed(pixels, 0, 1, width, coeff2, cache, next_point, corrected_edges)
        corners[0] = corrected_edges[0]
        corners[1] = corrected_edges[1]
        
        # 底边
        self.line_slide_parabola_java_wrapper_fixed(pixels, (height - 1) * width, 1, width, coeff2, cache, next_point, corrected_edges)
        corners[2] = corrected_edges[0]
        corners[3] = corrected_edges[1]
        
        # 左边
        self.line_slide_parabola_java_wrapper_fixed(pixels, 0, width, height, coeff2, cache, next_point, corrected_edges)
        corners[0] += corrected_edges[0]
        corners[2] += corrected_edges[1]
        
        # 右边
        self.line_slide_parabola_java_wrapper_fixed(pixels, width - 1, width, height, coeff2, cache, next_point, corrected_edges)
        corners[1] += corrected_edges[0]
        corners[3] += corrected_edges[1]
        
        # 对角线
        diag_length = min(width, height)
        coeff2diag = 2 * coeff2
        
        # 对角线1
        self.line_slide_parabola_java_wrapper_fixed(pixels, 0, width + 1, diag_length, coeff2diag, cache, next_point, corrected_edges)
        corners[0] += corrected_edges[0]
        
        # 对角线2
        self.line_slide_parabola_java_wrapper_fixed(pixels, width - 1, width - 1, diag_length, coeff2diag, cache, next_point, corrected_edges)
        corners[1] += corrected_edges[0]
        
        # 对角线3
        self.line_slide_parabola_java_wrapper_fixed(pixels, (height - 1) * width, 1 - width, diag_length, coeff2diag, cache, next_point, corrected_edges)
        corners[2] += corrected_edges[0]
        
        # 对角线4
        self.line_slide_parabola_java_wrapper_fixed(pixels, width * height - 1, -width - 1, diag_length, coeff2diag, cache, next_point, corrected_edges)
        corners[3] += corrected_edges[0]
        
        # 应用角点校正
        self.apply_corner_correction_fixed(pixels, width, height, corners)
    
    @staticmethod
    @jit(nopython=True, cache=False, boundscheck=False)
    def line_slide_parabola_java_wrapper_fixed(pixels: np.ndarray, start: int, inc: int, length: int,
                                              coeff2: float, cache: np.ndarray, next_point: np.ndarray,
                                              corrected_edges: np.ndarray):
        """包装函数，用于与Java版本兼容的角点校正"""
        if length <= 0:
            return
        
        min_value = np.finfo(np.float32).max
        lastpoint = 0
        first_corner = length - 1
        last_corner = 0
        v_previous1 = 0.0
        v_previous2 = 0.0
        curvature_test = 1.999 * coeff2
        
        p = start
        for i in range(length):
            v = pixels[p]
            cache[i] = v
            if v < min_value:
                min_value = v
            
            if i >= 2 and v_previous1 + v_previous1 - v_previous2 - v < curvature_test:
                next_point[lastpoint] = i - 1
                lastpoint = i - 1
            
            v_previous2 = v_previous1
            v_previous1 = v
            p += inc
        
        next_point[lastpoint] = length - 1
        next_point[length - 1] = np.iinfo(np.int32).max
        
        # 主循环
        i1 = 0
        while i1 < length - 1:
            v1 = cache[i1]
            min_slope = np.finfo(np.float32).max
            i2 = i1 + 1
            search_to = length
            recalculate_limit_now = 0
            
            # 找到第二个接触点
            j = next_point[i1]
            while j < search_to:
                v2 = cache[j]
                slope = (v2 - v1) / (j - i1) + coeff2 * (j - i1)
                
                if slope < min_slope:
                    min_slope = slope
                    i2 = j
                    recalculate_limit_now = -3
                
                if recalculate_limit_now == 0:
                    b = 0.5 * min_slope / coeff2
                    b_sq = b * b
                    term = (v1 - min_value) / coeff2
                    if b_sq + term >= 0:
                        max_search = i1 + int(b + math.sqrt(b_sq + term) + 1)
                        if 0 < max_search < search_to:
                            search_to = max_search
                
                recalculate_limit_now += 1
                if j == length - 1:
                    break
                j = next_point[j]
            
            if i1 == 0:
                first_corner = i2
            if i2 == length - 1:
                last_corner = i1
            
            # 在两个接触点之间插值
            for j in range(i1 + 1, i2):
                cache[j] = v1 + (j - i1) * (min_slope - (j - i1) * coeff2)
            
            i1 = i2
        
        # 将结果写回原数组
        p = start
        for i in range(length):
            pixels[p] = cache[i]
            p += inc
        
        # 计算校正后的边缘值
        if 4 * first_corner >= length:
            first_corner = 0
        if 4 * (length - 1 - last_corner) >= length:
            last_corner = length - 1
        
        v1 = cache[first_corner]
        v2 = cache[last_corner]
        slope = (v2 - v1) / (last_corner - first_corner)
        value0 = v1 - slope * first_corner
        coeff6 = 0.0
        mid = 0.5 * (last_corner + first_corner)
        
        # 与图像中部像素比较以检测渐晕
        for i in range((length + 2) // 3, (2 * length) // 3 + 1):
            dx = (i - mid) * 2.0 / (last_corner - first_corner)
            poly6 = dx * dx * dx * dx * dx * dx - 1.0
            if cache[i] < value0 + slope * i + coeff6 * poly6:
                coeff6 = -(value0 + slope * i - cache[i]) / poly6
        
        dx = (first_corner - mid) * 2.0 / (last_corner - first_corner)
        corrected_edges[0] = value0 + coeff6 * (dx ** 6 - 1.0) + coeff2 * first_corner * first_corner
        
        dx = (last_corner - mid) * 2.0 / (last_corner - first_corner)
        corrected_edges[1] = (value0 + (length - 1) * slope + 
                            coeff6 * (dx ** 6 - 1.0) + 
                            coeff2 * (length - 1 - last_corner) * (length - 1 - last_corner))
    
    @staticmethod
    @jit(nopython=True, cache=False, boundscheck=False)
    def apply_corner_correction_fixed(pixels: np.ndarray, width: int, height: int, corners: np.ndarray):
        """应用角点校正"""
        if pixels[0] > corners[0] / 3:
            pixels[0] = corners[0] / 3
        if pixels[width - 1] > corners[1] / 3:
            pixels[width - 1] = corners[1] / 3
        if pixels[(height - 1) * width] > corners[2] / 3:
            pixels[(height - 1) * width] = corners[2] / 3
        if pixels[width * height - 1] > corners[3] / 3:
            pixels[width * height - 1] = corners[3] / 3
    
    def filter3x3_fixed(self, fp: np.ndarray, filter_type: int) -> float:
        """3x3滤波器"""
        height, width = fp.shape
        pixels = fp.ravel()
        shift_by = 0.0
        
        # 处理所有行
        for y in prange(height):
            start = y * width
            shift_by += self.filter3_fixed(pixels, width, start, 1, filter_type)
        
        # 处理所有列
        for x in prange(width):
            shift_by += self.filter3_fixed(pixels, height, x, width, filter_type)
        
        return shift_by / (width * height)
    
    @staticmethod
    @jit(nopython=True, cache=False, boundscheck=False)
    def filter3_fixed(pixels: np.ndarray, length: int, pixel0: int, inc: int, filter_type: int) -> float:
        """处理一行或一列"""
        shift_by = 0.0
        v3 = pixels[pixel0]
        v2 = v3
        v1 = v2
        
        p = pixel0
        for i in range(length):
            v1 = v2
            v2 = v3
            if i < length - 1:
                v3 = pixels[p + inc]
            
            if filter_type == 0:
                max_val = max(v1, v2, v3)
                shift_by += max_val - pixels[p]
                pixels[p] = max_val
            else:
                pixels[p] = (v1 + v2 + v3) * 0.33333333
            
            p += inc
        
        return shift_by
    
    # =============== 滚动球算法 - 修复版 ===============
    
    def rolling_ball_float_background_fixed(self, fp: np.ndarray, radius: float,
                                           invert: bool, do_presmooth: bool,
                                           ball_data: np.ndarray, ball_width: int,
                                           shrink_factor: int) -> np.ndarray:
        """滚动球算法 - 修复版，确保与原始版本一致"""
        height, width = fp.shape
        pixels = fp.ravel().copy()
        
        shrink = shrink_factor > 1
        
        if invert:
            pixels = self.invert_array_fixed(pixels)
        
        if do_presmooth:
            fp_2d = pixels.reshape(height, width)
            self.filter3x3_fixed(fp_2d, self.MEAN)
            pixels = fp_2d.ravel()
        
        # 缩小图像
        if shrink:
            small_image = self.shrink_image_fixed(pixels.reshape(height, width), shrink_factor)
        else:
            small_image = pixels.reshape(height, width)
        
        # 滚动球 - 使用修复版的算法
        self.roll_ball_fixed(ball_data, ball_width, small_image)
        
        # 放大图像
        if shrink:
            background = self.enlarge_image_fixed(small_image, (height, width), shrink_factor)
        else:
            background = small_image
        
        if invert:
            background = -background
        
        self.pass_count += 1
        return background
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True, cache=False, boundscheck=False)
    def shrink_image_fixed(ip: np.ndarray, shrink_factor: int) -> np.ndarray:
        """缩小图像"""
        height, width = ip.shape
        s_width = (width + shrink_factor - 1) // shrink_factor
        s_height = (height + shrink_factor - 1) // shrink_factor
        
        small_image = np.zeros((s_height, s_width), dtype=np.float32)
        
        for y_small in prange(s_height):
            for x_small in range(s_width):
                min_val = np.finfo(np.float32).max
                y = shrink_factor * y_small
                for j in range(shrink_factor):
                    if y >= height:
                        break
                    x = shrink_factor * x_small
                    for k in range(shrink_factor):
                        if x >= width:
                            break
                        this_pixel = ip[y, x]
                        if this_pixel < min_val:
                            min_val = this_pixel
                        x += 1
                    y += 1
                
                if min_val == np.finfo(np.float32).max:
                    min_val = 0.0
                small_image[y_small, x_small] = min_val
        
        return small_image
    
    @staticmethod
    @jit(nopython=True, fastmath=True, cache=False, boundscheck=False)
    def roll_ball_fixed(ball_data: np.ndarray, ball_width: int, fp: np.ndarray):
        """修复版的滚动球算法 - 确保与原始版本完全一致"""
        height, width = fp.shape
        z_ball = ball_data.ravel()
        radius = ball_width // 2
        
        # 保存原始像素的最小值
        original_min = np.min(fp)
        
        # 初始化缓存
        cache = np.zeros(ball_width * width, dtype=np.float32)
        
        # 将输出初始化为负无穷大
        float_min = -np.finfo(np.float32).max
        output = np.full_like(fp, float_min)
        
        # 主循环 - 使用串行执行确保与原始版本一致
        # 注意：这里不使用并行，因为原始版本是串行的，并行会导致执行顺序不同
        for y_center in range(-radius, height + radius):
            next_line_to_write_in_cache = (y_center + radius) % ball_width
            next_line_to_read = y_center + radius
            
            # 读取新行到缓存
            if 0 <= next_line_to_read < height:
                row_start = next_line_to_write_in_cache * width
                for i in range(width):
                    cache[row_start + i] = fp[next_line_to_read, i]
                # 将原图像中的行设置为负无穷大
                for i in range(width):
                    fp[next_line_to_read, i] = float_min
            
            y0 = max(y_center - radius, 0)
            y_end = min(y_center + radius, height - 1)
            
            # 重要：这里使用串行循环，不使用prange，确保执行顺序与原始版本一致
            for x_center in range(-radius, width + radius):
                z = np.finfo(np.float32).max
                x0 = max(x_center - radius, 0)
                x_end = min(x_center + radius, width - 1)
                
                # 计算球在此位置的高度
                for yp in range(y0, y_end + 1):
                    y_ball = yp - y_center + radius
                    cache_ptr = (yp % ball_width) * width + x0
                    for xp in range(x0, x_end + 1):
                        bp = (xp - x_center + radius) + y_ball * ball_width
                        if 0 <= bp < len(z_ball):
                            cache_val = cache[cache_ptr]
                            z_reduced = cache_val - z_ball[bp]
                            if z_reduced < z:
                                z = z_reduced
                        cache_ptr += 1
                
                # 将球内像素提升到球表面
                for yp in range(y0, y_end + 1):
                    y_ball = yp - y_center + radius
                    for xp in range(x0, x_end + 1):
                        bp = (xp - x_center + radius) + y_ball * ball_width
                        if 0 <= bp < len(z_ball):
                            z_min = z + z_ball[bp]
                            # 使用原子操作确保线程安全（尽管现在是串行）
                            if output[yp, xp] < z_min:
                                output[yp, xp] = z_min
        
        # 恢复原始数据
        for y in range(height):
            for x in range(width):
                fp[y, x] = output[y, x]
        
        # 处理未处理的像素
        for y in range(height):
            for x in range(width):
                if fp[y, x] == float_min:
                    fp[y, x] = original_min
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True, cache=False, boundscheck=False)
    def enlarge_image_fixed(small_image: np.ndarray, target_shape: Tuple[int, int],
                           shrink_factor: int) -> np.ndarray:
        """放大图像"""
        height, width = target_shape
        small_height, small_width = small_image.shape
        
        # 创建插值数组
        x_small_indices = np.zeros(width, dtype=np.int32)
        x_weights = np.zeros(width, dtype=np.float32)
        y_small_indices = np.zeros(height, dtype=np.int32)
        y_weights = np.zeros(height, dtype=np.float32)
        
        # 计算x方向的插值参数
        for i in range(width):
            small_index = (i - shrink_factor // 2) // shrink_factor
            if small_index >= small_width - 1:
                small_index = small_width - 2
            if small_index < 0:
                small_index = 0
            x_small_indices[i] = small_index
            distance = (i + 0.5) / shrink_factor - (small_index + 0.5)
            x_weights[i] = 1.0 - distance
        
        # 计算y方向的插值参数
        for i in range(height):
            small_index = (i - shrink_factor // 2) // shrink_factor
            if small_index >= small_height - 1:
                small_index = small_height - 2
            if small_index < 0:
                small_index = 0
            y_small_indices[i] = small_index
            distance = (i + 0.5) / shrink_factor - (small_index + 0.5)
            y_weights[i] = 1.0 - distance
        
        # 双线性插值
        result = np.zeros(target_shape, dtype=np.float32)
        
        # 对每个像素进行插值
        for y in prange(height):
            y_small0 = y_small_indices[y]
            y_small1 = min(y_small0 + 1, small_height - 1)
            y_weight = y_weights[y]
            
            for x in range(width):
                x_small0 = x_small_indices[x]
                x_small1 = min(x_small0 + 1, small_width - 1)
                x_weight = x_weights[x]
                
                # 获取四个相邻像素
                v00 = small_image[y_small0, x_small0]
                v01 = small_image[y_small0, x_small1]
                v10 = small_image[y_small1, x_small0]
                v11 = small_image[y_small1, x_small1]
                
                # 双线性插值
                result[y, x] = (v00 * x_weight * y_weight +
                              v01 * (1.0 - x_weight) * y_weight +
                              v10 * x_weight * (1.0 - y_weight) +
                              v11 * (1.0 - x_weight) * (1.0 - y_weight))
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True, cache=False, boundscheck=False)
    def build_ball_data_fixed(data: np.ndarray, half_width: int, rsquare: float):
        """构建球体数据"""
        width = data.shape[0]
        for y in prange(width):
            yval = y - half_width
            for x in range(width):
                xval = x - half_width
                temp = rsquare - xval * xval - yval * yval
                if temp > 0:
                    data[y, x] = math.sqrt(temp)
                else:
                    data[y, x] = 0.0


# =============== 简化接口 ===============

def subtract_background_fixed(image: np.ndarray, radius: float = 50.0,
                              light_background: bool = True,
                              create_background: bool = False,
                              use_paraboloid: bool = False,
                              do_presmooth: bool = True,
                              correct_corners: bool = True) -> np.ndarray:
    """
    背景减去函数的修复接口
    """
    subtracter = FixedBackgroundSubtracter(
        radius=radius,
        light_background=light_background,
        create_background=create_background,
        use_paraboloid=use_paraboloid,
        do_presmooth=do_presmooth,
        correct_corners=correct_corners
    )
    
    return subtracter.run(image)


# =============== 测试代码 ===============

if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings('ignore')
    
    # # 创建测试图像
    # height, width = 512, 512
    # y, x = np.mgrid[0:height, 0:width]
    
    # # 创建包含点状结构的测试图像
    # background = 100 + 50 * np.sin(x/20) * np.cos(y/20)
    # # 添加一些点状结构
    # foreground = np.zeros((height, width), dtype=np.float32)
    # for _ in range(50):
    #     cx = np.random.randint(50, width-50)
    #     cy = np.random.randint(50, height-50)
    #     intensity = np.random.uniform(20, 50)
    #     size = np.random.uniform(3, 8)
    #     foreground += intensity * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*size**2))
    
    # test_image = background + foreground
    # test_image = test_image.astype(np.float32)
    
    # print(f"测试图像:")
    # print(f"  形状: {test_image.shape}")
    # print(f"  范围: [{test_image.min():.2f}, {test_image.max():.2f}]")
    # print(f"  均值: {test_image.mean():.2f}")
    
    # # 导入原始版本用于对比
    # print("\n1. 导入原始版本用于对比...")
    
    # # 创建原始版本的实例（使用相同的参数）
    # subtracter_original = BackgroundSubtracter(
    #     radius=30, 
    #     light_background=True, 
    #     create_background=False,
    #     use_paraboloid=False,
    #     do_presmooth=True,
    #     correct_corners=False
    # )
    
    # # 测试原始版本
    # print("\n2. 测试原始版本 (滚动球算法):")
    # start_time = time.time()
    # result_original = subtracter_original.run(test_image)
    # elapsed_time_original = time.time() - start_time
    # print(f"  执行时间: {elapsed_time_original:.3f}秒")
    # print(f"  结果范围: [{result_original.min():.2f}, {result_original.max():.2f}]")
    
    # # 测试修复版本
    # print("\n3. 测试修复版本 (滚动球算法):")
    # start_time = time.time()
    # result_fixed = subtract_background_fixed(
    #     test_image,
    #     radius=30,
    #     light_background=True,
    #     use_paraboloid=False,
    #     do_presmooth=True,
    #     correct_corners=False
    # )
    # elapsed_time_fixed = time.time() - start_time
    # print(f"  执行时间: {elapsed_time_fixed:.3f}秒")
    # print(f"  结果范围: [{result_fixed.min():.2f}, {result_fixed.max():.2f}]")
    
    # # 比较结果
    # print(f"\n结果比较:")
    # print(f"  原始版本时间: {elapsed_time_original:.3f}秒")
    # print(f"  修复版本时间: {elapsed_time_fixed:.3f}秒")
    # print(f"  加速比: {elapsed_time_original/elapsed_time_fixed:.2f}x")
    
    # # 计算误差
    # diff = np.abs(result_original - result_fixed)
    # print(f"  最大绝对误差: {diff.max():.6f}")
    # print(f"  平均绝对误差: {diff.mean():.6f}")
    # print(f"  均方根误差: {np.sqrt(np.mean(diff**2)):.6f}")
    
    # # 检查点状结构的差异
    # print(f"\n点状结构检查:")
    # # 找出差异最大的区域
    # max_diff_pos = np.unravel_index(np.argmax(diff), diff.shape)
    # print(f"  最大差异位置: {max_diff_pos}")
    # print(f"  原始值: {result_original[max_diff_pos]:.4f}")
    # print(f"  修复值: {result_fixed[max_diff_pos]:.4f}")
    # print(f"  差异: {diff[max_diff_pos]:.4f}")
    
    # # 检查差异超过阈值的像素比例
    # threshold = 1e-5  # 使用更严格的阈值
    # large_diff_pixels = np.sum(diff > threshold)
    # total_pixels = diff.size
    # print(f"  差异 > {threshold} 的像素比例: {large_diff_pixels/total_pixels*100:.4f}%")
    
    # if large_diff_pixels == 0:
    #     print("  ✅ 所有像素差异都在可接受范围内")
    # else:
    #     print(f"  ⚠️  有 {large_diff_pixels} 个像素差异超过阈值")
    
    # # 可视化差异
    # import matplotlib.pyplot as plt
    
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # # 原始图像
    # axes[0, 0].imshow(test_image, cmap='gray')
    # axes[0, 0].set_title('原始图像')
    # axes[0, 0].axis('off')
    
    # # 原始版本结果
    # axes[0, 1].imshow(result_original, cmap='gray')
    # axes[0, 1].set_title('原始版本结果')
    # axes[0, 1].axis('off')
    
    # # 修复版本结果
    # axes[0, 2].imshow(result_fixed, cmap='gray')
    # axes[0, 2].set_title('修复版本结果')
    # axes[0, 2].axis('off')
    
    # # 差异图
    # im = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=0.01)
    # axes[1, 0].set_title('差异图 (放大显示)')
    # axes[1, 0].axis('off')
    # plt.colorbar(im, ax=axes[1, 0])
    
    # # 差异直方图
    # axes[1, 1].hist(diff.ravel(), bins=50, log=True)
    # axes[1, 1].set_title('差异直方图 (log scale)')
    # axes[1, 1].set_xlabel('差异值')
    # axes[1, 1].set_ylabel('频率')
    # axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'阈值={threshold}')
    # axes[1, 1].legend()
    
    # # 点状结构区域放大
    # y_start = max(0, max_diff_pos[0] - 20)
    # y_end = min(height, max_diff_pos[0] + 20)
    # x_start = max(0, max_diff_pos[1] - 20)
    # x_end = min(width, max_diff_pos[1] + 20)
    
    # axes[1, 2].imshow(diff[y_start:y_end, x_start:x_end], cmap='hot')
    # axes[1, 2].set_title('最大差异区域放大')
    # axes[1, 2].axis('off')
    
    # plt.suptitle('滚动球算法修复结果对比', fontsize=16)
    # plt.tight_layout()
    # plt.show()