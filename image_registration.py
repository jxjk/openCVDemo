# -*- coding: utf-8 -*-
"""
图像配准模块
Image Registration Module

功能:
- 图纸与图像配准
- 特征点检测和匹配
- 变换矩阵计算
- 坐标系转换
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

from logger_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class TransformationMatrix:
    """变换矩阵"""
    matrix: np.ndarray  # 3x3变换矩阵
    type: str  # 'translation', 'rotation', 'similarity', 'affine', 'homography'
    confidence: float = 0.0  # 配准置信度
    inliers: int = 0  # 内点数


@dataclass
class MatchResult:
    """匹配结果"""
    image_points: List[cv2.KeyPoint]  # 图像中的关键点
    template_points: List[cv2.KeyPoint]  # 模板中的关键点
    matches: List[cv2.DMatch]  # 匹配点对
    transformation: Optional[TransformationMatrix] = None  # 变换矩阵
    success: bool = False  # 是否成功
    message: str = ""  # 消息


# =============================================================================
# 图像配准器类
# =============================================================================

class ImageRegistration:
    """图像配准器"""
    
    def __init__(self, method: str = 'ORB', min_matches: int = 10,
                 ratio_threshold: float = 0.7):
        """
        初始化图像配准器
        
        Args:
            method: 特征检测方法 ('ORB', 'SIFT', 'SURF', 'AKAZE')
            min_matches: 最小匹配点数
            ratio_threshold: 比率阈值（用于筛选匹配点）
        """
        self.method = method
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        
        # 初始化特征检测器
        self.detector = self._create_detector(method)
        
        # 初始化特征匹配器
        self.matcher = self._create_matcher(method)
    
    def _create_detector(self, method: str):
        """
        创建特征检测器
        
        Args:
            method: 检测方法
        
        Returns:
            特征检测器
        """
        if method == 'ORB':
            return cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_FAST_SCORE)
        elif method == 'SIFT':
            try:
                return cv2.SIFT_create(nfeatures=5000)
            except AttributeError:
                logger.warning("SIFT不可用，使用ORB替代")
                return cv2.ORB_create(nfeatures=5000)
        elif method == 'SURF':
            try:
                return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            except AttributeError:
                logger.warning("SURF不可用，使用ORB替代")
                return cv2.ORB_create(nfeatures=5000)
        elif method == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            logger.warning(f"未知方法 {method}，使用ORB")
            return cv2.ORB_create(nfeatures=5000)
    
    def _create_matcher(self, method: str):
        """
        创建特征匹配器
        
        Args:
            method: 检测方法
        
        Returns:
            特征匹配器
        """
        if method in ['SIFT', 'SURF']:
            # 使用FLANN匹配器
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # 使用BF匹配器
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def register(self, image: np.ndarray, template: np.ndarray,
                 method: str = 'homography') -> MatchResult:
        """
        注册图像到模板
        
        Args:
            image: 输入图像
            template: 模板图像（从DXF渲染得到）
            method: 变换方法 ('homography', 'affine', 'similarity', 'translation')
        
        Returns:
            匹配结果
        """
        # 检查输入
        if image is None or template is None:
            return MatchResult(
                image_points=[],
                template_points=[],
                matches=[],
                success=False,
                message="输入图像或模板为空"
            )
        
        # 检测特征点
        kp1, des1 = self.detector.detectAndCompute(template, None)
        kp2, des2 = self.detector.detectAndCompute(image, None)
        
        if des1 is None or des2 is None:
            return MatchResult(
                image_points=[],
                template_points=[],
                matches=[],
                success=False,
                message="无法检测到足够的特征点"
            )
        
        # 特征匹配
        if self.method in ['SIFT', 'SURF']:
            # FLANN匹配
            matches = self.matcher.knnMatch(des1, des2, k=2)
            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        else:
            # BF匹配
            matches = self.matcher.knnMatch(des1, des2, k=2)
            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # 检查匹配点数量
        if len(good_matches) < self.min_matches:
            return MatchResult(
                image_points=kp2,
                template_points=kp1,
                matches=good_matches,
                success=False,
                message=f"匹配点不足 ({len(good_matches)} < {self.min_matches})"
            )
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算变换矩阵
        transformation = self._compute_transformation(src_pts, dst_pts, method)
        
        if transformation is None:
            return MatchResult(
                image_points=kp2,
                template_points=kp1,
                matches=good_matches,
                success=False,
                message="无法计算变换矩阵"
            )
        
        logger.info(f"配准成功: 方法={method}, 匹配点={len(good_matches)}, "
                   f"置信度={transformation.confidence:.2f}, 内点={transformation.inliers}")
        
        return MatchResult(
            image_points=kp2,
            template_points=kp1,
            matches=good_matches,
            transformation=transformation,
            success=True,
            message=f"配准成功，匹配点数: {len(good_matches)}"
        )
    
    def _compute_transformation(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                               method: str) -> Optional[TransformationMatrix]:
        """
        计算变换矩阵
        
        Args:
            src_pts: 源点集
            dst_pts: 目标点集
            method: 变换方法
        
        Returns:
            变换矩阵
        """
        if method == 'homography':
            # 单应性变换
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return None
            inliers = np.sum(mask)
            confidence = inliers / len(src_pts)
            return TransformationMatrix(M, 'homography', confidence, inliers)
        
        elif method == 'affine':
            # 仿射变换
            M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
            if M is None:
                return None
            # 转换为3x3矩阵
            M = np.vstack([M, [0, 0, 1]])
            inliers = np.sum(mask)
            confidence = inliers / len(src_pts)
            return TransformationMatrix(M, 'affine', confidence, inliers)
        
        elif method == 'similarity':
            # 相似变换
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            if M is None:
                return None
            # 转换为3x3矩阵
            M = np.vstack([M, [0, 0, 1]])
            inliers = np.sum(mask)
            confidence = inliers / len(src_pts)
            return TransformationMatrix(M, 'similarity', confidence, inliers)
        
        elif method == 'translation':
            # 纯平移
            src_center = np.mean(src_pts, axis=0)
            dst_center = np.mean(dst_pts, axis=0)
            tx = dst_center[0, 0] - src_center[0, 0]
            ty = dst_center[0, 1] - src_center[0, 1]
            M = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            return TransformationMatrix(M, 'translation', 1.0, len(src_pts))
        
        else:
            logger.warning(f"未知变换方法: {method}，使用单应性变换")
            return self._compute_transformation(src_pts, dst_pts, 'homography')
    
    def transform_point(self, point: Tuple[float, float],
                       transformation: TransformationMatrix) -> Tuple[float, float]:
        """
        应用变换矩阵到点
        
        Args:
            point: 输入点 (x, y)
            transformation: 变换矩阵
        
        Returns:
            变换后的点 (x, y)
        """
        # 转换为齐次坐标
        pt = np.array([point[0], point[1], 1.0])
        
        # 应用变换
        transformed = transformation.matrix @ pt
        
        # 返回非齐次坐标
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    def transform_points(self, points: List[Tuple[float, float]],
                        transformation: TransformationMatrix) -> List[Tuple[float, float]]:
        """
        应用变换矩阵到点集
        
        Args:
            points: 输入点列表
            transformation: 变换矩阵
        
        Returns:
            变换后的点列表
        """
        return [self.transform_point(pt, transformation) for pt in points]
    
    def draw_matches(self, image: np.ndarray, template: np.ndarray,
                    match_result: MatchResult) -> np.ndarray:
        """
        绘制匹配结果
        
        Args:
            image: 输入图像
            template: 模板图像
            match_result: 匹配结果
        
        Returns:
            可视化图像
        """
        # 调整图像大小以便显示
        h1, w1 = template.shape[:2]
        h2, w2 = image.shape[:2]
        
        # 使用前50个最佳匹配
        matches = match_result.matches[:50]
        
        # 绘制匹配
        if self.method in ['SIFT', 'SURF']:
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=None,
                flags=2
            )
            result = cv2.drawMatches(
                template, match_result.template_points,
                image, match_result.image_points,
                matches, None, **draw_params
            )
        else:
            result = cv2.drawMatches(
                template, match_result.template_points,
                image, match_result.image_points,
                matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=2
            )
        
        return result


# =============================================================================
# 便捷函数
# =============================================================================

def register_images(image: np.ndarray, template: np.ndarray,
                   method: str = 'homography') -> Optional[TransformationMatrix]:
    """
    便捷函数：注册图像
    
    Args:
        image: 输入图像
        template: 模板图像
        method: 变换方法
    
    Returns:
        变换矩阵，失败返回None
    """
    reg = ImageRegistration()
    result = reg.register(image, template, method)
    return result.transformation if result.success else None


# =============================================================================
# 主函数（用于测试）
# =============================================================================

if __name__ == '__main__':
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("用法:")
        print("  python image_registration.py <模板图像> <输入图像> [变换方法] [输出图像]")
        print("\n变换方法:")
        print("  homography - 单应性变换（默认）")
        print("  affine - 仿射变换")
        print("  similarity - 相似变换")
        print("  translation - 纯平移")
        print("\n示例:")
        print("  python image_registration.py template.jpg image.jpg")
        print("  python image_registration.py template.jpg image.jpg homography result.jpg")
        sys.exit(1)
    
    template_path = sys.argv[1]
    image_path = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'homography'
    output_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    print(f"模板图像: {template_path}")
    print(f"输入图像: {image_path}")
    print(f"变换方法: {method}")
    print()
    
    # 读取图像
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None or image is None:
        print("错误: 无法读取图像")
        sys.exit(1)
    
    # 执行配准
    reg = ImageRegistration(method=method)
    result = reg.register(image, template, method)
    
    # 打印结果
    print("=" * 60)
    print("配准结果")
    print("=" * 60)
    print(f"状态: {'成功' if result.success else '失败'}")
    print(f"消息: {result.message}")
    
    if result.success and result.transformation:
        print(f"变换类型: {result.transformation.type}")
        print(f"置信度: {result.transformation.confidence:.2f}")
        print(f"内点数: {result.transformation.inliers}")
        print(f"变换矩阵:")
        print(result.transformation.matrix)
        
        # 绘制匹配结果
        matches_img = reg.draw_matches(image, template, result)
        
        if output_path:
            cv2.imwrite(output_path, matches_img)
            print(f"\n匹配结果已保存到: {output_path}")
        
        # 显示结果（如果有窗口支持）
        try:
            cv2.imshow('Matches', matches_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    else:
        sys.exit(1)
