import cv2
import numpy as np


class Filters:
    Kernels = {
        'Original': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Blur': (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
        'Gaussian blur':(1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Sobel (x)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Sobel (y)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    }

    def __init__(self, kernels=Kernels):
        self.kernels = kernels
        self.filter_names = list(self.kernels.keys())  # 필터 이름 리스트
        self.current_index = 0  # 현재 필터 인덱스
        self.current_filter = self.filter_names[self.current_index]

    def apply_filter(self, frame, filter_name=None) -> np.array:
        if filter_name is None:
            filter_name = self.current_filter
        kernel = self.kernels[filter_name]
        return cv2.filter2D(frame, -1, kernel)

    def get_current_filter_name(self) -> str:
        return self.current_filter

    def switch_next_filter(self):
        self.current_index = (self.current_index + 1) % len(self.filter_names)
        self.current_filter = self.filter_names[self.current_index]

    def switch_previous_filter(self):
        self.current_index = (self.current_index - 1) % len(self.filter_names)
        self.current_filter = self.filter_names[self.current_index]