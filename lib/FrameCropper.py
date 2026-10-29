import cv2
import numpy as np
import pygame

class FrameCropper:
    def __init__(self, src_w, src_h, window_w, window_h, display_w, display_h):
        self.src_w = src_w
        self.src_h = src_h
        self.window_w = window_w
        self.window_h = window_h
        self.display_w = display_w
        self.display_h = display_h

        self.crop_area = self.crop_infinite_area

        self.current_X = 0
        self.current_Y = 0
        self.current_Z = 0.7

        self.movement_lock = 2
        self.movement_release = 4
        self.active_movement = False
        self.first_movement = True

        self.reduce_K = 0.12
        self.ASPECT_RATIO = 9 / 16
        self.min_z = 0.1

        self.max_tilt = 650
        self.x_range = 800
        self.perspective_smooth = 0.35
        self.current_tilt = 0
        self.perspective_enabled = True

    def set_zoom_scale(self, K):
        self.reduce_K = K

    def set_zoom_value(self, aspect_ratio=0.0577, min_z=0.1):
        self.ASPECT_RATIO = aspect_ratio
        self.min_z = min_z

    def calculate_resolution(self, z):
        scale_factor = (self.min_z / z) ** self.reduce_K
        new_width = int(self.window_w * scale_factor)
        new_height = int(new_width * self.ASPECT_RATIO)
        return new_width, new_height

    def run(self, frame, center_x, center_y, z):
        # 화면 위치는 절대 움직이지 않게 고정
        cropped_frame = self.crop_area(frame, 0, 0, z)

        # center_x는 화면 이동이 아니라 3D 기울기 값으로만 사용
        if self.perspective_enabled:
                image = self.apply_perspective(cropped_frame, center_x)
        else:
                image = pygame.transform.scale(cropped_frame, (self.display_w, self.display_h))

        return image

    def pygame_surface_to_cv2(self, surface):
        arr = pygame.surfarray.array3d(surface)
        arr = arr.swapaxes(0, 1)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def cv2_to_pygame_surface(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

    def apply_perspective(self, surface, center_x):
        surface = pygame.transform.scale( surface,(int(self.display_w * 0.8),int(self.display_h * 0.8)))
        frame = self.pygame_surface_to_cv2(surface)

        h, w = frame.shape[:2]

        ratio = max(-1.0, min(1.0, center_x / self.x_range))
        target_tilt = ratio * self.max_tilt

        self.current_tilt += (target_tilt - self.current_tilt) * self.perspective_smooth
        tilt = int(self.current_tilt)

        src = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
        ])

        if tilt > 0:
                t = min(abs(tilt), w // 4)
                v = int(t * 0.2)

                dst = np.float32([
                        [t, v],
                        [w - t, 0],
                        [w - t, h],
                        [t, h - v]
                ])

        elif tilt < 0:
                t = min(abs(tilt), w // 4)
                v = int(t * 0.2)

                dst = np.float32([
                        [t, 0],
                        [w - t, v],
                        [w - t, h - v],
                        [t, h]
                ])

        else:
                dst = src.copy()

        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
                frame,
                matrix,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
        )

        return self.cv2_to_pygame_surface(warped)

    def crop_infinite_area(self, frame, cx, cy, z):
        cx = int((self.src_w / 2) + cx)
        cy = int((self.src_h / 2) + cy)

        ori_w, ori_h = self.calculate_resolution(z)

        startX = cx - ori_w // 2
        startY = cy - ori_h // 2
        endX = startX + ori_w
        endY = startY + ori_h

        cropped_and_padded_image = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)

        startX_clamped = max(startX, 0)
        startY_clamped = max(startY, 0)
        endX_clamped = min(endX, self.src_w)
        endY_clamped = min(endY, self.src_h)

        need_padding = startX < 0 or startY < 0 or endX > self.src_w or endY > self.src_h

        if need_padding:
            cropped_and_padded_image = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
            pad_top = max(0, -startY)
            pad_left = max(0, -startX)
            pad_bottom = max(0, endY - self.src_h)
            pad_right = max(0, endX - self.src_w)
            try:
                crop_area = frame[startY_clamped:endY_clamped, startX_clamped:endX_clamped]
                cropped_and_padded_image[pad_top:ori_h-pad_bottom, pad_left:ori_w-pad_right] = crop_area
            except:
                cropped_and_padded_image = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
        else:
            cropped_and_padded_image = frame[startY_clamped:endY_clamped, startX_clamped:endX_clamped].copy()

        frame_rgb = cv2.cvtColor(cropped_and_padded_image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        return frame_surface

    def crop_limited_area(self, frame, cx, cy, z):
        cx = int((self.src_w / 2) + cx)
        cy = int((self.src_h / 2) + cy)

        ori_w, ori_h = self.calculate_resolution(z)

        startX = max(cx - ori_w // 2, 0)
        startY = max(cy - ori_h // 2, 0)
        endX = min(startX + ori_w, self.src_w)
        endY = min(startY + ori_h, self.src_h)
        startX = min(startX, self.src_w - ori_w)
        startY = min(startY, self.src_h - ori_h)

        cropped_and_padded_image = frame[startY:endY, startX:endX]
        frame_rgb = cv2.cvtColor(cropped_and_padded_image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        return frame_surface
