"""Face detection utilities."""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import torch


class FaceDetector:
    """Face detection using DNN-based methods."""

    def __init__(self, device: str = "cpu"):
        """Initialize face detector.

        Args:
            device: Device to use (cuda or cpu)
        """
        self.device = device
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image.

        Args:
            image: Input image (BGR format)
            min_confidence: Minimum confidence threshold

        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return list(faces)

    def get_face_bbox(
        self,
        image: np.ndarray,
        padding: float = 0.3
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get the primary face bounding box with padding.

        Args:
            image: Input image
            padding: Padding around face as fraction of face size

        Returns:
            Bounding box (x1, y1, x2, y2) or None if no face detected
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None

        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)

        return (x1, y1, x2, y2)

    def crop_face(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Optional[np.ndarray]:
        """Crop and resize face from image.

        Args:
            image: Input image
            target_size: Target size (width, height)

        Returns:
            Cropped and resized face image or None
        """
        bbox = self.get_face_bbox(image)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, target_size)

        return face
