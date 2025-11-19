"""Lip sync engine for avatar animation."""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class LipSyncConfig:
    """Configuration for lip sync."""
    img_size: int = 96
    fps: int = 25
    mel_step_size: int = 16
    device: str = "cpu"


class Conv2d(nn.Module):
    """Custom Conv2d with optional batch norm."""

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Wav2LipModel(nn.Module):
    """Simplified Wav2Lip-style model for lip sync."""

    def __init__(self):
        super(Wav2LipModel, self).__init__()

        # Face encoder
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                         Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                         Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                         Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
        ])

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        # Face decoder
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(768, 384, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(512, 256, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(320, 128, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(160, 64, kernel_size=1, stride=1, padding=0)),
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        """Forward pass.

        Args:
            audio_sequences: Audio mel spectrogram (B, 1, 80, 16)
            face_sequences: Face images (B, 6, H, W)

        Returns:
            Generated face with synced lips
        """
        B = audio_sequences.size(0)

        # Encode face
        face_embedding = face_sequences
        feats = []
        for f in self.face_encoder_blocks:
            face_embedding = f(face_embedding)
            feats.append(face_embedding)

        # Encode audio
        audio_embedding = self.audio_encoder(audio_sequences)

        # Decode with skip connections
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                raise e
            feats.pop()
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, feats[-1]), dim=1)
        x = self.output_block(x)

        return x


class LipSyncEngine:
    """Lip sync engine using Wav2Lip-style approach."""

    def __init__(self, config: Optional[LipSyncConfig] = None):
        """Initialize lip sync engine.

        Args:
            config: Lip sync configuration
        """
        self.config = config or LipSyncConfig()
        self.device = torch.device(self.config.device)
        self.model: Optional[Wav2LipModel] = None

    def load_model(self, checkpoint_path: Optional[Path] = None):
        """Load the lip sync model.

        Args:
            checkpoint_path: Path to model checkpoint (optional)
        """
        self.model = Wav2LipModel().to(self.device)

        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        img = cv2.resize(image, (self.config.img_size, self.config.img_size))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img

    def get_smoothed_boxes(
        self,
        boxes: List,
        window_size: int = 5
    ) -> List:
        """Smooth face bounding boxes over time.

        Args:
            boxes: List of bounding boxes
            window_size: Smoothing window size

        Returns:
            Smoothed boxes
        """
        if len(boxes) < window_size:
            return boxes

        smoothed = []
        for i in range(len(boxes)):
            start = max(0, i - window_size // 2)
            end = min(len(boxes), i + window_size // 2 + 1)
            window = boxes[start:end]

            if window[0] is None:
                smoothed.append(None)
                continue

            avg_box = np.mean([b for b in window if b is not None], axis=0)
            smoothed.append(avg_box.astype(int))

        return smoothed

    @torch.no_grad()
    def generate_talking_face(
        self,
        face_image: np.ndarray,
        audio_features: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """Generate talking face frames.

        Args:
            face_image: Source face image
            audio_features: Audio mel spectrogram
            num_frames: Number of frames to generate

        Returns:
            List of generated frames
        """
        if self.model is None:
            # If no model loaded, return simple animation
            return self._generate_simple_animation(face_image, num_frames)

        frames = []
        face_tensor = self.preprocess_image(face_image).to(self.device)

        for i in range(num_frames):
            # Get audio segment for this frame
            audio_idx = min(i, audio_features.shape[1] - 1)
            audio_segment = audio_features[:, audio_idx:audio_idx+1]

            # Prepare inputs
            face_input = face_tensor.unsqueeze(0)
            face_input = torch.cat([face_input] * 2, dim=1)  # Duplicate for 6 channels
            audio_input = torch.from_numpy(audio_segment).float().unsqueeze(0).to(self.device)

            # Generate frame
            output = self.model(audio_input, face_input)
            frame = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)

            frames.append(frame)

        return frames

    def _generate_simple_animation(
        self,
        face_image: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """Generate simple mouth animation when no model is available.

        Args:
            face_image: Source face image
            num_frames: Number of frames to generate

        Returns:
            List of frames with simple mouth animation
        """
        frames = []
        h, w = face_image.shape[:2]

        for i in range(num_frames):
            frame = face_image.copy()

            # Simple oscillating mouth animation
            mouth_open = int(5 + 5 * np.sin(2 * np.pi * i / 10))

            # Draw animated mouth region (simplified)
            mouth_y = int(h * 0.7)
            mouth_x = int(w * 0.5)

            cv2.ellipse(
                frame,
                (mouth_x, mouth_y),
                (int(w * 0.15), mouth_open),
                0, 0, 180,
                (80, 50, 50),
                -1
            )

            frames.append(frame)

        return frames
