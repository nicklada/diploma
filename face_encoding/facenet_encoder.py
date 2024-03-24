import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch

from face_detection.mtcnn_detector import MTCNNDetector
from face_encoding.encoder import Encoder


class FaceNetEncoder(Encoder):
    def __init__(self):
        self.face_detector = MTCNNDetector()

    def encode(self, img: np.ndarray, is_test=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        face = self.face_detector.detect(img)
        if face is None:
            return None
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        aligned = [face]
        aligned = torch.stack(aligned).to(device)
        return resnet(aligned).detach().cpu().numpy()
