import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch

from face_encoding.encoder import Encoder


class FaceNetEncoder(Encoder):

    def encode(self, img: np.ndarray, is_test=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        return resnet(img).detach().cpu().numpy()
