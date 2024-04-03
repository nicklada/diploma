import cv2
from authenticator import Authenticator
from distance_calculaton.euclidian_dist_calculator import EuclidianCalculator
from distance_calculaton.l2_euclidian_dist_calculator import EuclidianL2Calculator
from face_detection.retina_face_detector import RetinaFaceDetector
from face_detection.retina_face_dlib_detector import RetinaFaceDlibDetector
from face_encoding.dlib_encoder import DlibEncoder
from face_encoding.ghost_facenet_encoder import GhostFaceNetEncoder
from person_manager import PersonManager

if __name__ == '__main__':
    # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
    # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
    test_img = cv2.imread('data_face_medium/20/20.5.jpg')
    detector = RetinaFaceDlibDetector()
    encoder = DlibEncoder()
    calculator = EuclidianCalculator()
    threshold = 0.6
    authenticator = Authenticator(detector, encoder, calculator, "db_dlib_medium_extracted.json", threshold)
    authenticator.authenticate(test_img)

# if __name__ == '__main__':
#     detector = RetinaFaceDetector()
#     test_img = cv2.imread('test_faces/m.0c63qw_0002.jpg')
#     encoder = GhostFaceNetEncoder()
#     calculator = EuclidianL2Calculator()
#     threshold = 1.2
#     authenticator = Authenticator(detector, encoder, calculator, "db_ghost_medium.json", threshold)
#     authenticator.authenticate(test_img)

# if __name__ == '__main__':
#     # создает объект класса PersonManager и вызывает метод add_persons_to_db,
#     # который делает ДБ в формате json из данных в директории data_face
#     detector = RetinaFaceDlibDetector()
#     encoder = DlibEncoder()
#     db_path = "db_dlib_medium_extracted.json"
#     person_manager = PersonManager(detector, encoder, db_path)
#     person_manager.add_persons_to_db()

# if __name__ == '__main__':
#     # создает объект класса PersonManager и вызывает метод add_persons_to_db,
#     # который делает ДБ в формате json из данных в директории data_face
#     detector = RetinaFaceDetector()
#     encoder = GhostFaceNetEncoder()
#     db_path = "db_ghost_medium.json"
#     person_manager = PersonManager(detector, encoder, db_path)
#     person_manager.add_persons_to_db()
