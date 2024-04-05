import cv2
from authenticator import Authenticator
from distance_calculaton.euclidian_dist_calculator import EuclidianCalculator
from distance_calculaton.l2_euclidian_dist_calculator import EuclidianL2Calculator
from face_detection.dlib_detector import DlibDetector
from face_detection.retina_face_detector import RetinaFaceDetector
from face_detection.retina_face_dlib_detector import RetinaFaceDlibDetector
from face_encoding.dlib_encoder import DlibEncoder
from face_encoding.ghost_facenet_encoder import GhostFaceNetEncoder
from person_manager import PersonManager


def dlib_detector_dlib_encoder_create_database():
    # создает объект класса PersonManager и вызывает метод add_persons_to_db,
    # который делает ДБ в формате json из данных в директории data_face
    detector = DlibDetector()
    encoder = DlibEncoder()
    db_path = "db_dlib_dlib_medium.json"
    person_manager = PersonManager(detector, encoder, db_path)
    person_manager.add_persons_to_db()


def retina_detector_dlib_encoder_create_database():
    detector = RetinaFaceDlibDetector()
    encoder = DlibEncoder()
    db_path = "db_retina_dlib_medium.json"
    person_manager = PersonManager(detector, encoder, db_path)
    person_manager.add_persons_to_db()


def retina_detector_ghost_encoder_create_database():
    detector = RetinaFaceDetector()
    encoder = GhostFaceNetEncoder()
    db_path = "db_retina_ghost_medium.json"
    person_manager = PersonManager(detector, encoder, db_path)
    person_manager.add_persons_to_db()


def dlib_detector_dlib_encoder_authenticate(img_path):
    # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
    # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
    test_img = cv2.imread(img_path)
    if test_img is None:
        return print("Изображение по указанному пути не найдено")
    detector = DlibDetector()
    encoder = DlibEncoder()
    calculator = EuclidianCalculator()
    threshold = 0.6
    authenticator = Authenticator(detector, encoder, calculator, "db_dlib_dlib_medium.json", threshold)
    authenticator.authenticate(test_img)


def retina_detector_dlib_encoder_authenticate(img_path):
    # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
    # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
    test_img = cv2.imread(img_path)
    if test_img is None:
        return print("Изображение по указанному пути не найдено")
    detector = RetinaFaceDlibDetector()
    encoder = DlibEncoder()
    calculator = EuclidianCalculator()
    threshold = 0.6
    authenticator = Authenticator(detector, encoder, calculator, "db_retina_dlib_medium.json", threshold)
    authenticator.authenticate(test_img)


def retina_detector_ghost_encoder_authenticate(img_path):
    # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
    # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
    test_img = cv2.imread(img_path)
    if test_img is None:
        return print("Изображение по указанному пути не найдено")
    detector = RetinaFaceDetector()
    encoder = GhostFaceNetEncoder()
    calculator = EuclidianCalculator()
    threshold = 1.3
    authenticator = Authenticator(detector, encoder, calculator, "db_retina_ghost_medium.json", threshold)
    authenticator.authenticate(test_img)


# if __name__ == '__main__':
#     dlib_detector_dlib_encoder_create_database()
#     print("БД db_dlib_dlib_medium.json сформирована")
#     retina_detector_dlib_encoder_create_database()
#     print("БД db_retina_dlib_medium.json сформирована")
#     retina_detector_ghost_encoder_create_database()
#     print("БД db_retina_ghost_medium.json сформирована")

if __name__ == '__main__':
    img_path = "test_faces/test person 1/turned.png"
    print("DLIB+DLIB")
    dlib_detector_dlib_encoder_authenticate(img_path)
    print("RETINA+DLIB")
    retina_detector_dlib_encoder_authenticate(img_path)
    print("RETINA+GHOST")
    retina_detector_ghost_encoder_authenticate(img_path)


# if __name__ == '__main__':
#     # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
#     # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
#     test_img = cv2.imread('data_face_medium/20/20.5.jpg')
#     detector = RetinaFaceDlibDetector()
#     encoder = DlibEncoder()
#     calculator = EuclidianCalculator()
#     threshold = 0.6
#     authenticator = Authenticator(detector, encoder, calculator, "db_dlib_medium_extracted.json", threshold)
#     authenticator.authenticate(test_img)

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

# if __name__ == '__main__':
#     # создает объект класса PersonManager и вызывает метод add_persons_to_db,
#     # который делает ДБ в формате json из данных в директории data_face
#     detector = DlibDetector()
#     encoder = DlibEncoder()
#     db_path = "db_dlib_medium_extracted.json"
#     person_manager = PersonManager(detector, encoder, db_path)
#     person_manager.add_persons_to_db()
