import cv2
from matplotlib import pyplot as plt

from authenticator import Authenticator
from distance_calculaton.euclidian_dist_calculator import EuclidianCalculator
from distance_calculaton.l2_euclidian_dist_calculator import EuclidianL2Calculator
from face_detection.dlib_detector import DlibDetector
from face_detection.dummy_face_detector import DummyFaceDetector
from face_detection.mtcnn_detector import MtcnnDetector
from face_detection.retina_face_detector import RetinaFaceDetector
from face_detection.retina_face_dlib_detector import RetinaFaceDlibDetector
from face_encoding.dlib_encoder import DlibEncoder
from face_encoding.facenet_encoder import FaceNetEncoder
from face_encoding.ghost_facenet_encoder import GhostFaceNetEncoder
from face_encoding.insight_face import InsightFaceEncoder
from person_manager import PersonManager

from quality_check.quality_checker import QualityChecker


def dlib_detector_dlib_encoder_create_database():
    # создает объект класса PersonManager и вызывает метод add_persons_to_db,
    # который делает ДБ в формате json из данных в директории data_face
    detector = DlibDetector()
    encoder = DlibEncoder()
    db_path = "databases/db_dlib_dlib_20.json"
    person_manager = PersonManager(detector, encoder, db_path)
    person_manager.add_persons_to_db()


def retina_detector_dlib_encoder_create_database():
    detector = RetinaFaceDlibDetector()
    encoder = DlibEncoder()
    db_path = "databases/db_retina_dlib_20.json"
    person_manager = PersonManager(detector, encoder, db_path)
    person_manager.add_persons_to_db()


def retina_detector_ghost_encoder_create_database():
    detector = RetinaFaceDetector()
    encoder = GhostFaceNetEncoder()
    db_path = "databases/db_retina_ghost_20.json"
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
    authenticator = Authenticator(detector, encoder, calculator, "databases/db_dlib_dlib_20.json", threshold)
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
    authenticator = Authenticator(detector, encoder, calculator, "databases/db_retina_dlib_20.json", threshold)
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
    authenticator = Authenticator(detector, encoder, calculator, "databases/db_retina_ghost_20.json", threshold)
    authenticator.authenticate(test_img)


# if __name__ == '__main__':
#     dlib_detector_dlib_encoder_create_database()
#     print("БД db_dlib_dlib_20.json сформирована")
#     retina_detector_dlib_encoder_create_database()
#     print("БД db_retina_dlib_20.json сформирована")
#     retina_detector_ghost_encoder_create_database()
#     print("БД db_retina_ghost_20.json сформирована")
#
# if __name__ == '__main__':
#     img_path = "datasets/test_faces/test_40/test_34/blur_34.jpeg"
#     print("DLIB+DLIB")
#     dlib_detector_dlib_encoder_authenticate(img_path)
#     print("RETINA+DLIB")
#     retina_detector_dlib_encoder_authenticate(img_path)
#     print("RETINA+GHOST")
#     retina_detector_ghost_encoder_authenticate(img_path)

# if __name__ == '__main__':
#     detector = DlibDetector()
#     encoder = DlibEncoder()
#     calculator = EuclidianCalculator()
#     quality_checker = QualityChecker(
#         detector,
#         encoder,
#         calculator,
#         "datasets/dataset/db",
#         "datasets/dataset/positive",
#         "datasets/dataset/negative",
#         "databases/quality_check.json"
#     )
#     quality_checker.check()

# if __name__ == '__main__':
#     detector = RetinaFaceDlibDetector()
#     encoder = DlibEncoder()
#     calculator = EuclidianCalculator()
#     quality_checker = QualityChecker(
#         detector,
#         encoder,
#         calculator,
#         "datasets/dataset/db",
#         "datasets/dataset/positive",
#         "datasets/dataset/negative",
#         "databases/quality_check.json"
#     )
#     quality_checker.check()

# if __name__ == '__main__':
#     detector = MtcnnDetector()
#     encoder = FaceNetEncoder()
#     calculator = EuclidianCalculator(2)
#     quality_checker = QualityChecker(
#         detector,
#         encoder,
#         calculator,
#         "datasets/dataset/db",
#         "datasets/dataset/positive",
#         "datasets/dataset/negative",
#         "databases/quality_check.json"
#     )
#     quality_checker.check()

# if __name__ == '__main__':
#     detector = DummyFaceDetector()
#     encoder = InsightFaceEncoder()
#     calculator = EuclidianCalculator(100)
#     quality_checker = QualityChecker(
#         detector,
#         encoder,
#         calculator,
#         "datasets/dataset/db",
#         "datasets/dataset/positive",
#         "datasets/dataset/negative",
#         "databases/quality_check.json"
#     )
#     quality_checker.check()

if __name__ == '__main__':
    detectors = [DlibDetector(), RetinaFaceDlibDetector(), MtcnnDetector(), DummyFaceDetector()]
    encoders = [DlibEncoder(), DlibEncoder(), FaceNetEncoder(), InsightFaceEncoder()]
    calculators = [EuclidianCalculator(), EuclidianCalculator(), EuclidianCalculator(2), EuclidianCalculator(100)]
    fpirs = []
    tpirs = []
    for i in range(4):
        quality_checker = QualityChecker(
            detectors[i],
            encoders[i],
            calculators[i],
            "datasets/dataset/db",
            "datasets/dataset/positive",
            "datasets/dataset/negative",
            "databases/quality_check.json"
        )
        result = quality_checker.check()
        fpirs.append(result[0])
        tpirs.append(result[1])
    plt.plot(fpirs[0], tpirs[0], label='Dlib+Dlib')
    plt.plot(fpirs[1], tpirs[1], label='RetinaFace+Dlib')
    plt.plot(fpirs[2], tpirs[2], label='MTCNN+FaceNet')
    plt.plot(fpirs[3], tpirs[3], label='RetinaFace+InsightFace')
    plt.legend()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Сравнение ROC-curve всех вариаций алгоритма')
    plt.show()
    plt.grid(True)
    plt.savefig('quality_check/results/roc_all.png')
