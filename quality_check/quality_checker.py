import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from person_manager import PersonManager
from sklearn.metrics import auc


class QualityChecker:
    def __init__(self, detector, encoder, calculator, img_for_db_path, positive_img_path, negative_img_path, db_path):
        self.detector = detector
        self.encoder = encoder
        self.calculator = calculator
        self.img_for_db_path = img_for_db_path
        self.positive_img_path = positive_img_path
        self.negative_img_path = negative_img_path
        self.db_path = db_path
        self.person_manager = PersonManager(self.detector, self.encoder, self.db_path, self.img_for_db_path)
        self.person_manager.add_persons_to_db()
        pass

    def check(self):
        positive_results = self.find(self.positive_img_path)
        negative_results = self.find(self.negative_img_path)

        print("positive:")
        print(positive_results)
        print("negative:")
        print(negative_results)

        thresholds = np.linspace(0, 1, 100)
        tpirs = []
        fpirs = []
        fnirs = []

        not_found = 0
        for r in positive_results:
            if not r.is_found:
                not_found += 1
        for r in negative_results:
            if not r.is_found:
                not_found += 1

        for t in thresholds:
            n1 = len(negative_results)
            a2 = 0
            for r in negative_results:
                if r.is_found and r.distance < t:
                    a2 += 1
            fpir = a2 / n1

            n = len(positive_results)
            a1 = 0
            for r in positive_results:
                if r.is_found and r.distance > t:
                    a1 += 1
            fnir = a1 / n
            tpir = 1 - fnir

            tpirs.append(tpir)
            fpirs.append(fpir)
            fnirs.append(fnir)

        auc_value = auc(fpirs, tpirs)
        not_found_rate = not_found / (len(positive_results) + len(negative_results))

        print(f'not_found_rate: {not_found_rate}')
        print(f'tpirs: {tpirs}')
        print(f'fpirs: {fpirs}')
        print(f'fnirs: {fnirs}')
        print(f'thresholds: {thresholds}')
        print(f'AUC: {auc_value}')

        optimal_idx = np.argmin(np.absolute((np.array(fnirs) - np.array(fpirs))))
        optimal_threshold = thresholds[optimal_idx]
        print(f'optimal threshold: {optimal_threshold}')

        # Построение графика ROC
        plt.plot(fpirs, tpirs)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        plt.savefig('quality_check/results/roc.png')

        # Очистить график для построения следующего графика
        plt.clf()

        # Построение графика DET
        plt.plot(fpirs, fnirs)
        plt.ylabel('False Negative Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        # Построение прямой FNIR=FPIR
        plt.plot(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
        plt.show()
        plt.savefig('quality_check/results/det.png')

    def find(self, img_path):
        results = []
        for person_id in os.listdir(img_path):
            person_path = os.path.join(img_path, person_id)
            img = cv2.imread(os.path.join(person_path, os.listdir(person_path)[0]))
            detected_img = self.detector.detect(img)
            if detected_img is None:
                print(f"Не удалось обнаружить лицо {person_id}")
                results.append(Result(False, person_id, "", 1))
                continue
            encoding = self.encoder.encode(detected_img)

            if encoding is not None:
                first_person = self.person_manager.persons[0]
                min_res = self.calculator.calculate(encoding, np.array(first_person.encoding))
                min_person_fullname = first_person.fullname

                for person in self.person_manager.persons:
                    res = self.calculator.calculate(encoding, np.array(person.encoding))
                    if res < min_res:
                        min_res = res
                        min_person_fullname = person.fullname

                results.append(Result(True, person_id, min_person_fullname, min_res))
            else:
                print(f'Не удалось построить вектор биометрии для лица {person_id}')
                results.append(Result(False, person_id, "", 1))
        return results


class Result:
    def __init__(self, is_found, name, name_from_db, distance):
        self.is_found = is_found
        self.name = name
        self.name_from_db = name_from_db
        self.distance = distance

    def __repr__(self):
        return f'\n{self.name} => {self.name_from_db}: {self.distance}'
