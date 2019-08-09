from mtcnn.mtcnn import MTCNN
import cv2
import numpy
import imutils

# Создание сети нахождения лиц
detector = MTCNN()

# Загрузка изображения с лицами
image = cv2.imread('demo/detection_image/input/people.jpg')

# Увеличение/уменьшение наименьшей стороны изображения до 1000 пикселей
if image.shape[0] < image.shape[1]:
    image = imutils.resize(image, height=1000)
else:
    image = imutils.resize(image, width=1000)

# Получить размеры изображения
image_size = numpy.asarray(image.shape)[0:2]

# Получение списка лиц с координатами и значением уверенности
faces_boxes = detector.detect_faces(image)

# Копия изображения для рисования рамок на нём
image_detected = image.copy()

# Копия изображения для рисования меток на нём
image_marked = image.copy()

# Замена BGR на RGB (так находит в два раза больше лиц)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Цвет меток BGR
marked_color = (0, 255, 0, 1)

# Работа с лицами
if faces_boxes:

    face_n = 0  # Инициализация счётчика лиц
    for face_box in faces_boxes:

        # Увеличение счётчика файлов
        face_n += 1

        # Координаты лица
        x, y, w, h = face_box['box']

        # Отступы для увеличения рамки
        d = h - w  # Разница между высотой и шириной
        w = w + d  # Делаем изображение квадратным
        x = numpy.maximum(x - round(d / 2), 0)
        x1 = numpy.maximum(x - round(w / 4), 0)
        y1 = numpy.maximum(y - round(h / 4), 0)
        x2 = numpy.minimum(x + w + round(w / 4), image_size[1])
        y2 = numpy.minimum(y + h + round(h / 4), image_size[0])

        print(x1, y1, x2, y2, image_size[0], image_size[1])

        # Получение картинки с лицом
        cropped = image_detected[y1:y2, x1:x2, :]
        face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

        # Имя файла (уверенность + номер)
        face_file_name = str(face_box['confidence']) + '.' + str(face_n) + '.jpg'

        # Отборка лиц {selected|rejected}
        if face_box['confidence'] > 0.99:  # 0.99 - уверенность сети в процентах что это лицо

            # Рисует белый квадрат на картинке по координатам
            cv2.rectangle(
                image_detected,
                (x1, y1),
                (x2, y2),
                (255, 255, 255, 1),
                1
            )

            # Путь к директории с качественными изображениями
            face_path = 'demo/detection_image/output/faces/selected/' + face_file_name

            # Информируем консоль
            print('\033[92m' + face_file_name + '\033[0m')

        else:

            # Рисует красный квадрат на картинке по координатам
            cv2.rectangle(
                image_detected,
                (x1, y1),
                (x2, y2),
                (0, 0, 255, 1),
                1
            )

            # Путь к директории с отбракованными изображениями
            face_path = 'demo/detection_image/output/faces/rejected/' + face_file_name

            # Информируем консоль
            print('\033[91m' + face_file_name + '\033[0m')

        # Рисует левый глаз
        cv2.rectangle(
            image_marked,
            (face_box['keypoints']['left_eye'][0], face_box['keypoints']['left_eye'][1]),
            (face_box['keypoints']['left_eye'][0] + 1, face_box['keypoints']['left_eye'][1] + 1),
            marked_color,
            1
        )

        # Рисует правый глаз
        cv2.rectangle(
            image_marked,
            (face_box['keypoints']['right_eye'][0], face_box['keypoints']['right_eye'][1]),
            (face_box['keypoints']['right_eye'][0] + 1, face_box['keypoints']['right_eye'][1] + 1),
            marked_color,
            1
        )

        # Рисует нос
        cv2.rectangle(
            image_marked,
            (face_box['keypoints']['nose'][0], face_box['keypoints']['nose'][1]),
            (face_box['keypoints']['nose'][0] + 1, face_box['keypoints']['nose'][1] + 1),
            marked_color,
            1
        )

        # Рисует левую часть рта
        cv2.rectangle(
            image_marked,
            (face_box['keypoints']['mouth_left'][0], face_box['keypoints']['mouth_left'][1]),
            (face_box['keypoints']['mouth_left'][0] + 1, face_box['keypoints']['mouth_left'][1] + 1),
            marked_color,
            1
        )

        # Рисует правую часть рта
        cv2.rectangle(
            image_marked,
            (face_box['keypoints']['mouth_right'][0], face_box['keypoints']['mouth_right'][1]),
            (face_box['keypoints']['mouth_right'][0] + 1, face_box['keypoints']['mouth_right'][1] + 1),
            marked_color,
            1
        )

        # Сохранение изображения лица на диск в директории {selected|rejected}
        cv2.imwrite(face_path, face_image)

    # Сохранение исходного изображения с выделенными лицами
    cv2.imwrite('demo/detection_image/output/people-detected.jpg', image_detected)

    # Сохранение исходного изображения с точками
    cv2.imwrite('demo/detection_image/output/people-marked.jpg', image_marked)
