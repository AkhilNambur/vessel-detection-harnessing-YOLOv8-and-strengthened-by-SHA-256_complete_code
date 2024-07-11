from ultralytics import YOLO
import cv2
import math
import hashlib

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model = YOLO("../YOLO-Weights/yolov8_web_wts.pt")
    classNames = ["Boat", "Speedboat", "Yacht", "Ship"]

    class_counts = {class_name: 0 for class_name in classNames}

    while True:
        success, img = cap.read()
        results = model(img)

        frame_hash = hashlib.SHA5(img.tobytes()).hexdigest()

        for class_name in classNames:
            class_counts[class_name] = 0

        any_class_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Draw bounding box
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # Draw label background
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                class_counts[class_name] += 1
                any_class_detected = True


        strip_height = int(frame_height * 0.06)
        strip_y = frame_height - strip_height


        cv2.rectangle(img, (0, strip_y), (frame_width, frame_height), (255, 255, 255), -1)


        hash_x = int(frame_width * 0.01)
        hash_y = strip_y + int(strip_height * 0.7)

        cv2.putText(img, f"Hash: {frame_hash}", (hash_x, hash_y), 0, 0.6, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)


        if any_class_detected:
            count_label = "Counts (Current Frame): "
            line_number = 0
            for class_name in classNames:
                if class_counts[class_name] > 0:
                    count_label += f"{class_name}:{class_counts[class_name]}  "
                    line_number += 1
                    cv2.putText(img, f"{class_name}:{class_counts[class_name]}", (10, 30 + 25 * line_number), 0, 0.8, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        else:
            count_label = "No vessel detected"
            cv2.putText(img, count_label, (10, 30), 0, 0.8, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

        yield img
        out.write(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()