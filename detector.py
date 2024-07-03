import face_recognition
import cv2

def load_known_faces(known_faces_dir):
    # Load known faces from the provided directory
    image_files = face_recognition.api.image_files_in_folder(known_faces_dir)
    known_face_encodings = []
    known_face_names = []

    for file in image_files:
        image = face_recognition.load_image_file(f"{known_faces_dir}/{file}")
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(file.split('.')[0])

    return known_face_encodings, known_face_names

def recognize_faces(known_faces_dir, test_image_path):
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    test_image = face_recognition.load_image_file(test_image_path)
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        if True in matches:
            index = matches.index(True)
            name = known_face_names[index]

        top, right, bottom, left = face_location
        cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(test_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Facial Recognition", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_faces_directory = "known_faces"
    test_image_path = "test_image.jpg"
    recognize_faces(known_faces_directory, test_image_path)
