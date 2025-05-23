import os
from datetime import datetime
from cmath import sqrt
from collections import defaultdict
import pickle
import cv2
from ultralytics import YOLO

# Liste contenant les centres des personnes détectées et leurs nombres de frames où elles sont considérées immobiles
lst_personnes = []

# Seuil d'admissibilité de déplacement du centre d'une personne où elle est considérée comme immobile d'une frame à l'autre
seuil_distance = 0.005
# Format [width, height]
taille_image = []

# Nombre de frames successives pour qu'une personne immobile soit considérée en attente
seuil_attente = 5

# Nombre de personnes en attente à chaque indice === frame
nb_attentes_frames = []

now = datetime.now()
# Create the folder name
my_folder_name = now.strftime("%Y-%m-%d_%H" + "h" + "%M" + "min" + "%S" + "sec")

results_filename = "results.pkl"

def get_frame_size(video_path):
    # Ouverture de la vidéo
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Vidéo introuvable")
        return None

    # On charge la première frame
    ret, frame = capture.read()
    if not ret:
        print("Pas de première frame dans la vidéo")
        return None

    height, width = frame.shape[:2]
    capture.release()
    return width, height

def detect_persons(video_path):
    """ Détecte les personnes immobiles pour extraire une liste de personnes en attente du feu par frame """
    width, height = get_frame_size(video_path)
    taille_image.append(width)
    taille_image.append(height)

    # Load the model
    model = YOLO("yolov8n.pt")

    # Filter results to keep only 'person' class
    person_class_id = 0  # Assuming 'person' is class 0 in COCO dataset
    detected_persons = []
    frame_idx = 0  # Manual frame counter
    original_results = []

    # Utiliser stream=True pour traiter les frames une par une
    for result in model(video_path, stream=True):
        original_results.append(result)
        for box in result.boxes:
            if box.cls[0] == person_class_id:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
                #confidence = box.conf[0]  # Confidence score of the detection
                detected_persons.append({
                    "frame_idx": frame_idx,
                    "bbox": [x1, y1, x2, y2]
                    # "confidence": confidence
                })
        frame_idx += 1  # Increment frame counter

    #return detected_persons, frame_idx
    return detected_persons, frame_idx, original_results

def save_results(results, output_dir, nb_attentes_frames):
    """
    Save the filtered results as images with the number of waiting persons in the filename.

    Args:
    - results (list): List of detection results.
    - output_dir (str): Directory to save the output images.
    - nb_attentes_frames (list): List indicating the number of waiting persons per frame.
    """
    global my_folder_name
    for idx, result in enumerate(results):
        # Plot the results on the image
        result_image = result.plot()

        # Include the number of waiting persons in the filename
        filename = f"{output_dir}/{my_folder_name}/filtered_result_{idx}_attente_{nb_attentes_frames[idx]}.jpg"

        # Save the image using OpenCV
        cv2.imwrite(filename, result_image)

def save_results_to_file(results, filename):
    """ Sauvegarde les résultats dans un fichier. """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results_from_file(filename):
    """ Charge les résultats à partir d'un fichier. """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main(video_path, output_dir):
    global lst_personnes, seuil_distance, taille_image, seuil_attente, nb_attentes_frames, my_folder_name

    # On récupère la liste totale des personnes dans la vidéo (frame par frame)
    detected_persons, max_frame, original_results = detect_persons(video_path)
    #detected_persons, max_frame = detect_persons(video_path)

    # Initialisation de la liste avec le nombre de personnes en attente par frame
    nb_attentes_frames = [0] * max_frame
    os.makedirs(os.path.join(output_dir, my_folder_name))

    # Création d'une liste avec la liste des personnes pour chaque frame
    lst_personnes_par_frame = defaultdict(list)
    for person in detected_persons:
        lst_personnes_par_frame[person['frame_idx']].append(person)
    lst_personnes_par_frame = list(lst_personnes_par_frame.values())

    for personnes_sur_cette_frame in lst_personnes_par_frame:
        frame_idx = personnes_sur_cette_frame[0]['frame_idx']
        lst_temp = []

        for person in personnes_sur_cette_frame:
            bbox = person['bbox']  # Coordonnées de la boîte englobante
            x = (bbox[0] + bbox[2]) / 2
            y = (bbox[1] + bbox[3]) / 2
            centre_personne = [x, y]

            min_distance = abs(sqrt(taille_image[0]**2 + taille_image[1]**2))
            coord_min = [[0, 0], 0]
            indice_min = 0

            for indice_person_prec in range(len(lst_personnes)):
                person_prec = lst_personnes[indice_person_prec]
                x_prec = person_prec[0][0]
                y_prec = person_prec[0][1]
                distance = abs(sqrt((x_prec - x)**2 + (y_prec - y)**2))
                if (distance < min_distance) & (not person_prec[2]):
                    min_distance = distance
                    coord_min[0] = [x_prec, y_prec]
                    coord_min[1] = person_prec[1]
                    indice_min = indice_person_prec

            if min_distance < seuil_distance * taille_image[1]:
                # Si on est dans le seuil de tolérance pour considérer que le box est sur la même personne,
                # on ajoute alors le centre et le compteur d'immobilité précédente incrémentée.
                lst_temp.append([centre_personne, coord_min[1] + 1, False])
                lst_personnes[indice_min][2] = True  # On active alors ce paramètre pour éviter de prendre plusieurs fois la même correspondance de personne.

                if coord_min[1] + 1 >= seuil_attente:
                    nb_attentes_frames[frame_idx] += 1
            else:
                # On considère que c'est une nouvelle personne
                lst_temp.append([centre_personne, 0, False])

        lst_personnes = lst_temp

    print(nb_attentes_frames)

    # Save the results
    save_results(original_results, output_dir, nb_attentes_frames)

# Example usage
if __name__ == "__main__":
    video_path = "data/videos/video_0163.mp4"
    output_dir = "./Resultats"
    main(video_path, output_dir)
