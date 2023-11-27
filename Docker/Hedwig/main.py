import os
import json
from io import BytesIO
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import (
    TreePredictor, Tree
)
from nanoowl.tree_drawing import draw_tree_output


from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreeOutput

# Varaibles

temp_labels_folder = '/var/tmp'

if not os.path.exists(temp_labels_folder):
    os.makedirs(temp_labels_folder)


if 'model_prompt' not in st.session_state:
    st.session_state.model_prompt = ''

if 'model_theshold' not in st.session_state:
    st.session_state.model_theshold = 0.1

if 'iou_nms_theshold' not in st.session_state:
    st.session_state.iou_nms_theshold = 0.5

if 'upload_images' not in st.session_state:
    st.session_state.upload_images = []

if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

if 'predictor' not in st.session_state:
    st.session_state.predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            "google/owlvit-base-patch32",
            image_encoder_engine="./models/owl_image_encoder_patch32.engine"
    )
)


# Functions

def extract_detections(output_vit: TreeOutput, tree_vit: Tree):

    boxes_xyxy = []
    classes = []
    scores = []

    detections = output_vit.detections
    list_labels = tree_vit.to_dict()['labels'][1:]
    list_labels = [x.replace('a ','').replace('an ','') for x in list_labels]
    output_names = dict(zip(list(range(len(list_labels))), list_labels))

    for detection in detections:

        box = [int(x) for x in detection.box]
        label = detection.labels[0] - 1
        score = detection.scores[0]

        if label >= 0:
            
            boxes_xyxy.append(box)
            classes.append(label)
            scores.append(score)
    
    return output_names, boxes_xyxy, classes, scores

def apply_NMS(boxes, classes, scores, score_threshold=0.0, iou_threshold=0.5):
    unique_classes = set(classes)
    final_boxes, final_classes, final_scores = [], [], []

    for cls in unique_classes:
        # Filtrar cajas, puntuaciones y clases por clase actual
        idxs = [i for i, c in enumerate(classes) if c == cls]
        boxes_cls = [boxes[i] for i in idxs]
        scores_cls = [scores[i] for i in idxs]

        # Convertir a NumPy arrays
        boxes_np = np.array(boxes_cls)
        scores_np = np.array(scores_cls)

        # Aplicar NMS
        nms_indices = cv2.dnn.NMSBoxes(boxes_np, scores_np, score_threshold, iou_threshold)

        if len(nms_indices) == 0:
            continue

        if nms_indices.ndim > 1:
            nms_indices = nms_indices.squeeze()

        # Agregar resultados filtrados a las listas finales
        for i in nms_indices:
            final_boxes.append(boxes_cls[i])
            final_scores.append(scores_cls[i])
            final_classes.append(cls)

    return final_boxes, final_classes, final_scores

def plot_predictions(image_bgr, detection_boxes, detection_classes, detection_confidences, classes_dict):
    output_image = image_bgr.copy()

    # Define an extensive list of colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (0, 255, 255), (255, 0, 255), 
        (128, 128, 128), (128, 0, 0), (128, 128, 0), 
        (0, 128, 0), (128, 0, 128), (0, 128, 128), 
        (0, 0, 128), (255, 128, 128), (128, 255, 128), 
        (128, 128, 255), (255, 128, 0), (128, 255, 0), 
        (0, 255, 128), (0, 128, 255), (255, 0, 128), 
        (128, 0, 255), (255, 255, 128), (255, 128, 255), 
        (128, 255, 255), (192, 192, 192), (64, 64, 64), 
        (64, 0, 0), (64, 64, 0), (0, 64, 0)
    ]

    # Calculate the font scale factor based on the image size
    height, width = image_bgr.shape[:2]
    font_scale = max(width, height) / 1000  # Adjust this value as needed

    # Adjust text location offset based on the image size
    x_offset = width // 110  # Horizontal offset
    y_offset = height // 28  # Vertical offset

    for i in range(len(detection_boxes)):
        # Boxes
        xmin, ymin, xmax, ymax = detection_boxes[i]
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)

        # Assign a color based on the class using modulo to avoid errors
        class_id = detection_classes[i]
        color = colors[class_id % len(colors)]

        output_image = cv2.rectangle(output_image, start_point, end_point, color, 2)

        # Classes and scores
        text_location = (xmin + x_offset, ymin + y_offset)  # Adjusted text location
        text = f'{classes_dict[class_id]}-{str(round(detection_confidences[i], 2))}'

        output_image = cv2.putText(output_image, text, text_location, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, color, 1, cv2.LINE_AA)

    return output_image


def inference_owlvit_bgr_image(input_image_bgr, input_prompt: str, model_predictor: OwlPredictor, model_threshold: float, iou_treshold: float):

    input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    input_image_pil = Image.fromarray(input_image_rgb)

    tree = Tree.from_prompt(input_prompt)
    clip_text_encodings = model_predictor.encode_clip_text(tree)
    owl_text_encodings = model_predictor.encode_owl_text(tree)
    
    output_predictor = model_predictor.predict(
            image=input_image_pil, 
            tree=tree,
            clip_text_encodings=clip_text_encodings,
            owl_text_encodings=owl_text_encodings,
            threshold=model_threshold
        )

    label_dict, boxes, classes, scores = extract_detections(output_predictor, tree)
    
    boxes, classes, scores = apply_NMS(boxes, classes, scores, model_threshold, iou_treshold)

    return boxes, classes, scores, label_dict

def create_json_labelme_from_image_preds(image_bgr, image_name, output_json_route, boxes, classes, confidence, classes_name):
    
    image_width, image_height = image_bgr.shape[1], image_bgr.shape[0]
    
    labelme_json_data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,  # Leave it as None for simplicity
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
    
    num_detections = len(boxes)
    
    for i in range(num_detections):
        
        # Boxes
        xmin, ymin, xmax, ymax = boxes[i]

        start_point = [float(xmin), float(ymin)]
        end_point = [float(xmax), float(ymax)]
        
        class_name = classes_name[classes[i]]

        shape_data = {
            "label": class_name,
            "points": [start_point, end_point],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }

        labelme_json_data["shapes"].append(shape_data)
            
    json_file_route = output_json_route + '/' + image_name[:-4] + '.json'
    
    with open(json_file_route, "w") as outfile:
        json_object = json.dumps(labelme_json_data, indent=2)
        outfile.write(json_object)

    return json_file_route


def main():

    #  Title

    st.set_page_config(layout="wide", page_title="Hedwig AI")

    # Main page

    st.write("# [Hedwig AI :owl:](https://github.com/CristianCristanchoT/Hedwig-IA)")
    st.write(
        "Try uploading an image and write what you want to detect in the image. :grin:"
    )

    col1, col2, col3 = st.columns(3)

    previous_button = col1.button(":arrow_backward: Previous Image", key="previous")
    next_button = col3.button("Next Image :arrow_forward:", key="next")


    # Sidebar

    st.sidebar.write("## 1. Upload the images :camera:")

    st.session_state.upload_images = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    st.sidebar.write("## 2. Prompt the model with your desired labels :gear:")

    st.session_state.model_prompt = st.sidebar.text_input(
            "Model Prompt",
            label_visibility="collapsed",
            placeholder="[a person [a face, a hand], a car]",
        )

    st.sidebar.write("## 3. Adjust the model's inference parameters. :wrench:")

    st.session_state.model_theshold = st.sidebar.slider('Model threshold', 0.01, 0.7, 0.1, 0.01, format="%f")

    st.session_state.iou_nms_theshold = st.sidebar.slider('NMS IOU threshold', 0.1, 1.0, 0.5, 0.1, format="%f")
    
    # Execution

    if st.session_state.upload_images:


        if previous_button:

            st.session_state.image_index -= 1

            if st.session_state.image_index < 0:
                st.session_state.image_index = 0

            print('previous_button')
            print(st.session_state.image_index)


        if next_button:

            st.session_state.image_index += 1

            if st.session_state.image_index > len(st.session_state.upload_images) - 1:
                st.session_state.image_index = len(st.session_state.upload_images) - 1 

            print('next_button')
            print(st.session_state.image_index)


        image_file = st.session_state.upload_images[st.session_state.image_index]
        image_name = image_file.name
        image_pil = Image.open(image_file)
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if st.session_state.model_prompt == '':

            output_image_pil = image_pil

        else:

            output_boxes, output_classes, output_scores, output_label_dict = inference_owlvit_bgr_image(
                image_bgr, 
                st.session_state.model_prompt, 
                st.session_state.predictor, 
                st.session_state.model_theshold, 
                st.session_state.iou_nms_theshold
                )
            
            output_image_bgr = plot_predictions(image_bgr, output_boxes, output_classes, output_scores, output_label_dict)
            output_image_rgb = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB)
            output_image_pil = Image.fromarray(output_image_rgb)

            json_path = create_json_labelme_from_image_preds(image_bgr, image_name, temp_labels_folder, output_boxes, output_classes, output_scores, output_label_dict)

            # Save json

            with open(json_path, "r") as file:
                json_content = file.read()

            col2.download_button(
                label="Save :floppy_disk:",
                data=BytesIO(json_content.encode()),
                file_name=image_name[:-4] + ".json",
                key="json_button"
            )

        # Image render
        st.image(output_image_pil, use_column_width="auto")


if __name__ == "__main__":
    main()