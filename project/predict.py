import numpy as np
import json
import cv2
from PIL import Image
from project.modules import model
import re
import unicodedata

model_manager = model.ModelManager()

def horizontal_histogram(binarizedImage):
    horizontal_projection = np.sum(binarizedImage, axis=1)/255
    return horizontal_projection

def find_local_minima(input_array):
    minima_indices = []
    for i in range(1, len(input_array) - 1):
        if input_array[i - 1] > input_array[i] < input_array[i + 1]:
            minima_indices.append(i)
    return minima_indices

def remove_elements(arr1, arr2):
    set_arr1 = set(arr1)
    set_arr2 = set(arr2)
    result_set = set_arr1 - set_arr2
    return np.array(list(result_set))

def process_consecutive_numbers(zeros_indices):
    processed_array = []
    current_consecutive = [zeros_indices[0]]

    for i in range(1, len(zeros_indices)):
        if zeros_indices[i] == zeros_indices[i - 1] + 1:
            current_consecutive.append(zeros_indices[i])
        else:
            if len(current_consecutive) > 1:
                mean_value = int(np.mean(current_consecutive))
                processed_array.append(mean_value)
            else:
                processed_array.extend(current_consecutive)

            current_consecutive = [zeros_indices[i]]

    if len(current_consecutive) > 1:
        mean_value = int(np.mean(current_consecutive))
        processed_array.append(mean_value)
    else:
        processed_array.extend(current_consecutive)

    return processed_array

def threshold(input_array,th=0.05):
    mean_value = np.mean(input_array)
    threshold_of_mean = th * mean_value
    return threshold_of_mean

def split_image_horizontal(image, segment_points):
    segments = []
    for i, segment_point in enumerate(segment_points):
        if i == 0:
            start_row = 0
        else:
            start_row = segment_points[i - 1]
        end_row = segment_point
        segment = image[start_row:end_row, :]
        segments.append(np.array(segment))

    return segments

def vertical_histogram(binarizedImage):
    vertical_projection = np.sum(binarizedImage, axis=0)/255
    return vertical_projection

def split_image_vertical(image, segment_points):
    segments = []
    for i, segment_point in enumerate(segment_points):
        if i == 0:
            start_row = 0
        else:
            start_row = segment_points[i - 1]

        end_row = segment_point
        segment = image[:, start_row:end_row]
        segments.append(np.array(segment))

    return segments

def search_key(key, json_file='project/model/mapping.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if key in data:
            return data[key]
        else:
            return None

def clean_string(input_string):

    cleaned_string = re.sub(r'\n+', '\n', input_string)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    cleaned_string = cleaned_string.strip()
    
    return cleaned_string

# Function to compute skew angle
def compute_skew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

# Correcting skew
def deskew(image):
    angle = compute_skew(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Function to detect and remove large lines
def remove_large_lines(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape
    
    line_threshold = int(0.2 * max(h, w))  # 20% of max dimension
    
    mask = np.ones_like(image, dtype=np.uint8) * 255
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w > line_threshold or h > line_threshold:
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    line_removed_image = cv2.bitwise_and(image, mask)
    
    return line_removed_image

def preprocess_image(image_path):
    image = Image.open(image_path)  
    gray_image = image.convert('L')
    gray_image_np = np.array(gray_image)    
    median_filtered = cv2.medianBlur(gray_image_np, 5)
    gaussian_blurred = cv2.GaussianBlur(median_filtered, (5, 5), 0)    
    _, binary_image = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    inverted_binary_image = cv2.bitwise_not(binary_image)    
    deskewed_image = deskew(inverted_binary_image)    
    line_removed_image = remove_large_lines(deskewed_image)    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    eroded_image = cv2.erode(line_removed_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    
    # Edge enhancement
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(dilated_image, -1, sharpen_kernel)
    
    # Contrast adjustment
    equalized_image = cv2.equalizeHist(sharpened_image)
    
    return equalized_image

def convert(img_path):

    binary_image = preprocess_image(img_path)
    h = horizontal_histogram(binary_image)

    zeros_indices = np.where(h <= 10)[0]
    local_minima = find_local_minima(h)
    local_minima = remove_elements(local_minima, zeros_indices)
    local_minima = np.where(local_minima <= threshold(h))[0]
    zeros_indices = process_consecutive_numbers(zeros_indices)
    sg = np.union1d(local_minima, zeros_indices)
    segment_lines = split_image_horizontal(binary_image, sg)
    converted = ""
    counter = 0
    for line in segment_lines:
        v = vertical_histogram(line)
        zeros_indices = np.where(v <= 1)[0]
        zeros_indices = process_consecutive_numbers(zeros_indices)
        v_sg = zeros_indices
        segment_words = split_image_vertical(line, v_sg)
        for words in segment_words:
            contours, _ = cv2.findContours(words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_size=20
            max_size = 200
            character_images = []

            for idx, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if (w < min_size and h < min_size) or (w > max_size) or h > max_size:
                    continue

                cropped = words[y:y+h, x:x+w]
                max_dim = max(cropped.shape)
                scale = 32.0 / max_dim

                try:
                    resized = cv2.resize(cropped, None, fx=scale, fy=scale)
                except:
                    continue

                new_img = np.zeros((32, 32), dtype=np.uint8)
                y_new, x_new = resized.shape
                start_x = (32 - x_new) // 2
                start_y = (32 - y_new) // 2
                new_img[start_y:start_y+y_new, start_x:start_x+x_new] = resized
                character_images.append(new_img)
            
            for character in character_images:
                #image_path = "project/temp/image"+str(counter)+".png"
                image_path = "project/temp/image.png"
                counter += 1 
                cv2.imwrite(image_path, character)
                predicted_class = model_manager.predict(image_path)
                class_name = search_key(str(predicted_class))
                if class_name == None:
                    continue
                if predicted_class in [44, 46, 47, 48, 49, 50, 51, 52]:
                    if converted and converted[-1] == ' ':
                        converted = converted[:-1] + class_name
                        continue
                if converted.endswith(' '):
                    try:
                        if converted[-2] in ["െ", "േ"]:
                            buffer = converted[-2]
                            converted = converted[:-2] + buffer + class_name
                            continue
                    except:
                        pass

                converted += class_name
            if converted.endswith(' ') or converted.endswith('\n'):
                continue
            else:
                converted += " "
        if converted.endswith('\n'):
            continue
        else:
            converted += '\n'
    converted = unicodedata.normalize('NFKD', converted)
    correction_map = {
        'ംം': 'ഃ'
    }
    for incorrect, correct in correction_map.items():
        converted = converted.replace(incorrect, correct)
    converted = unicodedata.normalize('NFKD', converted)
    return converted.strip()