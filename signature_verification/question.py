import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# Load the trained model
model_path = r"D:\Aaa Class\sem_7\biometrics\archive (9)\CEDAR\signature_verification_model.h5"
model = load_model(model_path)

def preprocess_signature(signature, target_size=(150, 150)):
    # Resize the signature
    signature = cv2.resize(signature, target_size)
    # Convert to grayscale if it's not already
    if len(signature.shape) == 3:
        signature = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    signature = signature.astype('float32') / 255.0
    # Reshape for model input
    signature = np.expand_dims(signature, axis=-1)
    signature = np.expand_dims(signature, axis=0)
    return signature

def detect_signatures(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    signatures = []
    for contour in contours:
        # Filter contours based on area (adjust these values as needed)
        if cv2.contourArea(contour) > 1000 and cv2.contourArea(contour) < 50000:
            x, y, w, h = cv2.boundingRect(contour)
            signature = image[y:y+h, x:x+w]
            signatures.append((signature, (x, y, w, h)))
    
    return signatures

def verify_signature(signature):
    preprocessed = preprocess_signature(signature)
    prediction = model.predict(preprocessed)[0][0]
    result = "Genuine" if prediction > 0.5 else "Forged"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence

def process_document():
    # Open file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select document image", 
                                           filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    
    if not file_path:
        print("No file selected. Exiting.")
        return

    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to read the image. Please make sure it's a valid image file.")
        return

    # Detect signatures
    detected_signatures = detect_signatures(image)
    
    if not detected_signatures:
        print("No signatures detected in the document.")
        return

    # Process each detected signature
    results = []
    for idx, (signature, bbox) in enumerate(detected_signatures, 1):
        result, confidence = verify_signature(signature)
        results.append((idx, result, confidence, bbox))
        
        # Draw bounding box and label on the image
        x, y, w, h = bbox
        color = (0, 255, 0) if result == "Genuine" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        label = f"{idx}: {result} ({confidence:.2f})"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display results
    print("\nSignature Verification Results:")
    for idx, result, confidence, _ in results:
        print(f"Signature {idx}: {result} (Confidence: {confidence:.2f})")

    # Calculate overall accuracy
    correct_predictions = sum(1 for _, result, _, _ in results if result == "Genuine")
    accuracy = correct_predictions / len(results)
    print(f"\nOverall Accuracy: {accuracy:.2f}")

    # Save and display the result image
    result_path = os.path.join(os.path.dirname(file_path), "result_" + os.path.basename(file_path))
    cv2.imwrite(result_path, image)
    print(f"\nResult image saved as: {result_path}")
    
    # Display the image (you might need to adjust this based on your environment)
    cv2.imshow("Document with Detected Signatures", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_document()