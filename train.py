import os
import pickle
import face_recognition
from sklearn import svm

# Prepare training data
encodings = []
names = []

# Path to the dataset folder
dataset_path = 'dataset'

# Loop over each person's folder in the dataset
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue  # Skip if it's not a directory

    # Loop over the images inside each person's folder
    for img_name in os.listdir(person_dir):
        # Only process image files (jpg, png, jpeg)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Construct the full image path
        img_path = os.path.join(person_dir, img_name)

        try:
            # Load the image
            image = face_recognition.load_image_file(img_path)

            # Get the face encodings from the image
            face_encs = face_recognition.face_encodings(image)

            if face_encs:  # If there are any face encodings
                encodings.append(face_encs[0])  # Take the first face encoding (assuming one face per image)
                names.append(person_name)  # Append the person's name (folder name) to the list

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Check if there are any encodings, if not exit
if not encodings:
    print("No faces found, exiting.")
    exit()

# Train the SVM model with the face encodings and associated names
clf = svm.SVC(probability=True)
clf.fit(encodings, names)

# Save the trained model to a file
with open('trained_svm_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved as 'trained_svm_model.pkl'")
