{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37b2c6a3-a153-4f6f-9912-11b3d1ac02dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Define test folder path\n",
    "test_folder = '/Users/matthewkim/Desktop/Test_Images'\n",
    "\n",
    "# Function to preprocess the image\n",
    "def preprocess_image(image_path):\n",
    "    img = cv2.imread(image_path)\n",
    "    if img is None:\n",
    "        print(f\"Warning: Unable to read image at {image_path}. Skipping.\")\n",
    "        return None  # Return None for invalid images\n",
    "    img = cv2.resize(img, (128, 128))  # Resize to match the model input\n",
    "    img = img / 255.0  # Normalize pixel values\n",
    "    return np.expand_dims(img, axis=0)  # Expand dimensions for model input\n",
    "\n",
    "# Function to visualize predictions\n",
    "def visualize_predictions():\n",
    "    # Labels for the classes\n",
    "    class_labels = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']\n",
    "    \n",
    "    for filename in os.listdir(test_folder):\n",
    "        image_path = os.path.join(test_folder, filename)\n",
    "        \n",
    "        # Preprocess the image\n",
    "        img = preprocess_image(image_path)\n",
    "        if img is None:\n",
    "            continue  # Skip invalid images\n",
    "        \n",
    "        # Make a prediction\n",
    "        prediction = model.predict(img)\n",
    "        predicted_label = np.argmax(prediction, axis=1)[0]\n",
    "        \n",
    "        # Display the image and the prediction\n",
    "        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))\n",
    "        plt.title(f\"Predicted: {class_labels[predicted_label]}\")\n",
    "        plt.axis('off')\n",
    "        plt.show()\n",
    "\n",
    "# Run the visualization\n",
    "visualize_predictions()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
