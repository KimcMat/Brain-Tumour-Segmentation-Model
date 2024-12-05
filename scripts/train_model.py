{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "96d2890c-9db6-44b9-bfa7-f7d1e76cc945",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset shape: (1314, 128, 128, 3), Labels shape: (1314, 4)\n",
      "Epoch 1/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 112ms/step - accuracy: 0.3243 - loss: 1.8676 - val_accuracy: 0.6312 - val_loss: 0.9550\n",
      "Epoch 2/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 112ms/step - accuracy: 0.7423 - loss: 0.7283 - val_accuracy: 0.6540 - val_loss: 0.7886\n",
      "Epoch 3/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 110ms/step - accuracy: 0.7965 - loss: 0.5347 - val_accuracy: 0.7757 - val_loss: 0.5774\n",
      "Epoch 4/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 111ms/step - accuracy: 0.8648 - loss: 0.3595 - val_accuracy: 0.7605 - val_loss: 0.6438\n",
      "Epoch 5/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 111ms/step - accuracy: 0.9131 - loss: 0.2321 - val_accuracy: 0.7719 - val_loss: 0.5603\n",
      "Epoch 6/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 111ms/step - accuracy: 0.9541 - loss: 0.1540 - val_accuracy: 0.7757 - val_loss: 0.6125\n",
      "Epoch 7/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 112ms/step - accuracy: 0.9770 - loss: 0.0831 - val_accuracy: 0.7795 - val_loss: 0.6004\n",
      "Epoch 8/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 111ms/step - accuracy: 0.9890 - loss: 0.0517 - val_accuracy: 0.7757 - val_loss: 0.7235\n",
      "Epoch 9/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 111ms/step - accuracy: 0.9883 - loss: 0.0411 - val_accuracy: 0.7871 - val_loss: 0.7190\n",
      "Epoch 10/10\n",
      "\u001b[1m33/33\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 111ms/step - accuracy: 1.0000 - loss: 0.0104 - val_accuracy: 0.7833 - val_loss: 0.7569\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# Set the path to your dataset folder\n",
    "dataset_folder = '/Users/matthewkim/Desktop/Brain Tumour Detection/data'\n",
    "\n",
    "# Define a function to load and preprocess images\n",
    "def load_images():\n",
    "    images = []\n",
    "    labels = []\n",
    "    tumor_types = {'gliomas': 0, 'meningiomas': 1, 'pituitary': 2, 'no': 3}  # Label mapping\n",
    "\n",
    "    for label, class_idx in tumor_types.items():\n",
    "        folder_path = os.path.join(dataset_folder, 'yes' if label != 'no' else 'no', label if label != 'no' else '')\n",
    "\n",
    "        if not os.path.exists(folder_path):\n",
    "            print(f\"Folder not found: {folder_path}\")\n",
    "            continue\n",
    "\n",
    "        for filename in os.listdir(folder_path):\n",
    "            image_path = os.path.join(folder_path, filename)\n",
    "            img = cv2.imread(image_path)\n",
    "\n",
    "            if img is None:\n",
    "                print(f\"Warning: Could not read image {filename}\")\n",
    "                continue\n",
    "\n",
    "            # Resize and normalize the image\n",
    "            img = cv2.resize(img, (128, 128))\n",
    "            img = img / 255.0\n",
    "\n",
    "            images.append(img)\n",
    "            labels.append(class_idx)\n",
    "\n",
    "    images = np.array(images)\n",
    "    labels = np.array(labels)\n",
    "    labels = to_categorical(labels, len(tumor_types))  # Convert labels to one-hot encoding\n",
    "    return images, labels\n",
    "\n",
    "# Load the dataset\n",
    "X, y = load_images()\n",
    "print(f\"Dataset shape: {X.shape}, Labels shape: {y.shape}\")\n",
    "\n",
    "# Split the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Build the CNN model\n",
    "model = tf.keras.Sequential([\n",
    "    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),\n",
    "    tf.keras.layers.MaxPooling2D((2, 2)),\n",
    "    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    tf.keras.layers.MaxPooling2D((2, 2)),\n",
    "    tf.keras.layers.Flatten(),\n",
    "    tf.keras.layers.Dense(128, activation='relu'),\n",
    "    tf.keras.layers.Dense(4, activation='softmax')  # 4 output classes\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Train the model\n",
    "model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))\n",
    "\n",
    "# Save the trained model\n",
    "model.save('/Users/matthewkim/Desktop/Brain Tumour Detection/models/tumor_classification_model.h5')\n",
    "print(\"Model saved successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37b2c6a3-a153-4f6f-9912-11b3d1ac02dd",
   "metadata": {},
   "outputs": [],
   "source": []
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
