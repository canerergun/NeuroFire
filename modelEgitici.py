import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

# 🔧 Ayarlar
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "fire_dataset"
MODEL_PATH = "NeuroFire.h5"

# 📁 Eğitim ve Doğrulama İçin Veri Artırma
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# ⚖️ class_weight Hesaplama
labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))
print("🎯 Class Weights:", class_weights_dict)

# 🧠 Gelişmiş CNN Modeli
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 🚀 Model Eğitimi
print("\n📢 Model eğitiliyor...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights_dict
)

# 💾 Model Kaydetme
model.save(MODEL_PATH)
print(f"\n✅ Eğitim tamamlandı ve model '{MODEL_PATH}' olarak kaydedildi.")

# 📈 Eğitim Grafikleri
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Eğitim Başarısı")
plt.plot(epochs_range, val_acc, label="Doğrulama Başarısı")
plt.title("Başarı Grafiği")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Eğitim Kaybı")
plt.plot(epochs_range, val_loss, label="Doğrulama Kaybı")
plt.title("Kayıp Grafiği")
plt.legend()

plt.tight_layout()
plt.savefig("egitim_grafigi.png")
print("📊 Grafik 'egitim_grafigi.png' olarak kaydedildi.")
