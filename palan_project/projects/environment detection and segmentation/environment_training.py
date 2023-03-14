import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tarfile
import os

# Define the input shape of the images
input_shape = (224, 224, 3)

# Load the pre-trained MobileNetV2 model, excluding the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add a custom top layer for object detection
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators for the training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('train/', target_size=input_shape[:2], batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('validation/', target_size=input_shape[:2], batch_size=32, class_mode='binary')

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('mobilenetv2.h5')

# Create a tar.gz archive of the model
with tarfile.open('mobilenetv2.tar.gz', 'w:gz') as tar:
    tar.add('mobilenetv2.h5', arcname=os.path.basename('mobilenetv2.h5'))

# Delete the original model file
os.remove('mobilenetv2.h5')
