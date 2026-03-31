import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from preprocessing import load_data

# Load data generators with a target image size of 300x300
train_generator, val_generator = load_data(img_size=(300, 300), batch_size=16)

# Get class indices and compute class weights based on the training data
class_indices = train_generator.class_indices
class_labels = np.array([class_indices[class_name] for class_name in train_generator.class_indices.keys()])
y_train = train_generator.classes
class_weights = class_weight.compute_class_weight('balanced', classes=class_labels, y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_indices))}

# Build the EfficientNet model
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False  # Freeze the base model

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax')
])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint(
    'best_leaf_disease_model.keras',  # Saves model with best val_loss
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, checkpoint]
)

# Save the final model
model.save("C:/f/project/final_leaf_disease_model.keras")
print("Model saved as 'final_leaf_disease_model.keras'")

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
