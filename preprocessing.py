import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_dir="C:/f/PlantVillage/PlantVillage", img_size=(300, 300), batch_size=32, validation_split=0.2):
    # Check that the base directory exists
    if not os.path.isdir(base_dir):
        raise ValueError(f"Directory '{base_dir}' does not exist. Please provide a valid dataset directory.")

    # Instantiate an ImageDataGenerator for training with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,        # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split  # Set the validation split
    )

    # Create training and validation generators
    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # For multi-class classification
        subset='training',    # Training subset
        shuffle=True          # Shuffle training data
    )

    val_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # For multi-class classification
        subset='validation',  # Validation subset
        shuffle=True          # Shuffle validation data
    )

    # Debug information
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")

    return train_generator, val_generator
