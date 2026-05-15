"""
Dataset Configuration for PlantVillage
Contains all dataset-specific parameters and class mappings
"""

# PlantVillage Dataset Paths (update these based on your download location)
PLANTVILLAGE_PATHS = {
    "original": "./data/plantvillage/raw",  # Original dataset
    "augmented": "./data/plantvillage/augmented",  # Augmented version
    "processed": "./data/plantvillage/processed",  # Preprocessed
    "splits": "./data/plantvillage/splits",  # Train/val/test splits
}

# 26 Disease Classes (PlantVillage categories)
PLANT_DISEASE_CLASSES = {
    # Apple
    "Apple___Apple_scab": 0,
    "Apple___Black_rot": 1,
    "Apple___Cedar_apple_rust": 2,
    "Apple___healthy": 3,
    
    # Blueberry
    "Blueberry___healthy": 4,
    
    # Cherry
    "Cherry___Powdery_mildew": 5,
    "Cherry___healthy": 6,
    
    # Corn (Maize)
    "Corn___Cercospora_leaf_spot": 7,
    "Corn___Common_rust": 8,
    "Corn___Northern_Leaf_Blight": 9,
    "Corn___healthy": 10,
    
    # Grape
    "Grape___Black_rot": 11,
    "Grape___Esca_(Black_Measles)": 12,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 13,
    "Grape___healthy": 14,
    
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": 15,
    
    # Peach
    "Peach___Bacterial_spot": 16,
    "Peach___healthy": 17,
    
    # Pepper
    "Pepper,_bell___Bacterial_spot": 18,
    "Pepper,_bell___healthy": 19,
    
    # Potato
    "Potato___Early_blight": 20,
    "Potato___Late_blight": 21,
    "Potato___healthy": 22,
    
    # Raspberry
    "Raspberry___healthy": 23,
    
    # Soybean
    "Soybean___healthy": 24,
    
    # Squash
    "Squash___Powdery_mildew": 25,
    
    # Strawberry
    "Strawberry___Leaf_scorch": 26,
    "Strawberry___healthy": 27,
    
    # Tomato
    "Tomato___Bacterial_spot": 28,
    "Tomato___Early_blight": 29,
    "Tomato___Late_blight": 30,
    "Tomato___Leaf_Mold": 31,
    "Tomato___Septoria_leaf_spot": 32,
    "Tomato___Spider_mites_Two-spotted_spider_mite": 33,
    "Tomato___Target_Spot": 34,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 35,
    "Tomato___Tomato_mosaic_virus": 36,
    "Tomato___healthy": 37,
}

# Reduced to 26 classes for HEMSAM (combining similar diseases)
HEMSAM_CLASSES = {
    "Apple_Scab": 0,
    "Apple_Black_Rot": 1,
    "Apple_Cedar_Rust": 2,
    "Apple_Healthy": 3,
    "Cherry_Powdery_Mildew": 4,
    "Corn_Cercospora": 5,
    "Corn_Common_Rust": 6,
    "Corn_Northern_Blight": 7,
    "Grape_Black_Rot": 8,
    "Grape_Esca": 9,
    "Grape_Leaf_Blight": 10,
    "Orange_Huanglongbing": 11,
    "Peach_Bacterial_Spot": 12,
    "Pepper_Bacterial_Spot": 13,
    "Potato_Early_Blight": 14,
    "Potato_Late_Blight": 15,
    "Squash_Powdery_Mildew": 16,
    "Strawberry_Leaf_Scorch": 17,
    "Tomato_Bacterial_Spot": 18,
    "Tomato_Early_Blight": 19,
    "Tomato_Late_Blight": 20,
    "Tomato_Leaf_Mold": 21,
    "Tomato_Septoria": 22,
    "Tomato_Spider_Mites": 23,
    "Tomato_YLCV": 24,
    "Healthy": 25,
}

# Reverse mapping for inference
CLASS_ID_TO_NAME = {v: k for k, v in HEMSAM_CLASSES.items()}

# Image processing parameters
IMAGE_CONFIG = {
    "input_size": 224,  # SAM default
    "mask_size": 256,   # Mask output size
    "mean": [0.485, 0.456, 0.406],  # ImageNet mean
    "std": [0.229, 0.224, 0.225],   # ImageNet std
}

# Dataset split ratios
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

# DataLoader parameters
DATALOADER_CONFIG = {
    "batch_size": 8,  # Adjust based on GPU memory
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
}

# Augmentation parameters
AUGMENTATION_CONFIG = {
    "train": {
        "rotation_degrees": 45,
        "brightness_range": 0.2,  # Changed from tuple to float for compatibility
        "contrast_range": 0.2,
        "hue_range": 0.1,
        "saturation_range": 0.2,
        "gaussian_blur_kernel": (3, 3),
        "elastic_alpha": 50,
        "elastic_sigma": 5,
        "cutout_holes": 8,
        "cutout_size": 16,
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.3,
    },
    "val": {
        "only_resize": True,
    }
}