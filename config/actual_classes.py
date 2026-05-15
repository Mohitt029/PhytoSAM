# config/actual_classes.py

# Actual classes from your downloaded PlantVillage dataset
ACTUAL_PLANTVILLAGE_CLASSES = {
    "Pepper__bell___Bacterial_spot": 0,
    "Pepper__bell___healthy": 1,
    "Potato___Early_blight": 2,
    "Potato___healthy": 3,
    "Potato___Late_blight": 4,
    "Tomato_Bacterial_spot": 5,
    "Tomato_Early_blight": 6,
    "Tomato_healthy": 7,
    "Tomato_Late_blight": 8,
    "Tomato_Leaf_Mold": 9,
    "Tomato_Septoria_leaf_spot": 10,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 11,
    "Tomato__Target_Spot": 12,
    "Tomato__Tomato_mosaic_virus": 13,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 14,
}

# Reverse mapping
CLASS_ID_TO_NAME = {v: k for k, v in ACTUAL_PLANTVILLAGE_CLASSES.items()}

# Dataset statistics
DATASET_STATS = {
    "total_images": 41273,
    "num_classes": 15,
    "class_distribution": {
        "Pepper__bell___Bacterial_spot": 1994,
        "Pepper__bell___healthy": 2955,
        "Potato___Early_blight": 2000,
        "Potato___healthy": 304,
        "Potato___Late_blight": 2000,
        "Tomato_Bacterial_spot": 4254,
        "Tomato_Early_blight": 2000,
        "Tomato_healthy": 3182,
        "Tomato_Late_blight": 3816,
        "Tomato_Leaf_Mold": 1904,
        "Tomato_Septoria_leaf_spot": 3542,
        "Tomato_Spider_mites_Two_spotted_spider_mite": 3352,
        "Tomato__Target_Spot": 2808,
        "Tomato__Tomato_mosaic_virus": 746,
        "Tomato__Tomato_YellowLeaf__Curl_Virus": 6416,
    }
}