from ultralytics import YOLO

# Load trained model
model = YOLO(r"C:\Users\prera\Downloads\runs\runs\classify\train\weights\best.pt")

# Disease/stress -> nutrient mapping
disease_to_nutrient = {
    "Bacterial Blight": "Possible Potassium Deficiency",
    "Curl Virus": "Possible Zinc Deficiency",
    "Healthy Leaf": "No Deficiency",
    "Herbicide Growth Damage": "Nutrient not applicable (chemical injury)",
    "Leaf Hopper Jassids": "Possible Nitrogen Stress",
    "Leaf Redding": "Possible Phosphorus Deficiency",
    "Leaf Variegation": "Possible Magnesium Deficiency"
}

# Predict
results = model(r"C:\Users\prera\Downloads\runs\runs\classify\train\test_leaf.jpg")

# Top predicted class
pred_class = results[0].names[results[0].probs.top1]

# Lookup nutrient
nutrient = disease_to_nutrient.get(pred_class, "Unknown")

print("Detected:", pred_class)
print("Deficient Nutrient:", nutrient)