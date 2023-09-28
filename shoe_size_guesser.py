# A sample terminal application that uses the trained models to predict shoe size based on height

import joblib
import math
import pandas as pd

gender = input("Enter gender (m or f): ")
height = float(input("Enter height in cm: "))
height_data = pd.DataFrame({"height": [height]})

if gender == "m":
    model_loaded = joblib.load('./men_height_shoe_size_model.pkl')
    approx_shoe = model_loaded.predict(height_data)
else:
    model_loaded = joblib.load('./women_height_shoe_size_model.pkl')
    approx_shoe = model_loaded.predict(height_data)
    
print(f"Approximate shoe size: {math.ceil(approx_shoe[0])} EUR")