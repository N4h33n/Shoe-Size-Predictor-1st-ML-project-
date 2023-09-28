# first simple machine learning project. Uses OLS model. Created with help of Microsoft Learn's intro to machine learning course.
# Not a very accurate model since dataset is small and not very diverse.
# dataset from https://doi.org/10.17605/OSF.IO/JA9DW

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("wo_men.csv")

#clean data
df.loc[df['height'] < 10, 'height'] *= 100  # assume if height less than 10, height is in meters. convert height to cm
df = df.dropna(subset=['sex', 'height', 'shoe_size'])

#obtain necessary info
men_df = df[df["sex"] == "man"][["height", "shoe_size"]]
women_df = df[df["sex"] == "woman"][["height", "shoe_size"]]
formula = "shoe_size ~ height"

#fit model
men_model = smf.ols(formula=formula, data=men_df)
women_model = smf.ols(formula=formula, data=women_df)

fit_men = men_model.fit()
fit_women = women_model.fit()

#display parameters for models
print("The following model parameters have been found for men:\n" +
        f"Line slope: {fit_men.params[1]}\n"+
        f"Line Intercept: {fit_men.params[0]}")

print("The following model parameters have been found for women:\n" +
        f"Line slope: {fit_women.params[1]}\n"+
        f"Line Intercept: {fit_women.params[0]}")

#plot data and trendline
plt.subplot(1, 2, 1)
plt.scatter(men_df["height"], men_df["shoe_size"], label="Men's Data")
plt.plot(men_df["height"], fit_men.predict(men_df), color='blue', label='Men Trendline')
plt.xlabel('Height')
plt.ylabel('Shoe Size') 
plt.title("Men's Shoe Size vs. Height")
plt.legend()

plt.subplot(1, 2, 2) 
plt.scatter(women_df["height"], women_df["shoe_size"], label="Women's Data")
plt.plot(women_df["height"], fit_women.predict(women_df), color='red', label='Women Trendline')
plt.xlabel('Height')
plt.ylabel('Shoe Size')  # Switched the labels
plt.title("Women's Shoe Size vs. Height")
plt.legend()

plt.tight_layout()
plt.show()

#save models
model_filename1 = './men_height_shoe_size_model.pkl'
joblib.dump(fit_men, model_filename1)

print("Men Model saved!")

model_filename2 = './women_height_shoe_size_model.pkl'
joblib.dump(fit_women, model_filename2)

print("Women Model saved!")