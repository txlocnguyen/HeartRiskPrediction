# Loc Nguyen
# lngu242@wgu.edu
import pandas as pand
import numpy as nump
import streamlit as strl
import matplotlib.pyplot as mplpyp
import seaborn as sea
from sklearn import linear_model

strl.title('Heart Disease Risk Calculator')
myData = pand.read_csv('heart_2020_cleaned.csv')

def perctDiseases(dt):
    totCounts = dt['HeartDisease'].count()
    noOfDiseases = dt[dt['HeartDisease'] == 1]['HeartDisease'].count()
    percentOfDiseases = (noOfDiseases / totCounts) * 100
    return percentOfDiseases

# create a list of age categories
age_categories = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                  '60-64', '65-69', '70-74', '75-79', '80 or older']

# create a dictionary to store disease percent for each age group
disease_perCat = {}
for age in age_categories:
    disease_perCat[age] = perctDiseases(myData[myData['AgeCategory'] == age])

# access the disease percent of each age group by its key
diseaseAgeCat18 = disease_perCat['18-24']
diseaseAgeCat25 = disease_perCat['25-29']
diseaseAgeCat30 = disease_perCat['30-34']
diseaseAgeCat35 = disease_perCat['35-39']
diseaseAgeCat40 = disease_perCat['40-44']
diseaseAgeCat45 = disease_perCat['45-49']
diseaseAgeCat50 = disease_perCat['50-54']
diseaseAgeCat55 = disease_perCat['55-59']
diseaseAgeCat60 = disease_perCat['60-64']
diseaseAgeCat65 = disease_perCat['65-69']
diseaseAgeCat70 = disease_perCat['70-74']
diseaseAgeCat75 = disease_perCat['75-79']
diseaseAgeCat80 = disease_perCat['80 or older']

# Mapping disease percentage to age categories
diseaseByAge = {'18-24': diseaseAgeCat18, '25-29': diseaseAgeCat25, '30-34': diseaseAgeCat30, '35-39': diseaseAgeCat35,
'40-44': diseaseAgeCat40, '45-49': diseaseAgeCat45, '50-54': diseaseAgeCat50, '55-59': diseaseAgeCat55,
'60-64': diseaseAgeCat60, '65-69': diseaseAgeCat65, '70-74': diseaseAgeCat70, '75-79': diseaseAgeCat75,
'80 or older': diseaseAgeCat80}

# Plotting line graph to show heart disease vs. age relationship
strl.write('Age positively affects heart disease risk:')
graphAgeLn = mplpyp.figure(figsize=(13, 5))
mplpyp.xlabel('Age Group')
mplpyp.ylabel('Risk Percentage')
mplpyp.plot(diseaseByAge.keys(), diseaseByAge.values(), marker='o')
strl.pyplot(graphAgeLn)
 
# Calculating disease percentage based on sex
diseasePerMale = perctDiseases(myData[myData['Sex'] == 'Male'])
diseasePerFemale = perctDiseases(myData[myData['Sex'] == 'Female'])

# Plotting bar graph to show heart disease risk for males and females
strl.write('Males have higher heart disease risk:')
sex_fig = mplpyp.figure(figsize=(11, 5))
sea.barplot(x=['Female', 'Male'], y=[diseasePerFemale, diseasePerMale])
mplpyp.ylabel('Risk Percentage')
strl.pyplot(sex_fig)

# Determine the disease percentages by BMI category
disIfHealthy = perctDiseases(myData[myData['BMI'] < 25])
disIfOverweight = perctDiseases(myData[(myData['BMI'] >= 25) & (myData['BMI'] < 30)])
disIfObesity = perctDiseases(myData[myData['BMI'] >= 30])

# Find the disease percentages by smoking status
disIfSmoking = perctDiseases(myData[myData['Smoking'] == 'Yes'])
disIfNoSmoking = perctDiseases(myData[myData['Smoking'] == 'No'])

# Define the logistic regression variables
bmi = myData['BMI']
heartDiseaseYesNo = myData['HeartDisease']

# Display the graphs to show heart disease risk by BMI and smoking
strl.write('Modifiable lifestyle choices such as BMI and smoking can also impact heart disease risk:')

# Logistic regression plot
plotForBMI = mplpyp.figure()
sea.regplot(x=bmi, y=heartDiseaseYesNo, data=myData, logistic=True, ci=None)
mplpyp.ylabel('Risk Percentage')
strl.pyplot(plotForBMI)

# Bar graph
plotForSmoking = mplpyp.figure()
sea.barplot(x=['Non-Smoker', 'Smoker'], y=[disIfNoSmoking, disIfSmoking])
mplpyp.ylabel('Risk Percentage')
strl.pyplot(plotForSmoking)

# Present user with dropdown menus to enter their stats
strl.header("Assess Your Risk Of Having a Heart Disease")
age_category = strl.selectbox("Age Group",
('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
'60-64', '65-69', '70-74', '75-79', '80 or older'))
gender = strl.selectbox("Gender", ('Male', 'Female'))
bmi = strl.selectbox("BMI", tuple(range(15,101)))
smokes = strl.selectbox("Smoking Status", ('Yes', 'No'))

# Create a population with the same characteristics as the user
population = myData[(myData['AgeCategory'] == age_category) & (myData['Sex'] == gender) &
(myData['Smoking'] == smokes)]

# Get non-smoker population with similar age and gender as the user
nonSmokerPopulation = myData[(myData['AgeCategory'] == age_category) & (myData['Sex'] == gender) &
(myData['Smoking'] == 'No')]

# Train a logistic regression model based on the population
x = population['BMI'].values.reshape(-1, 1)
y = population['HeartDisease'].values
logisticReg = linear_model.LogisticRegression()
logisticReg.fit(x, y)

# Create a function that computes heart disease probability using provided BMI
def risk_probability(logisticReg, bmi):
    oddsLogiscticReg = logisticReg.coef_ * bmi + logisticReg.intercept_
    calculatedOdds = nump.exp(oddsLogiscticReg)
    finalProbability = calculatedOdds / (1 + calculatedOdds)
    return float(finalProbability[0] * 100)

# Assess the user's risk of having a heart disease
riskOfUser = risk_probability(logisticReg, bmi)

# Show Information About Risk of Heart Disease
if strl.button("Determine Risk"):
    risk = ("%.2f" % riskOfUser) + "%."
    strl.write(f"Your heart disease risk is {risk}")
if bmi < 25 and smokes == 'No':
    strl.write("Consult with your doctor and refer to the chart below.")
    riskPlot = mplpyp.figure(figsize=(11, 5))
    sea.barplot(x=["Risk"], y=[riskOfUser])
    mplpyp.ylabel("Risk (%)")
    mplpyp.ylim(0, riskOfUser * 2)
    mplpyp.title("Your Heart Disease Risk")
    strl.pyplot(riskPlot)

if bmi > 24 and smokes == 'No':
    strl.write("You can lower your risk by reaching a BMI of 24. Refer to the chart below.")
    riskIfHealthyBMI = risk_probability(logisticReg, 24)
    riskIfHealthyBMI = ("%.2f" % riskIfHealthyBMI) + '%.'
    strl.write(f"Your heart disease risk would reduce to {riskIfHealthyBMI}")
    riskPlot = mplpyp.figure(figsize=(11, 5))
    sea.barplot(x=["Current BMI", "BMI 24"], y=[riskOfUser, riskIfHealthyBMI])
    mplpyp.ylabel("Risk (%)")
    mplpyp.title("Your Heart Disease Risk")
    strl.pyplot(riskPlot)

if bmi < 25 and smokes == 'Yes':
    strl.write("You can lower your risk by quitting smoking. Refer to the chart below.")
    usrX = nonSmokerPopulation['BMI'].values.reshape(-1, 1)
    usrY = nonSmokerPopulation['HeartDisease'].values
    usrLogisticReg = linear_model.LogisticRegression()
    usrLogisticReg.fit(usrX, usrY)
    usrProbability = risk_probability(usrLogisticReg, bmi)
    riskIfNoSmoke = ("%.2f" % usrProbability) + '%.'
    strl.write(f"Quitting smoking reduces your heart disease risk to {riskIfNoSmoke}")
    riskPlot = mplpyp.figure(figsize=(11, 5))
    sea.barplot(x=["Smoking", "Non-Smoking"], y=[riskOfUser, usrProbability])
    mplpyp.ylabel("Risk (%)")
    mplpyp.title("Your Heart Disease Risk")
    strl.pyplot(riskPlot)

if bmi > 24 and smokes == 'Yes':
    strl.write('Reducing your BMI could lower your risk.')
    user_prob_healthy = risk_probability(logisticReg, 24)
    strl.write('Achieving a BMI of 24 could decrease your heart disease risk to', ("%.2f" % user_prob_healthy), '%.')
    strl.write('Quitting smoking is also a way to decrease the risk.')
    usrX = nonSmokerPopulation['BMI'].values.reshape(-1, 1)
    usrY = nonSmokerPopulation['HeartDisease'].values
    usrLogisticReg = linear_model.LogisticRegression()
    usrLogisticReg.fit(usrX, usrY)
    usrProbability = risk_probability(usrLogisticReg, bmi)
    strl.write('Stopping smoking could reduce your heart disease risk to', ("%.2f" % usrProbability), '%.')
    strl.write('Please check the chart below and talk to your doctor if you have any questions.')
    riskPlot = mplpyp.figure(figsize=(11, 5))
    sea.barplot(x=['Current Risk', 'BMI 24 Risk', 'Non-Smoking Risk'],
    y=[logisticReg, user_prob_healthy, usrProbability])
    mplpyp.ylabel('Risk Percentage')
    mplpyp.title('Your Heart Disease Risk')
    strl.pyplot(riskPlot)
