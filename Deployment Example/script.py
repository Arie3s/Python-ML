import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_star_rating import st_star_rating

path = "C:\Users\walit\OneDrive\Documents\GitHub\Python ML\Deployment Example\"
cities=pd.read_excel(path+"average_salary.xlsx", sheet_name=1)
cities['normalized']=cities.Population/cities.Population.sum()

with open(path+"model.pkl", "rb") as f:
    model = pickle.load(f)

html_temp="""
<div style="background-color:SkyBlue; padding:13px">
<h1 style="color:black";text-align:center;">Probability of Default Prediction</h1>
</div>
"""
st.markdown(html_temp,unsafe_allow_html = True)
Gender = st.selectbox('Gender', ('Male', "Female"))
Married = st.selectbox("Marriage",("Widow/Divorced","Single","Married"))
Education = st.selectbox("Education", ("No Education", "Primary School", "Middle School",
"High School", "Higher Secondary School", "Degree College", "University"))
Job = st.selectbox("Job Category",("Not Employed", "Elementary Occupation", 'Crafts & Related workers',
   'Plant/Machine Operators', 'Service and Sales workers',
   'Skilled in agriculture, forestry, & fishery','Technician & Associate Professionals',
   'Clinical support workers','Professionals', 'Managers'))
City = st.selectbox("City",cities.City)
Age = st.number_input("Age", min_value=20, max_value=80,)
Income = st.number_input("Income", min_value=5000)
Loan = st.number_input("Loan Amount", min_value=5000, max_value=2000000)
Duration = st.slider("Payment Duration in Months", min_value=6, max_value=12)
month_on_book = st.slider("Month on Book", min_value=1, max_value=24)

city_pop= cities.loc[cities.City==City, 'normalized'].values[0]

if Gender=='Male':
    Male=1
    Female=0
else:
    Male=0
    Female=1

No_Education = Primary=Middle=High=Higher= College= University=0

if Education=='No Education':
    No_Education = 1
elif Education=="Primary School":
    Primary=1
elif Education=="Middle School":
    Middle=1,
elif Education=="High School":
    High=1
elif Education == "Higher Secondary School":
    Higher=1
elif Education == "Degree College":
    College=1
else:
    University=1

no_job=elem= crafts= operator= sales= agri= tech= clinic= prof= manager=0

if Job=='Not Employed':
    no_job=1
elif Job=="Elementary Occupation":
    elem=1
elif Job=='Crafts & Related workers':
    crafts=1
elif Job=='Plant/Machine Operators':
    operator=1
elif Job=='Service and Sales workers':
    sales=1
elif Job=='Skilled in agriculture, forestry, & fishery':
    agri=1
elif Job=='Technician & Associate Professionals':
    tech=1
elif Job=='Clinical support workers':
    clinic=1
elif Job=='Professionals':
    prof=1
else:
    manager=1

widow=single=married=0

if Married=='Widow/Divorced':
    widow=1
elif Married=='Single':
    single=1
else:
    married=1


mean=pd.Series([224157.736904, 77083.198292],  index=["Loan", "Income"])
std=pd.Series([322364.875661,  44396.293237],  index=["Loan", "Income"])

Income=(Income-mean['Income'])/std['Income']
Loan=(Loan-mean['Loan'])/std['Loan']
Age=(Age-20)/(80-20)
Duration=(Duration-6)/(12-6)
month_on_book=(month_on_book-1)/(24-1)


test=np.array([[Age, Income, city_pop, Loan, Duration, month_on_book,
       Male, Female, No_Education, Primary, Middle, High, Higher, College, University,
       no_job, elem, crafts, operator, sales, agri, tech, clinic, prof, manager,
       widow, single, married]])

result=""
ks=pd.read_csv(path+"KS.csv")

if st.button("Predict"):
    result=model.predict_proba(test)
    score=1000-int(result[0,1]*1000)
    st.progress(score/1000, "Applicant Credit Score")
    st.success("{} out of 1000".format(score))
    x=0
    for i in range(5):
        if result[0,1]>=ks.loc[i,'min_prob'] and result[0,1]<=ks.loc[i,'max_prob']:
            x=i+1
            break

    st_star_rating("Applicant Rating", maxValue=5, defaultValue=x,read_only=True)

    ind=[]
    for i in range(len(model.feature_names_in_)):
        if test[0,i]!=0 and model.coef_[0,i]*test[0,i]>0:
            if model.feature_names_in_[i] in ['Income', 'Month on book',
                    'Payment duration in months', 'Not Employed', 'No Education',
                    'Primary', 'Middle School', 'High School','Higher Secondary School',]:
                ind.append(i)

    if len(ind)!=0:
        st.write("Applicant can increase credit score by improving following factors.")
        data=model.coef_[0,ind]*test[0,ind]
    for i in range(len(ind)):
        st.progress(data[i]/np.sum(data), model.feature_names_in_[ind[i]])


























