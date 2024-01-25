from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def Cancer(request):
    path="C:\\Users\\Ranjitha\\OneDrive\\Desktop\\KTS_Internship\\Data\\bdiag.csv"
    data=pd.read_csv(path)
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    inputs=data.drop(['diagnosis','id'],'columns')
    output=data['diagnosis']
    x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model=SVC() 
    model.fit(x_train_scaled,y_train)
    y_pred = model.predict(x_test_scaled)
    acc=100*(accuracy_score(y_test, y_pred))
    if(request.method=="POST"):
        data=request.POST  
        radius_mean=float(data.get('txtradius_mean'))
        texture_mean=float(data.get('txttexture_mean'))
        perimeter_mean=float(data.get('txtperimeter_mean'))
        area_mean=float(data.get('txtarea_mean'))
        smoothness_mean=float(data.get('txtsmoothness_mean'))
        compactness_mean=float(data.get('txtcompactness_mean'))
        concavity_mean=float(data.get('txtconcavity_mean'))
        concave_points_mean=float(data.get('txtconcave points_mean'))
        symmetry_mean=float(data.get('txtsymmetry_mean'))
        fractal_dimension_mean=float(data.get('txtfractal_dimension_mean'))
        radius_se=float(data.get('txtradius_se'))
        texture_se=float(data.get('txttexture_se'))
        perimeter_se=float(data.get('txtperimeter_se'))
        area_se=float(data.get('txtarea_se'))
        smoothness_se=float(data.get('txtsmoothness_se'))
        compactness_se=float(data.get('txtcompactness_se'))
        concavity_se=float(data.get('txtconcavity_se'))
        concave_points_se=float(data.get('txtconcave points_se'))
        symmetry_se=float(data.get('txtsymmetry_se'))
        fractal_dimension_se=float(data.get('txtfractal_dimension_se'))
        radius_worst=float(data.get('txtradius_worst'))
        texture_worst=float(data.get('txttexture_worst'))
        perimeter_worst=float(data.get('txtperimeter_worst'))
        area_worst=float(data.get('txtarea_worst'))
        smoothness_worst=float(data.get('txtsmoothness_worst'))
        compactness_worst=float(data.get('txtcompactness_worst'))
        concavity_worst=float(data.get('txtconcavity_worst'))
        concave_points_worst=float(data.get('txtconcave points_worst'))
        symmetry_worst=float(data.get('txtsymmetry_worst'))
        fractal_dimension_worst=float(data.get('txtfractal_dimension_worst'))
        if('submit' in request.POST):
            results=model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])
            res=int(results[0])
            if res==1:
                result="Malignant"
            else:
                result="Benign"
            return render(request,'Cancer.html',context={'result':result,'acc':acc})
    return render(request,'Cancer.html')
def company(request):
    return render(request,'company.html')
def index1(request):
    return render(request,'index1.html')
