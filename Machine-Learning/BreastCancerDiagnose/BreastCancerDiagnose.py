from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from sklearn import preprocessing
from IPython.display import Image 
from pydot import graph_from_dot_data
from statistics import mean 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Carga del conjunto de datos
data = pd.read_csv('wdbc.data', header = None)
df = pd.DataFrame(data)
cols = ["IDNum","Diagnosis"]
statistic_indicator = ["mean","ste", "worst"]
features = ["Radius","Texture","Perimeter","Area","Smoothness","Compactness","Concavity","ConcavePoints","Symmetry","FractalDimension"]
for s in statistic_indicator:
    for f in features:
        cols.append(s+"_"+f)
data.columns = cols
att_num = len(data.columns)
reg_num = len(data)
class_ocurrences = pd.Index(df['Diagnosis'])
#Análisis inicial
print("==== INFORMACIÓN PRELIMINAR ====")
print("---------------------------------")
print("Número de Registros:",reg_num)
print("Número de Atributos:",att_num)
print("Ocurrencias de cada clase (Benigno: B, Maligno: M)")
print(class_ocurrences.value_counts())
print("Registros con datos faltantes:",sum(df.isna().sum())) 
print("Registros eliminados:",sum(df.isna().sum()))

#Holdout con el método de clasficiación de Árboles de Decisión
#Inicialización de arreglos donde se acumularán las métricas resultantes de cada iteración
h_accuracy_gini = []
h_precision_gini = []
h_recall_gini = []
h_f1score_gini = []
#-------------------------#
h_accuracy_infogain = []
h_precision_infogain = []
h_recall_infogain = []
h_f1score_infogain = []
#Definimos 10 iteraciones de Holdout para el entrenamiento del modelo
HOLDOUT = 10
for i in range(HOLDOUT):
    #Partición del DataFrame en training (75%) y test (25%).
    le = preprocessing.LabelEncoder()
    df_copy = df.apply(le.fit_transform)
    X = df_copy.drop(['IDNum','Diagnosis'],axis=1)
    y = df_copy['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    
    #Entrenamiento del modelo de clasificación usando los datos designados para training 
    #Se usa función Gini e information gain para determinar el mejor método de decisión 
    dt_gini = DecisionTreeClassifier(criterion='gini')
    dt_infogain = DecisionTreeClassifier(criterion='entropy')
    dt_gini.fit(X_train, y_train)
    dt_infogain.fit(X_train, y_train)
    
    #Se hace la predicción sobre los datos de prueba y se calculan los indicadores de evaluación de aprendizaje
    #Positive Class: B (Benigno)
    #Negative Class: M (Maligno)
    pred_gini = dt_gini.predict(X_test)
    pred_infogain = dt_infogain.predict(X_test)
    conf_mat_gini = confusion_matrix(y_test,pred_gini)
    conf_mat_infogain = confusion_matrix(y_test,pred_infogain)
    
    #Cálculo de métricas para función gini
    tn, fp, fn, tp = conf_mat_gini.ravel()
    accuracy_gini = (tp+tn)/(tp+tn+fp+fn)
    ps_gini = precision_score(y_test,pred_gini)
    rs_gini = recall_score(y_test,pred_gini)
    f1s_gini = f1_score(y_test,pred_gini)
    
    #Acumulación de métricas de evaluación de aprendizaje para función gini:
    h_accuracy_gini.append(accuracy_gini)
    h_precision_gini.append(ps_gini)
    h_recall_gini.append(rs_gini)
    h_f1score_gini.append(f1s_gini)
    
    #Cálculo de métricas para information gain:
    tn, fp, fn, tp = conf_mat_infogain.ravel()
    accuracy_infogain = (tp+tn)/(tp+tn+fp+fn)
    ps_infogain = precision_score(y_test,pred_infogain)
    rs_infogain = recall_score(y_test,pred_infogain)
    f1s_infogain = f1_score(y_test,pred_infogain)

    #Acumulación de métricas de evaluación de aprendizaje para information gain:
    h_accuracy_infogain.append(accuracy_infogain)
    h_precision_infogain.append(ps_infogain)
    h_recall_infogain.append(rs_infogain)
    h_f1score_infogain.append(f1s_infogain)

print("==== REPORTE HOLDOUT ====")
print("---------------------------")
print("Métricas de aprendizaje para función gini")
print("-------------------------------------------")
print("Accuracy:",mean(h_accuracy_gini))
print("Precision:",mean(h_precision_gini))
print("Recall:",mean(h_recall_gini))
print("F1_Score:",mean(h_f1score_gini))
print("---------------------------")
print("Métricas de aprendizaje para information gain")
print("-------------------------------------------")
print("Accuracy:",mean(h_accuracy_infogain))
print("Precision:",mean(h_precision_infogain))
print("Recall:",mean(h_recall_infogain))
print("F1_Score:",mean(h_f1score_infogain))
print("========== GRÁFICOS: FUNCIÓN GINI =========")

plt.subplots(1, figsize=(5,5))
plt.title('Precision Per Hold-out Iteration - Gini Function')
plt.scatter([i for i in range (10)], h_precision_gini)
plt.ylabel("Precision")
plt.xlabel("Hold-out Iteracion")
plt.show()
print("----------------------------------")
lmin = int(min(h_precision_gini)*10000)/10000
lmax = int(max(h_precision_gini)*10000)/10000
per_lmin = lmin*100
per_lmax = lmax*100
print("Del gráfico se puede observar que la capacidad del modelo para retornar solamente instancias relevantes se encuentra entre el "+str(per_lmin)+"% y "+str(per_lmax)+"%")
print("Es decir, existe entre "+str(int((100-per_lmax)*100)/100)+"% y "+str(int((100-per_lmin)*100)/100)+"% de probabilidad de que el modelo entregue un diagnóstico erróneo")
print("----------------------------------")
plt.subplots(1, figsize=(5,5))
plt.title('Recall Per Hold-out Iteration - Gini Function')
plt.scatter([i for i in range (10)], h_recall_gini)
plt.ylabel("Recall")
plt.xlabel("Hold-out Iteracion")
plt.show()
print("----------------------------------")
m_recall = int(mean(h_recall_gini)*10000)/10000
per_m = int((m_recall*100)*100)/100
print("En promedio, el modelo solamente es capaz de diagnosticar correctamente un tumor en el "+str(per_m)+"% de los casos")
print("----------------------------------")
plt.subplots(1, figsize=(5,5))
plt.title('F1 Score Per Hold-out Iteration - Gini Function')
plt.scatter([i for i in range (10)], h_f1score_gini)
plt.ylabel("F1 Score")
plt.xlabel("Hold-out Iteracion")
plt.show()
print("----------------------------------")
m_f1score = int(mean(h_f1score_gini)*10000)/10000
per_m = int((m_f1score*100)*100)/100
print("En promedio, la proporción entre exhaustividad y precisión del modelo es del "+str(per_m)+"%.")
print("========== GRÁFICOS: INFORMATION GAIN =========")

plt.subplots(1, figsize=(5,5))
plt.title('Precision Per Hold-out Iteration - Information Gain')
plt.scatter([i for i in range (10)], h_precision_infogain)
plt.ylabel("Precision")
plt.xlabel("Hold-out Iteracion")
plt.show()
print("----------------------------------")
lmin = int(min(h_precision_infogain)*10000)/10000
lmax = int(max(h_precision_infogain)*10000)/10000
per_lmin = lmin*100
per_lmax = lmax*100
print("Del gráfico se puede observar que la capacidad del modelo para retornar solamente instancias relevantes se encuentra entre el "+str(per_lmin)+"% y "+str(per_lmax)+"%")
print("Es decir, existe entre "+str(int((100-per_lmax)*100)/100)+"% y "+str(int((100-per_lmin)*100)/100)+"% de probabilidad de que el modelo entregue un diagnóstico erróneo")
print("----------------------------------")
plt.subplots(1, figsize=(5,5))
plt.title('Recall Per Hold-out Iteration - Information Gain')
plt.scatter([i for i in range (10)], h_recall_infogain)
plt.ylabel("Recall")
plt.xlabel("Hold-out Iteracion")
plt.show()
print("----------------------------------")
m_recall = int(mean(h_recall_infogain)*10000)/10000
per_m = int((m_recall*100)*100)/100
print("En promedio, el modelo solamente es capaz de diagnosticar correctamente un tumor en el "+str(per_m)+"% de los casos")
print("----------------------------------")
plt.subplots(1, figsize=(5,5))
plt.title('F1 Score Per Hold-out Iteration - Information Gain')
plt.scatter([i for i in range (10)], h_f1score_infogain)
plt.ylabel("F1 Score")
plt.xlabel("Hold-out Iteracion")
plt.show()
print("----------------------------------")
m_f1score = int(mean(h_f1score_infogain)*10000)/10000
per_m = int((m_f1score*100)*100)/100
print("En promedio, la proporción entre exhaustividad y precisión del modelo es del "+str(per_m)+"%.")

