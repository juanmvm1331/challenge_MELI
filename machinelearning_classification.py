import numpy as np

#%%
def normalizar(x,maxi,mini):
    """Permite realizar una normalizacion de mis datos en un rango diferente.
    
    Args:
        x: array de numpy con la data
        maxi: array de numpy con los maximos de cada caracteristica o con los maximos que el usuario desee normalizar
        mini: array de numpy con los minimos de cada caracteristica o con los maximos que el usuario desee normalizar
    Returns:
        xn: array de numpy con X normalizada
    """
    dim = np.shape(x)
    xn = np.zeros([dim[0],dim[1]])
    for i in range(dim[1]):
        xn[:,i]=(x[:,i]-mini[i])/(maxi[i]-mini[i])
    return (xn)

#%%
def entrenamiento(modelo,repeticiones,kfold,x,y,paralelismo):
    """ funcion para evaluar el desempeño con diferentes metricas de los modelos estandar
    
    Args:
        modelo: string del modelo a utilizar
        repeticiones: entero que permite determinar cuantas repeticiones quiero hacer en el entrenamiento
        kfold: estructura del k-fold que se desea usar
        x: array de numpy con la data
        y: array de numpy con las etiquetas
        paralelismo: entero que al ser -1 permite utilizar toda la capacidad del computador dado que comenzara a trabajar en paralelo
    Returns:
        diccionario: diccionario con el grid de los modelos entrenados.
    """
    from sklearn.model_selection import cross_val_score
    from time import time
    performance = np.zeros([repeticiones,5])
    total_time = []
    for i in range(repeticiones):
        start_time = time()
        try:
            performance[i,0] = np.mean(cross_val_score(modelo, x, y, cv=kfold,scoring='accuracy',n_jobs=paralelismo))
        except Exception:
            performance[i,0] = 99999

        try:
            performance[i,1] = np.mean(cross_val_score(modelo, x, y, cv=kfold,scoring='precision',n_jobs=paralelismo))
        except Exception:
            performance[i,1] = 99999

        try:
            performance[i,2] = np.mean(cross_val_score(modelo, x, y, cv=kfold,scoring='recall',n_jobs=paralelismo))
        except Exception:
            performance[i,2] = 99999

        try:
            performance[i,3] = np.mean(cross_val_score(modelo, x, y, cv=kfold,scoring='f1',n_jobs=paralelismo))
        except Exception:
            performance[i,3] = 99999

        try:
            performance[i,4] = np.mean(cross_val_score(modelo, x, y, cv=kfold,scoring='roc_auc',n_jobs=paralelismo))
        except Exception:
            performance[i,4] = 99999
        elapsed_time = time() - start_time

        total_time.append(elapsed_time)

    diccionario = {1:performance,2:total_time}
    return (diccionario)

#%%
def creacion_modelos(modelos,repeticiones,kfold,x,y,paralelismo):
    """ Funcion para crear los modelos con parametros por default.
    
    Args:
        modelos: lista de modelos a optimizar
        repeticiones: entero que permite determinar cuantas repeticiones quiero hacer en el entrenamiento
        kfold: estructura del k-fold que se desea usar
        x: array de numpy con la data
        y: array de numpy con las etiquetas
        paralelismo: entero que al ser -1 permite utilizar toda la capacidad del computador dado que comenzara a trabajar en paralelo
    Returns:
        diccionario: diccionario con el grid de los modelos entrenados.
    """
    import pandas as pd
    diccionario = {}
    for model in modelos:
        if model == 'NaiveBayes':
            from sklearn.naive_bayes import GaussianNB
            modelo = GaussianNB()

        if model == 'LinearDiscriminantAnalysis':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            modelo = LinearDiscriminantAnalysis()

        if model == 'SVM':
            from sklearn.svm import SVC
            modelo = SVC(gamma='auto')

        if model == 'DecisionTree':
            from sklearn import tree
            modelo = tree.DecisionTreeClassifier()

        if model == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            modelo = RandomForestClassifier()

        if model == 'AdaBoost':
            from sklearn.ensemble import AdaBoostClassifier
            modelo = AdaBoostClassifier()

        if model == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingClassifier
            modelo = GradientBoostingClassifier()

        if model == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            modelo = KNeighborsClassifier()

        if model == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            modelo = LogisticRegression()

        if model == 'MLP':
            from sklearn.neural_network import MLPClassifier
            modelo = MLPClassifier()

        if model == 'XGBoost':
            from xgboost import XGBClassifier
            modelo = XGBClassifier()

        performance = entrenamiento(modelo,repeticiones,kfold,x,y,paralelismo)
        df = pd.concat([pd.DataFrame(performance[1])*100, pd.DataFrame(performance[2])], axis=1)
        df.columns = ['Accuracy '+model,'Precision '+model,'Recall '+model,'F1 '+model,'Area bajo la curva ROC '+model,'tiempo (s)'+model]
        df[['Accuracy '+model,'Precision '+model,'Recall '+model,'F1 '+model,'Area bajo la curva ROC '
            +model,'tiempo (s)'+model]] = df[['Accuracy '+model,'Precision '+model,'Recall '+model,'F1 '
            +model,'Area bajo la curva ROC '+model,'tiempo (s)'+model]].applymap("{0:.2f}".format)
        diccionario[model] = df
    return(diccionario)


#%%
def optimizacion_modelos(modelos,parameters,kfold,paralelismo,visualizar_avance,metrica,x,y):
    """ Funcion para optimizar los modelos usando grid search
    
    Args:
        modelos: lista de modelos a optimizar
        parameters: diccionario que contiene un diccionario de las combinaciones que realizara cada modelo
        kfold: estructura del k-fold que se desea usar
        paralelismo: entero que al ser -1 permite utilizar toda la capacidad del computador dado que comenzara a trabajar en paralelo
        visualizar_avance: entero que permite anular o no la visualizacion del avance del grid search
        x: array de numpy con la data
        y: array de numpy con las etiquetas
    Returns:
        grid_final: diccionario con el grid de los modelos entrenados.
    """
    from sklearn.model_selection import GridSearchCV

    grid_final = {}
    for model in modelos:
        if model == 'NaiveBayes':
            from sklearn.naive_bayes import GaussianNB
            print()
            print('Comienza el gridsearch de '+model)
            modelo = GaussianNB()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'LinearDiscriminantAnalysis':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            print()
            print('Comienza el gridsearch de '+model)
            modelo = LinearDiscriminantAnalysis()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'SVM':
            from sklearn.svm import SVC
            print()
            print('Comienza el gridsearch de '+model)
            modelo = SVC()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'DecisionTree':
            from sklearn import tree
            print()
            print('Comienza el gridsearch de '+model)
            modelo = tree.DecisionTreeClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            print()
            print('Comienza el gridsearch de '+model)
            modelo = RandomForestClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'AdaBoost':
            from sklearn.ensemble import AdaBoostClassifier
            print()
            print('Comienza el gridsearch de '+model)
            modelo = AdaBoostClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingClassifier
            print()
            print('Comienza el gridsearch de '+model)
            modelo = GradientBoostingClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            print()
            print('Comienza el gridsearch de '+model)
            modelo = KNeighborsClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            print()
            print('Comienza el gridsearch de '+model)
            modelo = LogisticRegression()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'MLP':
            from sklearn.neural_network import MLPClassifier
            print()
            print('Comienza el gridsearch de '+model)
            modelo = MLPClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)

        if model == 'XGBoost':
            from xgboost import XGBClassifier
            print()
            print('Comienza el gridsearch de '+model)
            modelo = XGBClassifier()
            grid = GridSearchCV(modelo, parameters[model],scoring=metrica,n_jobs=paralelismo,verbose=visualizar_avance,cv=kfold)
            grid.fit(x, y)
        grid_actual = {model:grid}
        grid_final.update(grid_actual)
    return(grid_final)

#%%
def evidenciar_resultados(performance, model):
    """Permite imprimir el resultado de cada metrica depues de ser promediada.
    
    Args:
        performance: diccionario con valores de cada metrica.
        model: string con el nombre del modelo.
    """
    print("El F1 del modelo        {} es de      {} ".format(model,np.mean(performance['test_f1'])))
    print("El ROC_AUC del modelo   {} es de      {} ".format(model,np.mean(performance['test_roc_auc'])))
    print("El ACCURACY del modelo  {} es de      {} ".format(model,np.mean(performance['test_accuracy'])))
    print("El PRECISION del modelo {} es de      {} ".format(model,np.mean(performance['test_precision'])))
    print("El RECAL del modelo     {} es de      {} ".format(model,np.mean(performance['test_recall'])))
    print()
    print()