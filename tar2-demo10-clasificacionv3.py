import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB # Modelos de Probabilidad y Bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
import os

# Suprimir advertencias
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ClasificadorEstadistico:
    """
    Clase para manejar la carga de datos, el preprocesamiento,
    el entrenamiento y la evaluación de modelos de clasificación multiclase.
    """
    def __init__(self, filepath=''):
        self.scaler = StandardScaler()
        self.labels = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.metric_results = {}
        self.filepath = filepath
        self.drop_first = True
        self._df = None
        self._random_state = 42
        
    def preproceso(self, target_col='', columns=[], sep=',', test_size=0.2, random_state=42):
        """Preprocesamiento del dataset y división en conjuntos de entrenamiento y prueba."""
        if not self.filepath:
            raise ValueError("No se estableció archivo a procesar.")
        self._random_state = random_state
        self._load_and_preprocess(self.filepath, target_col, columns, test_size, sep)
    
    def val_max_K(self):
        """Retorna el valor máximo de K (raíz cuadrada del número total de filas)."""
        if self._df is None:
                 raise ValueError("No se cargó el dataset.")    
        else:
            return int(np.sqrt(self._df.shape[0]))  
        
    def _load_and_preprocess(self, filepath, target_col, columns, test_size, sep):
        """Carga un dataset desde un archivo CSV y preprocesa."""
        
        print("1. Cargando y Preprocesando Datos...")
        if not target_col:
            raise ValueError("El target no tiene la columna a evaluar.")
    
        self._df = pd.read_csv(filepath, sep=sep)
        
        # One-Hot Encoding (Convierte 'color' a binario)
        df_encoded = self._one_hot_encode_column(columns, drop_first=True)
      
        # Separación de características y target
        X = df_encoded.drop(target_col, axis=1).values
        y = df_encoded[target_col].values.ravel()
        
        # División de datos
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_state 
        )
        
        # Escalado (fit solo en train, transform en train y test)
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        self.labels = np.unique(self.y_train)
        print(f"   Datos listos. Clases de calidad: {self.labels}")
        print("-" * 50)


    def _calculate_metrics(self, y_true, y_pred):
        """Calcula Precision, Recall, F1-Score y Specificity (Promedio One-vs-Rest)."""
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=self.labels)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=self.labels)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=self.labels)
        
        # Cálculo de Specificity (Media)
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        specificity_list = []
        
        for i in range(len(self.labels)):
            # Total de verdaderos negativos (TN) para la clase i
            TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            # Total de falsos positivos (FP) para la clase i
            FP = cm[:, i].sum() - cm[i, i]
            
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            
            if not np.isnan(spec): specificity_list.append(spec)
            
        overall_specificity = np.mean(specificity_list) if specificity_list else 0.0

        return {
            'Precision (Weighted)': precision,
            'Recall (Weighted)': recall,
            'F1-Score (Weighted)': f1,
            'Specificity (Mean)': overall_specificity
        }


    def evaluar_modelo(self, model_class, params=None, is_scaled=False, model_name="Modelo", fixed_params=None):
        """
            Entrena y evalúa un modelo para un conjunto de hiperparámetros.
            Almacena el resultado para el modelo con el mejor F1-Score.
        """
        
        # Seleccionar datos de entrenamiento/prueba (escalados o sin escalar)
        is_tree_based = "RandomForest" in model_name or "DecisionTree" in model_name or "KNeighbors" in model_name
        
        X_train = self.X_train_scaled if is_scaled else self.X_train_scaled 
        X_test = self.X_test_scaled if is_scaled else self.X_test_scaled

        # Invertir el escalado para modelos que no lo requieren (RF, KNN, DT, Naive Bayes)
        if is_tree_based or "Naive Bayes" in model_name:
            X_train = self.scaler.inverse_transform(self.X_train_scaled) 
            X_test = self.scaler.inverse_transform(self.X_test_scaled) 
        
        # Manejo de modelos sin hiperparámetros (Naive Bayes)
        if params is None:
            model_instance = model_class(**(fixed_params or {}))
            model_instance.fit(X_train, self.y_train)
            y_pred = model_instance.predict(X_test)
            metrics = self._calculate_metrics(self.y_test, y_pred)
            self.metric_results[model_name] = {'Best Parameter': 'N/A', **metrics}
            print(f"2. Evaluando {model_name} (Parámetro Fijo)... F1: {metrics['F1-Score (Weighted)']:.4f}")
            return
        
        
        metric_data = {}
        param_key = list(params.keys())[0] 
        param_values = params[param_key]
        
        # Combina parámetros fijos con los variables
        combined_fixed_params = fixed_params or {}
        
        print(f"2. Evaluando {model_name} ({param_key} range: {param_values[0]:.4f} to {param_values[-1]:.4f})...")

        for p_value in param_values:
            init_params = {param_key: p_value}
            init_params.update(combined_fixed_params) # Agregar parámetros fijos

            # Agregar random_state solo si es necesario y si no está ya en fixed_params
            if ("Logistic" in model_name or "Support Vector" in model_name or "Random Forest" in model_name or "AdaBoost" in model_name) and 'random_state' not in init_params:
                init_params['random_state'] = self._random_state
            
            # Crear la instancia del modelo 
            model_instance = model_class(**init_params)
            
            # Entrenamiento y Predicción
            model_instance.fit(X_train, self.y_train)
            y_pred = model_instance.predict(X_test)
            
            # Cálculo y almacenamiento
            metrics = self._calculate_metrics(self.y_test, y_pred)
            metric_data[p_value] = metrics
            
        # Encontrar el mejor resultado basado en F1-Score
        best_param = max(metric_data, key=lambda p: metric_data[p]['F1-Score (Weighted)'])
        best_metrics = metric_data[best_param]
        
        # Almacenar el mejor resultado final
        self.metric_results[model_name] = {
            'Best Parameter': f"{param_key} = {best_param:.4f}",
            **best_metrics 
        }

    
    def display_best_results(self, metrica_objetivo='F1-Score (Weighted)'):
        """Muestra el resumen comparativo y el ganador general."""
        
        if not self.metric_results:
            print("No hay resultados para mostrar. Ejecute 'evaluar_modelo' primero.")
            return

        # Convertir a DataFrame para fácil visualización
        df_results = pd.DataFrame.from_dict(self.metric_results, orient='index')
        df_results = df_results.sort_values(by=metrica_objetivo, ascending=False)
        
        # Encontrar el ganador general
        ganador = df_results.iloc[0]
        
        print("\n" + "=" * 60)
        print("          RENDIMIENTO ÓPTIMO POR MODELO (Según F1-Score)")
        print("=" * 60)
        print(df_results.to_string(float_format="%.4f"))

        print("\n" + "=" * 60)
        print("             MEJOR ALTERNATIVA GENERAL")
        print("=" * 60)
        print(f"El mejor modelo (basado en {metrica_objetivo}) es: {df_results.index[0]}")
        print(f"Hiperparámetro: {ganador['Best Parameter']}")
        print(f"F1-Score: {ganador[metrica_objetivo]:.4f}")

    
    def _one_hot_encode_column(self, columns, drop_first=True):
        """
        Aplica pd.get_dummies() para One-Hot Encoding en las columnas especificadas.
        """
        df_encoded = self._df.copy() # Trabajar en una copia

        for column_name in columns: 
            if column_name in df_encoded.columns:
                df_encoded = pd.get_dummies(
                    df_encoded, 
                    columns=[column_name], 
                    prefix=column_name, 
                    drop_first=drop_first
                )
            else:
                 raise ValueError(f"Error: La columna '{column_name}' no existe en el DataFrame.")
        
        return df_encoded
    
    
# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Definición de Parámetros de Entrada
    NOMBRE_ARCHIVO = "wine_data_combined.csv"
    COLUMNA_OBJETIVO = 'quality' 
    COLUMNAS_CATEGORICAS = ['color']
    
    # 2. Inicializar y Preprocesar
    analizador = ClasificadorEstadistico(filepath=NOMBRE_ARCHIVO)
    analizador.preproceso(target_col=COLUMNA_OBJETIVO, sep=',', columns=COLUMNAS_CATEGORICAS)
    
    # 3. Definición de Rangos de Hiperparámetros
    c_range = np.logspace(-2, 1, 10) 
    k_range = range(1, analizador.val_max_K(), 3)
    depth_range = range(1, 16, 2) 

    # 4. Evaluación de modelos
    print("\n3. Resultados Finales:")
    
    # --- Modelos Lineales / Regresionales (Requieren Escalamiento) ---
    analizador.evaluar_modelo(LogisticRegression, {'C': c_range}, is_scaled=True, model_name="Logistic Regression")
    
    # --- SVM (Evaluando Kernels RBF y LINEAL) ---
    analizador.evaluar_modelo(SVC, {'C': c_range}, is_scaled=True, model_name="Support Vector Machine (RBF)", fixed_params={'kernel': 'rbf'})
    analizador.evaluar_modelo(SVC, {'C': c_range}, is_scaled=True, model_name="Support Vector Machine (Linear)", fixed_params={'kernel': 'linear'})
    
    # --- Modelos de Vecinos ---
    analizador.evaluar_modelo(KNeighborsClassifier, {'n_neighbors': k_range}, is_scaled=False, model_name="K-Nearest Neighbors")
    
    # --- Modelos de Árboles y Conjuntos ---
    analizador.evaluar_modelo(DecisionTreeClassifier, {'max_depth': depth_range}, is_scaled=False, model_name="Decision Tree")
    analizador.evaluar_modelo(RandomForestClassifier, {'max_depth': depth_range}, is_scaled=False, model_name="Random Forest")
    analizador.evaluar_modelo(GradientBoostingClassifier, {'n_estimators': range(50, 201, 50)}, is_scaled=False, model_name="Gradient Boosting")
    analizador.evaluar_modelo(AdaBoostClassifier, {'n_estimators': range(50, 201, 50)}, is_scaled=False, model_name="AdaBoost")
    
    # --- Modelos de Probabilidad y Bayes (No tienen hiperparámetros de ajuste aquí) ---
    # Se utiliza un valor de ajuste (var_smoothing) para que pase por el bucle de evaluación
    analizador.evaluar_modelo(GaussianNB, {'var_smoothing': np.logspace(-9, -7, 10)}, is_scaled=False, model_name="Gaussian Naive Bayes")


    # 5. Mostrar los resultados
    analizador.display_best_results()