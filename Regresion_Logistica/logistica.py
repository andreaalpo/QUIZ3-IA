import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox

try:
    data = pd.read_csv("datos1.csv")

    # Verificar que el archivo tenga las columnas necesarias
    if 'Horas' in data.columns and 'Resultado' in data.columns:
        # Verificar el balance de clases en el dataset
        print("Distribución de clases en 'Resultado':")
        print(data['Resultado'].value_counts())

        # Dividir los datos en características (X) y la variable objetivo (y)
        X = data[['Horas']]
        y = data['Resultado']

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Entrenar el modelo de regresión logística
        modelo_logistico = LogisticRegression()
        modelo_logistico.fit(X_train, y_train)

        # Evaluación inicial (opcional, para verificar el modelo)
        y_pred = modelo_logistico.predict(X_test)
        print("Informe de Clasificación Inicial:")
        print(classification_report(y_test, y_pred, zero_division=1))  # Maneja clases sin predicciones

        # Crear la interfaz gráfica con Tkinter
        def predecir_aprobacion():
            try:
                # Obtener el valor ingresado por el usuario
                horas = float(entry_horas.get())
                
                # Crear un DataFrame con el valor de horas
                horas_df = pd.DataFrame([[horas]], columns=['Horas'])
                
                # Realizar la predicción y calcular la probabilidad
                probabilidad = modelo_logistico.predict_proba(horas_df)[0][1]  # Probabilidad de aprobar
                
                # Definir umbral (ajustado a 0.5 como predeterminado)
                umbral = 0.5
                
                # Mostrar el resultado en un cuadro de diálogo
                if probabilidad > umbral:
                    mensaje = f"Con {horas} horas de estudio, el estudiante probablemente aprobará (Probabilidad: {probabilidad:.2f})."
                else:
                    mensaje = f"Con {horas} horas de estudio, el estudiante probablemente no aprobará (Probabilidad: {probabilidad:.2f})."
                
                messagebox.showinfo("Resultado", mensaje)
            except ValueError:
                messagebox.showerror("Error", "Por favor ingresa un número válido para las horas de estudio.")

        # Configurar la ventana de la interfaz
        ventana = tk.Tk()
        ventana.title("Predicción de Aprobación")

        # Crear los elementos de la interfaz
        label_instruccion = tk.Label(ventana, text="Ingresa las horas de estudio:")
        label_instruccion.pack()

        entry_horas = tk.Entry(ventana)
        entry_horas.pack()

        boton_predecir = tk.Button(ventana, text="Predecir", command=predecir_aprobacion)
        boton_predecir.pack()

        # Iniciar el bucle de la interfaz gráfica
        ventana.mainloop()
    else:
        print("El archivo CSV no contiene las columnas 'Horas' y 'Resultado' necesarias para el análisis.")
except Exception as e:
    print("Ocurrió un error:", e)
