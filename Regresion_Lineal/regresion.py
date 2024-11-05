import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

# Leer datos desde un archivo CSV
data = pd.read_csv("datos2.csv")

# Asegurarse de que el archivo tenga las columnas necesarias
if 'Tamaño' in data.columns and 'Habitaciones' in data.columns and 'Baños' in data.columns and 'Precio' in data.columns:
    # Dividir en conjunto de entrenamiento y prueba
    X = data[['Tamaño', 'Habitaciones', 'Baños']]  # Características
    y = data['Precio']  # Variable objetivo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Crear y entrenar el modelo
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)

    # Predicción y evaluación
    y_pred = modelo_lineal.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (MSE):", mse)

    # Función para hacer predicciones con la interfaz gráfica
    def predecir_precio():
        try:
            tamaño = float(entry_tamaño.get())
            habitaciones = int(entry_habitaciones.get())
            baños = int(entry_baños.get())
            # Crear un DataFrame para la entrada
            entrada = pd.DataFrame([[tamaño, habitaciones, baños]], columns=['Tamaño', 'Habitaciones', 'Baños'])
            precio_estimado = modelo_lineal.predict(entrada)[0]
            mensaje = f"Para una casa de {tamaño:.2f} m², {habitaciones} habitaciones y {baños} baños, el precio estimado es: ${precio_estimado:.2f} mil dólares"
            messagebox.showinfo("Precio Estimado", mensaje)
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa números válidos para el tamaño de la casa, el número de habitaciones y el número de baños.")

    # Configurar la interfaz gráfica
    ventana = tk.Tk()
    ventana.title("Predicción de Precio de Casa")

    label_instruccion_tamaño = tk.Label(ventana, text="Ingresa el tamaño de la casa en m²:")
    label_instruccion_tamaño.pack()

    entry_tamaño = tk.Entry(ventana)
    entry_tamaño.pack()

    label_instruccion_habitaciones = tk.Label(ventana, text="Ingresa el número de habitaciones:")
    label_instruccion_habitaciones.pack()

    entry_habitaciones = tk.Entry(ventana)
    entry_habitaciones.pack()

    label_instruccion_baños = tk.Label(ventana, text="Ingresa el número de baños:")
    label_instruccion_baños.pack()

    entry_baños = tk.Entry(ventana)
    entry_baños.pack()

    boton_predecir = tk.Button(ventana, text="Predecir Precio", command=predecir_precio)
    boton_predecir.pack()

    # Iniciar el bucle de la interfaz gráfica
    ventana.mainloop()

    # Visualización (opcional)
    # plt.scatter(X_test['Tamaño'], y_test, color='blue', label='Datos reales')
    # plt.scatter(X_test['Tamaño'], y_pred, color='red', label='Predicción')
    # plt.xlabel('Tamaño de la casa (m²)')
    # plt.ylabel('Precio de la casa (miles de $)')
    # plt.title('Regresión Lineal')
    # plt.legend()
    # plt.show()
else:
    print("El archivo CSV no contiene las columnas necesarias para el análisis.")
