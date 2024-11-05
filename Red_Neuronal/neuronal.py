import numpy as np
import tkinter as tk
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class MNISTApp:
    def __init__(self, master):
        self.master = master
        master.title("Clasificación de Dígitos Manuscritos")

        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = np.zeros((280, 280), dtype=np.float32)  # Inicializar la imagen

        self.predict_button = tk.Button(master, text="Predecir Dígito", command=self.make_prediction)
        self.predict_button.pack(pady=20)

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.load_data()
        self.train_model()

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255  # Normalizar los datos
        self.x_test = self.x_test.astype('float32') / 255

        # Aplanar las imágenes
        self.x_train = self.x_train.reshape(-1, 28 * 28)
        self.x_test = self.x_test.reshape(-1, 28 * 28)

    def train_model(self):
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')  # 10 clases para los dígitos 0-9
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32, verbose=2)

    def paint(self, event):
        # Captura el clic del mouse y pinta un círculo
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='black', outline='black')  # Aumentar el tamaño del pincel

        # Marcar el área de la imagen que se ha pintado
        if 0 <= x < 280 and 0 <= y < 280:
            self.image[y-10:y+10, x-10:x+10] = 1  # Normaliza a escala de 0-1

    def make_prediction(self):
        # Redimensionar la imagen a 28x28 píxeles
        img = self.image[::10, ::10]  # Muestreo para reducir a 28x28
        img = img.reshape(1, 28 * 28)  # Aplanar la imagen
        img = img.astype('float32')  # Asegurarse de que sea float
        img /= 1.0  # Normalizar la imagen

        # Hacer la predicción
        prediction = self.model.predict(img)
        predicted_digit = np.argmax(prediction)
        self.result_label.config(text=f"Dígito Predicho: {predicted_digit}")
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)  # Reiniciar la imagen

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTApp(root)
    root.mainloop()
