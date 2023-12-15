import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class MNISTDataLoader:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()

        self.train_images = self.process_images(self.train_images)
        self.test_images = self.process_images(self.test_images)

        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def process_images(self, images):
        images = images.reshape((images.shape[0], 28, 28, 1)).astype('float32') / 255
        return images

class CNNModel:
    def __init__(self):
        self.model = models.Sequential()
        self.build_model()

    def build_model(self):
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

class Trainer:
    def __init__(self, model, data_loader):
        self.model = model.model
        self.train_images = data_loader.train_images
        self.train_labels = data_loader.train_labels

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, epochs=5, batch_size=64, validation_split=0.2):
        print("Starting training...")
        history = self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        print("Training completed.")
        return history

class Evaluator:
    def __init__(self, model, data_loader):
        self.model = model.model
        self.test_images = data_loader.test_images
        self.test_labels = data_loader.test_labels

    def evaluate_model(self):
        print("\nEvaluating the model on the test set...")
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f'Test set accuracy: {test_acc}')

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

class Main:
    def __init__(self):
        self.data_loader = MNISTDataLoader()
        self.model = CNNModel()
        self.trainer = Trainer(self.model, self.data_loader)
        self.evaluator = Evaluator(self.model, self.data_loader)

    def run(self):
        self.trainer.compile_model()
        history = self.trainer.train_model(epochs=10)
        self.evaluator.evaluate_model()
        self.evaluator.plot_accuracy(history)

if __name__ == "__main__":
    main = Main()
    main.run()