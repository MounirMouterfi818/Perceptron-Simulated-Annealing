import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class Minimerror:
    def __init__(self, N, recuit=0.01):
        # Initialiser les poids et la température
        self.W = [-0.4, -1.1, 0.1] #poids initiaux
        self.recuit = recuit  # Paramètre de recuit
        self.N = N  # Nombre de caractéristiques
        

    def heaviside(self, x):
        # Fonction d'activation de Heaviside
        return 1 if x > 0 else -1

    def tanh_derivative(self, x):
        # Dérivée de la fonction tanh utilisée dans le gradient de la fonction de coût
        return 1 - np.tanh(x) ** 2

    def V(self, x):
        # Fonction V(x) = 1 - tanh(x)
        return 1 - np.tanh(x)

    def predict(self, X, addBias=True):
        # Prédire la sortie en fonction des entrées
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]  # Ajout du biais
        return self.heaviside(np.dot(X, self.W))

    def fit(self, X, Y, epochs, T_initial=1.0, beta_decay=0.99):
        # Apprentissage avec recuit simulé
        X = np.c_[X, np.ones((X.shape[0]))]  # Ajout du biais
        T = T_initial  # Initialisation de la température
        epoch_costs = []  # Liste pour suivre les coûts par époque

        for epoch in range(epochs):
            total_cost = 0  # Coût total pour l'époque
            for x, y in zip(X, Y):
                # Calculer la stabilité
                stability = (y * np.dot(x, self.W)) / (2 * T * np.sqrt(self.N))

                # Calcul du coût et mise à jour du coût total
                total_cost += self.V(stability)

                # Mise à jour des poids si le critère est respecté
                if stability < 2 * T:
                    grad_E = -(y * x * self.tanh_derivative(stability)) / (2 * T * np.sqrt(self.N))
                    self.W -= self.recuit * grad_E

            # Normalisation des poids pour respecter \|w\| = sqrt(N)
            self.W = (self.W / LA.norm(self.W)) * np.sqrt(self.N)

            # Refroidissement de la température
            T *= beta_decay

            # Enregistrer le coût de l'époque
            epoch_costs.append(total_cost)

            # Arrêter si les coûts augmentent
            if epoch > 0 and epoch_costs[-1] > epoch_costs[-2]:
                print("Arrêt de l'entraînement : les coûts commencent à augmenter.")
                break

        # Tracer le graphique des coûts
        plt.plot(epoch_costs, marker='o')
        plt.title("Évolution des coûts par époque")
        plt.xlabel("Époques")
        plt.ylabel("Coût total")
        plt.grid()
        plt.show()

def calculate_errors_and_stabilities(perceptron, train_X, train_Y, test_X, test_Y):
    # Calcul de l'erreur d'apprentissage Ea
    Ea = 0
    for x, target in zip(train_X, train_Y):
        pred = perceptron.predict(x)
        if pred != target:
            Ea += 1

    # Calcul de l'erreur de généralisation Eg
    Eg = 0
    stabilities = []
    for x, target in zip(test_X, test_Y):
        X_bias = np.append(x, 1)  # Ajouter le biais
        stability = target * np.dot(X_bias, perceptron.W) / LA.norm(perceptron.W)
        stabilities.append(stability)

        pred = perceptron.predict(x)
        if pred != target:
            Eg += 1

    return Ea / len(train_Y), Eg / len(test_Y), stabilities

def plot_stabilities(stabilities):
    # Tracer le graphique des stabilités
    plt.figure()
    plt.bar(range(len(stabilities)), stabilities, color='blue', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Stabilités des exemples de test")
    plt.xlabel("Index des exemples")
    plt.ylabel("Stabilité")
    plt.grid()
    plt.show()

def main():
    # Charger les données (exemple avec des données fictives)
    train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Données d'entraîment
    train_Y = np.array([-1, -1, -1, 1])  # Classes pour l'entraîment
    test_X = np.array([[0, 1], [1, 0]])  # Données de test
    test_Y = np.array([-1, -1])  # Classes pour le test

    # Créer un perceptron et l'entraîner
    perceptron = Minimerror(N=train_X.shape[1])
    perceptron.fit(train_X, train_Y, epochs=1000, T_initial=1.0, beta_decay=0.99)

    # Calculer les erreurs et les stabilités
    Ea, Eg, stabilities = calculate_errors_and_stabilities(perceptron, train_X, train_Y, test_X, test_Y)

    # Afficher les résultats
    print(f"Erreur d'apprentissage Ea: {Ea * 100:.2f}%")
    print(f"Erreur de généralisation Eg: {Eg * 100:.2f}%")
    print(f"Poids du perceptron (W): \n{perceptron.W}")

    # Tracer le graphique des stabilités
    plot_stabilities(stabilities)

if __name__ == "__main__":
    main()
