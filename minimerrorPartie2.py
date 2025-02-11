import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class Minimerror:
    def __init__(self, N, recuit=0.01):
        # Initialiser les poids et la température
        self.W = [-0.4, -1.1, 0.1]  # Poids initiaux
        self.recuit = recuit  # Paramètre de recuit
        self.N = N  # Nombre de caractéristiques
        self.beta_plus = 1.0  # Température initiale beta+
        self.beta_minus = 0.5  # Température initiale beta-

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

    def fit(self, X, Y, epochs, beta_ratio=2.0):
        # Apprentissage avec recuit simulé et deux températures
        X = np.c_[X, np.ones((X.shape[0]))]  # Ajout du biais
        epoch_costs = []  # Liste pour suivre les coûts par époque

        for epoch in range(epochs):
            total_cost = 0  # Coût total pour l'époque
            T_plus = self.beta_plus  # Température plus
            T_minus = self.beta_minus  # Température moins

            for x, y in zip(X, Y):
                # Calculer la stabilité
                stability = (y * np.dot(x, self.W)) / (2 * T_plus * np.sqrt(self.N))

                # Calcul du coût et mise à jour du coût total
                total_cost += self.V(stability)

                # Mise à jour des poids si le critère est respecté
                if stability < 2 * T_plus:
                    grad_E = -(y * x * self.tanh_derivative(stability)) / (2 * T_plus * np.sqrt(self.N))
                    self.W -= self.recuit * grad_E

            # Normalisation des poids pour respecter \|w\| = sqrt(N)
            self.W = (self.W / LA.norm(self.W)) * np.sqrt(self.N)

            # Refroidissement de la température avec un rapport beta_ratio
            T_plus *= beta_ratio
            T_minus *= beta_ratio

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

    def save_weights(self, filename):
        # Sauvegarder les poids dans un fichier
        np.savetxt(filename, self.W, delimiter=',')

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

def plot_stabilities(stabilities, beta_values):
    # Tracer le graphique des stabilités en fonction de beta
    plt.figure()
    plt.plot(beta_values, stabilities, marker='o', color='blue', alpha=0.7)
    plt.title("Stabilités des exemples de test en fonction de β")
    plt.xlabel("Valeur de β")
    plt.ylabel("Stabilité moyenne")
    plt.grid()
    plt.show()

def main():
    # Charger les données (exemple avec des données fictives)
    train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Données d'entraînement
    train_Y = np.array([-1, -1, -1, 1])  # Classes pour l'entraînement
    test_X = np.array([[0, 1], [1, 0]])  # Données de test sans classes
    test_Y = np.array([-1, -1])  # Classes pour le test, utilisées uniquement pour le calcul d'erreur

    # Créer un perceptron et l'entraîner
    perceptron = Minimerror(N=train_X.shape[1])
    perceptron.fit(train_X, train_Y, epochs=1000, beta_ratio=1.2)

    # Calculer les erreurs et les stabilités
    Ea, Eg, stabilities = calculate_errors_and_stabilities(perceptron, train_X, train_Y, test_X, test_Y)

    # Afficher les résultats
    print(f"Erreur d'apprentissage Ea: {Ea * 100:.2f}%")
    print(f"Erreur de généralisation Eg: {Eg * 100:.2f}%")
    print(f"Poids du perceptron (W): \n{perceptron.W}")

    # Sauvegarder les poids dans un fichier
    perceptron.save_weights('weights.csv')

    # Calculer et tracer les stabilités pour différentes valeurs de beta
    beta_values = np.arange(1, 11, 1)  # Valeurs de β = 1,2,...,10
    stability_for_betas = []

    # Pour chaque β, recalculer les stabilités
    for beta in beta_values:
        perceptron.beta_plus = beta  # Modifier β+
        perceptron.beta_minus = beta / 2  # Modifier β-
        _, _, stabilities_for_beta = calculate_errors_and_stabilities(perceptron, train_X, train_Y, test_X, test_Y)
        stability_for_betas.append(np.mean(stabilities_for_beta))  # Moyenne des stabilités

    # Tracer les stabilités en fonction de beta
    plot_stabilities(stability_for_betas, beta_values)

if __name__ == "__main__":
    main()
