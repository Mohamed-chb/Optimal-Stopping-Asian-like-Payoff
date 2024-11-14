#!/usr/bin/env python
# coding: utf-8

# In[12]:


#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import optuna
import matplotlib.pyplot as plt

# Paramètres de simulation globaux
S0 = 10       # Prix initial
sigma = 0.2   # Volatilité
delta_t = 1 / 252  # Incrément de temps
N = 22        # Nombre de jours

class Simulator:
    def __init__(self, S0, sigma, delta_t, N):
        self.S0 = S0
        self.sigma = sigma
        self.delta_t = delta_t
        self.N = N

    def simulate_paths(self, M):
        """Simule M trajectoires des prix S et calcule la moyenne cumulative A"""
        # Matrice de bruit gaussien pour M trajectoires sur N jours
        increments = np.random.normal(0, 1, (M, self.N))
        
        # Initialisation des trajectoires de S et assignation du prix initial
        S = np.zeros((M, self.N + 1))
        S[:, 0] = self.S0
        
        # Calcul vectorisé de chaque trajectoire S
        S[:, 1:] = S[:, [0]] * np.exp(
            np.cumsum((-0.5 * self.sigma**2 * self.delta_t) + self.sigma * np.sqrt(self.delta_t) * increments, axis=1)
        )

        # Calcul vectorisé de la moyenne cumulative A
        A = np.cumsum(S, axis=1) / np.arange(1, self.N + 2)

        return S, A

    def plot_trajectories(self, M=5):
        """Trace quelques trajectoires de l'actif S et des moyennes A sans boucle."""
        # Simule M trajectoires
        S, A = self.simulate_paths(M)
        time = np.arange(self.N + 1)

        plt.figure(figsize=(12, 6))
    
        # Tracer toutes les trajectoires S et A en une seule opération
        plt.plot(time, S.T, label=[f'Trajectoire S {i+1}' for i in range(M)], alpha=0.7)
        plt.plot(time, A.T, linestyle='--', label=[f'Moyenne A {i+1}' for i in range(M)], alpha=0.7)
    
        plt.title("Trajectoires simulées de l'actif et des moyennes associées")
        plt.xlabel("Temps (jours)")
        plt.ylabel("Prix de l'actif / Moyenne cumulative")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Légende déportée pour plus de lisibilité
        plt.show()

    def calculate_payoff(self, M):
        """
        Calcule le prix du payoff E(A_N / S_N) en utilisant la méthode de Monte Carlo,
        sans utiliser de boucle explicite.
        """
        # Génère M trajectoires simultanément pour S et A, de taille (M, N+1)
        S, A = self.simulate_paths(M)

        # Sélectionne les valeurs finales des trajectoires (A_N et S_N)
        A_N = A[:, -1]  # Dernière valeur pour chaque trajectoire dans A
        S_N = S[:, -1]  # Dernière valeur pour chaque trajectoire dans S

        # Calcule le ratio A_N / S_N pour chaque trajectoire, puis prend la moyenne
        payoff = np.mean(A_N / S_N)
        return payoff


class Strategy:
    def __init__(self, simulator):
        self.simulator = simulator

    def monte_carlo_payoff(self, a, M=1000):
        """Stratégie de seuil : arrêt lorsque A >= a * S."""
        # Génère les trajectoires de S et A pour M simulations
        S, A = self.simulator.simulate_paths(M)
        
        # Calcul du payoff final AN / SN
        payoff_final = A[:, -1] / S[:, -1]
        
        # Critère de temps d'arrêt : A >= a * S
        stopping_criteria = (A >= a * S)
        stopping_times = np.argmax(stopping_criteria, axis=1)
        
        # Vérifie les arrêts valides et définit les arrêts finaux si non rencontrés
        valid_stops = stopping_criteria[np.arange(M), stopping_times]
        stopping_times[~valid_stops] = self.simulator.N

        # Récupère A et S aux temps d'arrêt
        A_tau = A[np.arange(M), stopping_times]
        S_tau = S[np.arange(M), stopping_times]
        
        # Calcule le ratio moyen et retourne l'espérance et le payoff final
        return np.mean(A_tau / S_tau), np.mean(payoff_final)

    def snell_envelope_strategy(self, M=1000):
        """Stratégie basée sur l'enveloppe de Snell pour un arrêt optimal."""
        # Génère les trajectoires de S et A pour M simulations
        S, A = self.simulator.simulate_paths(M)
        
        # Calcul du payoff Z = A / S à chaque temps
        Z = A / S     # Matrice contenant les payoffs instantanée pour chaque jour 
        
        # Calcul de l'enveloppe de Snell en récurrence inversée
        snell_values = np.zeros_like(Z) # snell_values est une matrice avec les mêmes dimensions que la matrice Z
        snell_values[:, -1] = Z[:, -1]
        
        for t in range(self.simulator.N - 1, -1, -1): # On commence à N-1, on termine à 0
            snell_values[:, t] = np.maximum(Z[:, t], snell_values[:, t + 1])
        # La matrice snell_values est remplit au fur et à mesure c'est l'enveloppe de Snell
        # snell_values est la matrice de l'enveloppe de Snell, qui contient la valeur optimale (le payoff maximal possible)
        # que l’on peut espérer pour chaque jour et chaque simulation.

        # Temps d'arrêt optimal basé sur l'enveloppe de Snell
        stopping_times = np.argmax(Z >= snell_values, axis=1)
        # np.argmax renvoie l'indice du 1er élément True
        
        # Récupère A et S aux temps d'arrêt pour chaque simulation
        A_tau = A[np.arange(M), stopping_times]
        S_tau = S[np.arange(M), stopping_times]
        
        return np.mean(A_tau / S_tau)

# Fonctions d'optimisation pour optuna

def objective_threshold(trial):
    """Objectif pour optimiser la stratégie de seuil avec le paramètre a."""
    a = trial.suggest_uniform('a', 1, 3)
    strategy = Strategy(Simulator(S0, sigma, delta_t, N))
    payoff, _ = strategy.monte_carlo_payoff(a)
    return -payoff  # Maximisation en minimisant l'opposé

# Exécution de l'optimisation avec optuna pour la stratégie de seuil
study_threshold = optuna.create_study(direction="minimize")
study_threshold.optimize(objective_threshold, n_trials=100)

# Résultats de la stratégie de seuil
best_a = study_threshold.best_params['a']
simulator = Simulator(S0, sigma, delta_t, N)
strategy = Strategy(simulator)
payoff_threshold, payoff_final = strategy.monte_carlo_payoff(best_a, M=1000)
payoff_estime = simulator.calculate_payoff(M=1000)
print(f"Prix estimé du payoff E(A_N / S_N) : {payoff_estime}")
print(f"Meilleur paramètre a pour la stratégie de seuil : {best_a}")
print(f"Espérance maximale estimée avec seuil : {payoff_threshold:.4f}")
print(f"Prix estimé du payoff A_N / S_N : {payoff_final:.4f}")

# Résultats de la stratégie avec l'enveloppe de Snell
simulator = Simulator(S0, sigma, delta_t, N)
strategy = Strategy(simulator)

payoff_s = strategy.snell_envelope_strategy(M=1000)
print(f"Espérance maximale estimée avec l'enveloppe de Snell : {payoff_s:.4f}")

# Tracé des trajectoires simulées
simulator.plot_trajectories(M=5)


# In[ ]:





# In[ ]:






# In[ ]:




