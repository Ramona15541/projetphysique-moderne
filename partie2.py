import numpy as np
import matplotlib.pyplot as plt

# a. Constantes physiques (hbar et masse électron en unités SI)
hbar = 1.054e-34

m = 9.109e-31

# b. Définition du paquet d'ondes gaussien

def paquet(k, etalement, x, t):
    """
    Calcule la fonction d'onde d'un paquet de vagues gaussien à l'instant t.
    """
    # Calcul de la dispersion temporelle alpha(t)
    alpha= etalement+ 1j * (hbar *t) / (2* m)

    
    # Coefficient pour la normalisation de la fonction d'onde
    norme =(2 * etalement / np.pi)**0.25/ np.sqrt(2 *alpha)
    
    # Calcul de la vitesse de groupe de la particule
    vg= (hbar* k) / m
    
    # Calcul des deux termes exponentiels (la cloche de Gauss et la phase oscillante)
    
    enveloppe = np.exp(- (x - vg * t)**2 / (4 * alpha))
    phase = np.exp(1j * (k * x - (hbar * k**2 / (2 * m)) * t))
               
    return norme * enveloppe * phase

# c. Test de la fonction d'onde à l'instant initial t = 0
# Définition de la grille spatiale à l'échelle du nanomètre

supportx = np.linspace(-6e-9, 6e-9, 2000) 
k_init = 6e9
ecart = 1e-18

# Évaluation du signal quantique à t = 0
onde0 = paquet(k_init, ecart, supportx, 0)


# Configuration et tracé du graphique
plt.figure(figsize=(10, 6))

# Affichage des deux composantes complexes (réelle et imaginaire)
plt.plot(supportx, np.real(onde0), label="Partie réelle", color="red")
plt.plot(supportx, np.imag(onde0), label="Partie imaginaire", color="green", linestyle="--")

# Réglages des légendes et des titres
plt.title("Paquet d'ondes gaussien à t=0")

plt.xlabel("Position x (m)")
plt.ylabel(r"$\Psi(x, 0)$")

plt.legend()
plt.grid()
plt.show()
