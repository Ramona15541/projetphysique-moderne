
from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# 1.1 - Question 2.a : Définition de la fonction d'onde plane
# ─────────────────────────────────────────────────────────────────
def PlaneWave(amp,k, omega, x, t):
    """
    Calcule l'expression d'une onde plane à une dimension d'espace.
    
    Formule : Psi(x, t) = amp * exp(i * (k*x - omega*t))
    """
    # Utilisation de 1j pour représenter le nombre complexe i en Python
    return amp *exp(1j *(k * x - omega * t))

# ─────────────────────────────────────────────────────────────────
# 1.1 - Question 2.b : Test et tracé graphique simple
# ─────────────────────────────────────────────────────────────────

# Paramètres arbitraires pour le test visuel
amplitude1 =1
nombre_onde_k= 2*pi      # Équivaut à une longueur d'onde lambda = 1 m

pulsation_w =4*pi        # Équivaut à une période T = 0.5 s
instant_t =0.25            # Instant d'observation

# Grille spatiale d'évaluation
grille_x = linspace(-3.0, 3.0, 5000)

# Évaluation de l'onde plane
onde_evaluee = PlaneWave(amplitude1, nombre_onde_k, pulsation_w, grille_x, instant_t)

# Structure d'affichage imposée par le document
fig,ax =plt.subplots()

# Tracé des composantes réelle et imaginaire
ax.plot(grille_x, real(onde_evaluee), label="Partie Réelle", color="violet", linestyle="-")

ax.plot(grille_x, imag(onde_evaluee), label="Partie Imaginaire", color="orange", linestyle="--")

# Habillage minimaliste du graphique
ax.set_title("Visualisation d'une onde plane 1D")

ax.set_xlabel("Position x (m)")
ax.set_ylabel("Amplitude de la fonction d'onde")
ax.grid(True)
ax.legend()


# ─────────────────────────────────────────────────────────────────
# 1.2 - Question 2 : Superposition de trois ondes planes
# ─────────────────────────────────────────────────────────────────

# Définition des paramètres de l'énoncé
amplitude_centrale = 1.0
k0 = 8 * pi
delta_k = 2 * pi
t_initial = 0.0

# Bornes de l'intervalle demandé : [-pi/delta_k, pi/delta_k]
x_superposition =linspace(-pi/ delta_k, pi /delta_k, 1000)

# Calcul individuel des trois ondes constituantes à t = 0
# (Pour simplifier ici sans la relation de dispersion complète, on fixe omega=0 à t=0)
onde1= PlaneWave(amplitude_centrale, k0, 0, x_superposition, t_initial)
onde2= PlaneWave(amplitude_centrale / 2, k0 - delta_k / 2, 0, x_superposition, t_initial)

onde3 = PlaneWave(amplitude_centrale / 2, k0 + delta_k / 2, 0, x_superposition, t_initial)

# Somme des ondes
onde_totale = onde1 + onde2 +onde3

# Enveloppe analytique issue du calcul théorique de la somme
# Psi_tot = A*exp(i*k0*x)* (1 + cos(delta_k *x/ 2))
enveloppe_superieure= amplitude_centrale * (1 + cos(delta_k * x_superposition/ 2))
enveloppe_inferieure = -enveloppe_superieure

# La tracé de la superposition
fig_sup, ax_sup= plt.subplots()

# L'ondes individuelles (parties réelles)
ax_sup.plot(x_superposition, real(onde1), label="Onde k0", color="red", alpha=0.75)
ax_sup.plot(x_superposition, real(onde2), label="Onde k0 - Dk/2", color="blue", alpha=0.75)

ax_sup.plot(x_superposition, real(onde3), label="Onde k0 + Dk/2", color="green", alpha=0.75)

# Onde résultante et son enveloppe
ax_sup.plot(x_superposition, real(onde_totale), label="Somme Réelle", color="purple", linewidth=1.5)
ax_sup.plot(x_superposition, enveloppe_superieure, label="Enveloppe", color="darkblue", linestyle="--")
ax_sup.plot(x_superposition, enveloppe_inferieure, color="darkblue", linestyle="--")

ax_sup.set_title("Superposition de 3 ondes planes avec enveloppe")
ax_sup.set_xlabel("Position x (m)")

ax_sup.set_ylabel("Re(Psi)")
ax_sup.grid(True)
ax_sup.legend()

# Affichage des fenêtres
plt.show()
