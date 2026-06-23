import numpy as np
import matplotlib.pyplot as plt

# Constantes du problème
hbar = 1.0
m = 1.0

k0 = 5.0  

a_wp = 1.0    
x0 = -10.0    
    

# ============================================================
# 1. Le paquet d'ondes initial (La fonction d'onde)
# ============================================================

def paquet(k, ecart, x, centre, t):
    """
    Crée la forme initiale gaussienne complexe de la particule.
    """
    alpha = ecart+ 1j *hbar * t/ (2* m)
    norme =(2 * ecart / np.pi) ** 0.25 /np.sqrt(2 * alpha)
    vg =hbar * k/m
    phase= np.exp(1j * (k * (x - centre)- hbar *k**2 / (2* m)* t))
    cloche = np.exp(-((x-centre - vg *t)** 2) / (4 * alpha))
    return norme* cloche *phase


# ============================================================
# 2. Construction du mur de potentiel
# ============================================================
def mur(x, debut, large,v0):
    """
    Génère un tableau pour le potentiel de la barrière rectangulaire.
    """
    v = np.zeros_like(x)
    v[(x >=debut) & (x <=debut+large)] =v0
    return v


# ============================================================
# 3. Méthode de Thomas (Solveur de matrice tridiagonale)
# ============================================================
def thomas(a, b, c, d):
    """
    Résolution rapide du système linéaire de Crank-Nicolson.
    """
    n = len(d)
    bc = b.copy()
    dc = d.copy()
    for i in range(1, n):
        f = a[i - 1] / bc[i - 1]

        bc[i]-= f * c[i - 1]
        dc[i] -= f * dc[i - 1]
    sol = np.zeros(n, dtype=complex)

    sol[-1]= dc[-1]/bc[-1]
    for i in range(n - 2, -1, -1):
        sol[i] = (dc[i] -c[i]* sol[i + 1])/ bc[i]
    return sol


# ============================================================
# 4. Résolution de Schrödinger (Crank-Nicolson)
# ============================================================
def schrodinger(nx, nt, xmin, xmax, tmin, tmax, k, ecart, centre, debut, large, v0):
    """
    Fait avancer le paquet d'ondes dans le temps pas à pas.
    """
    xmat = np.linspace(xmin, xmax, nx)

    tmat = np.linspace(tmin, tmax, nt)
    dx = xmat[1] - xmat[0]
    dt = tmat[1] - tmat[0]

    vtot = mur(xmat, debut, large, v0)

    vint = vtot[1:-1]

    psi = np.zeros((nt, nx), dtype=complex)
    psi[0, :] = paquet(k, ecart, xmat, centre, 0.0)

    nint = nx - 2
    r = 1j * hbar * dt / (4 * m * dx**2)
    s = 1j * vint * dt / (2 * hbar)

    amateur = np.full(nint - 1, -r, dtype=complex)

    bmateur = np.ones(nint, dtype=complex) * (1 + 2 * r) + s
    cmateur = np.full(nint - 1, -r, dtype=complex)

    for j in range(nt - 1):
        psiint = psi[j, 1:-1]
        rhs = (1 - 2 * r - s) *psiint + r* psi[j, :-2]+ r * psi[j, 2:]
        psi[j + 1,1:-1] =thomas(amateur, bmateur, cmateur, rhs)

        psi[j + 1,0] =0.0
        psi[j + 1, -1] = 0.0

    return xmat, tmat, psi, dx


# ============================================================
# 5. Calcul de la norme de probabilité
# ============================================================
def getnorme(onde, dx):
    """
    Fait la somme des carrés du module pour vérifier que la probabilité totale vaut 1.
    """
    return np.sum(np.abs(onde) ** 2) * dx


# ============================================================
# 6. Coefficient de transmission théorique T2
# ============================================================

def transmission(k, large, v0):
    """
    Formule analytique de la probabilité de passage (onde plane).
    """
    e= hbar**2 *k**2 / (2*m)
    if v0 <= 0.0:
        return 1.0
    if e >= v0:
        k2= np.sqrt(2 * m * (e - v0)) / hbar
        if k2== 0.0:
            return 1.0
        
        denom= 1 + (k**2- k2**2)**2 *np.sin(k2 *large) ** 2 / (4 *k**2 *k2**2)
        return 1.0 / denom
    else:
        kappa=np.sqrt(2 * m * (v0 - e)) / hbar
        denom =1 +(k**2 + kappa**2)**2 *np.sinh(kappa *large)**2 / (4* k**2 * kappa**2)

        return 1.0 / denom


# ============================================================
# 7. Formules pour les temps théoriques et Hartman
# ============================================================

def vitgroupe(k):

    """Vitesse de groupe de la particule."""
    return hbar * k / m


def tlibre(k, large):
    """Temps de vol sans obstacle."""
    return m* large/ (hbar * k)


def phasetrans(k, large, v0):
    """Phase après la traversée en régime tunnel."""
    ek = hbar**2 * k**2/ (2 * m)
    if ek >= v0:
        return None
    kappa = np.sqrt(2 * m *(v0- ek))/ hbar
    eta = (kappa**2 - k**2) / (2 * k * kappa)

    return -k * large - np.arctan(eta * np.tanh(kappa* large))


def retgroupe(k, large, v0):
    """Calcul de la dérivée de la phase pour le retard de groupe."""
    e = hbar**2 * k**2 / (2* m)
    if e >= v0:
        return None
    dk =1e-6
    phip =phasetrans(k +dk,large, v0)
    phim= phasetrans(k - dk, large, v0)
    if phip is None or phim is None:
        return None
    dphidk = (phip - phim) / (2 * dk)
    vg = hbar * k /m
    return dphidk /vg


def ttunnel(k, large, v0):
    """Temps tunnel théorique global (Hartman)."""
    e = hbar**2 *k**2 / (2* m)
    if e >= v0:
        return None
    tg = retgroupe(k, large, v0)
    if tg is None:
        return None
    return tlibre(k, large) + tg


# ============================================================
# 8. Chronomètre numérique
# ============================================================
def topchrono(psi, t, idxx):
    """
    Trouve l'instant précis où le pic de la vague passe par un point.
    """
    amp = np.array([np.abs(psi[j, idxx]) ** 2 for j in range(len(t))])
    return t[np.argmax(amp)]


# ============================================================
# 9. Script Principal
# ============================================================
def main():
    # Définition des grilles de simulation
    nx= 3000
    nt =6000

    xmin, xmax =-30.0, 30.0



    tmin, tmax =0.0,6.0

    # Variables de base du problème physique
    k0 =5.0       
    a_wp = 1.0     
    x0 = -10.0   




    x_b= 0.0      
    a_b = 1.0      
    V0 =15.0   


    # Calculs préparatoires

    E = hbar**2 * k0**2 / (2 * m)
    vg = vitgroupe(k0)
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar

    # Affichage d'origine nettoyé et aligné sur chaque ligne
    print("=" * 58)
    print("Paramètres physiques")

    print("=" * 58)
    print(f"  Énergie de la particule (E) : {E:.3f} (Régime : effet tunnel)")
    print(f"  Potentiel de la barrière (V0) : {V0:.3f}")
    print(f"  Vitesse de groupe (v_g)       : {vg:.3f}")
    print(f"  Coeff d'atténuation (κ)       : {kappa:.4f}")
    print(f"  Indice d'opacité (κ·a)         : {kappa * a_b:.3f} (> 1 : régime opaque)")
    

    # --------------------------------------------------------
    # Partie 4.1.b - Simulation sans barrière
    # --------------------------------------------------------

    print("\n--- 4.1.b : particule libre (V0 = 0) ---")
    x, t, psilibre, dx = schrodinger(nx, nt, xmin, xmax, tmin, tmax, k0, a_wp, x0, x_b, a_b, 0.0)

    idxentree = np.argmin(np.abs(x - x_b))
    idxsortie = np.argmin(np.abs(x - (x_b + a_b)))

    tentree = topchrono(psilibre, t, idxentree)
    tsortielibre = topchrono(psilibre, t, idxsortie)
    tau_0_num = tsortielibre - tentree
    tau_0_th = tlibre(k0, a_b)

    print(f"  t_entree      = {tentree:.4f}  (attendu {(x_b - x0) / vg:.4f})")
    print(f"  t_sortie      = {tsortielibre:.4f}  (attendu {(x_b + a_b - x0) / vg:.4f})")
    print(f"  \u03c4_{{0,num}}    = {tau_0_num:.4f}")
    print(f"  \u03c4_{{0,th}}     = a/v_g = {tau_0_th:.4f}")

    # --------------------------------------------------------
    # Partie 4.1.a-c - Simulation avec effet tunnel
    # --------------------------------------------------------
    print("\n--- 4.1.a-c : barrière de potentiel (V0 > 0) ---")
    x, t, psibarr, dx = schrodinger(nx, nt, xmin, xmax, tmin, tmax, k0, a_wp, x0, x_b, a_b, V0)

    tsortietunnel = topchrono(psibarr, t, idxsortie)
    tau_t_num = tsortietunnel - tentree
    tau_t_th = ttunnel(k0, a_b, V0)

    print(f" t_sortie tunnel= {tsortietunnel:.4f}")

    print(f"  \u03c4_{{t,num}} = {tau_t_num:.4f}")
    if tau_t_th is not None:
        print(f"  \u03c4_{{t,th}} (Hartman) = {tau_t_th:.4f}")
        print(f"  \u03c4_{{t,th}} / \u03c4_{{0,th}} = {tau_t_th / tau_0_th:.3f}  (< 1 : effet Hartman)")

    normeinit = getnorme(psibarr[0, :], dx)




    normefin = getnorme(psibarr[-1, :], dx)

    masktrans = x > x_b + a_b
    normetrans = getnorme(psibarr[-1, masktrans], dx)
    print(f"\n  Norme initiale    : {normeinit:.6f}")
    print(f"  Norme finale      : {normefin:.6f}")
    print(f"  Norme transmise (paquet numérique) : {normetrans:.4e}")
    

    # --------------------------------------------------------
    # Partie 4.1.d - Boucle de variation de la largeur a
    # --------------------------------------------------------

    print("\n--- 4.1.d : influence de la largeur a ---")
    aval_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    tau0_list = []
    tautnum_list = []
    tautth_list = []
    t2_list = []

    for aval in aval_list:
        tau0_list.append(tlibre(k0, aval))
        tautth_list.append(ttunnel(k0, aval, V0))
        t2_list.append(transmission(k0, aval, V0))

        _, _, psitmp, _ = schrodinger(
            nx, nt, xmin, xmax, tmin, tmax, k0, a_wp, x0, x_b, aval, V0
        )
        idxs = np.argmin(np.abs(x - (x_b + aval)))
        ts = topchrono(psitmp, t, idxs)
        tautnum_list.append(ts - tentree)

        print(
            f"  a={aval:.1f} : \u03c4_0={tau0_list[-1]:.3f}  "
            f"\u03c4_t_num={tautnum_list[-1]:.3f}  "
            f"|T|²={t2_list[-1]:.2e}"
        )

    # --------------------------------------------------------
    # Partie 4.1.e - Boucle de variation de la hauteur V0
    # --------------------------------------------------------
    print("\n--- 4.1.e : influence de V0 ---")
    v0val_list = [13.5, 14.0, 15.0, 17.0, 20.0, 25.0]
    v0valid = []
    tautv0_list = []

    t2v0_list = []

    for v0val in v0val_list:
        if v0val <= E:
            print(f"  V0={v0val:.1f} : E >= V0, au-dessus de la barrière")
            continue
        kap = np.sqrt(2 * m * (v0val - E)) / hbar
        tt = ttunnel(k0, a_b, v0val)
        t2 = transmission(k0, a_b, v0val)
        v0valid.append(v0val)
        tautv0_list.append(tt)
        t2v0_list.append(t2)
        print(
            f"  V0={v0val:.1f} : \u03ba·a={kap * a_b:.3f}  " 
            f"\u03c4_t_th={tt:.4f}  |T|²={t2:.2e}")

    # --------------------------------------------------------
    # Affichage des Graphiques
    # --------------------------------------------------------
    vplot = mur(x, x_b, a_b, V0)

    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    instants = [0, nt // 2, -1]
    titres = ["t = 0 (initial)", f"t = {t[nt // 2]:.2f}", f"t = {t[-1]:.2f} (final)"]

    for ax, j, titre in zip(axes, instants, titres):
        ymax = max(
            np.max(np.abs(psilibre[j, :]) ** 2),
            np.max(np.abs(psibarr[j, :]) ** 2),
        )
        echelle = 0.4 * ymax if ymax > 0 else 0.01
        barriereplot = vplot / V0 * echelle
        ax.fill_between(x, 0, barriereplot, alpha=0.25, color="gray",
                        label=f"Barrière V0={V0}")
        ax.plot(x, np.abs(psilibre[j, :]) ** 2, color="pink", linewidth=2, label="|\u03c8 libre|²")
        ax.plot(x, np.abs(psibarr[j, :]) ** 2, color="violet", linewidth=2,label="|\u03c8 tunnel|²")
        ax.set_title(titre)
        ax.set_xlabel("x")

        ax.set_ylabel("|\u03c8|²")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":")

    plt.suptitle(f"Évolution : k0={k0}, a={a_b}, V0={V0}, |T|²={transmission(k0, a_b, V0):.2e}",fontsize=10,)
    
    plt.tight_layout()
    plt.savefig("Figure_1_evolution.png", dpi=100)

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
