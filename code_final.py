#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Projet 211 - Identification parametrique et optimisation par SALLAM Shyrine et MESSAOUDI Adam

Identification paramétrique de la dynamique
d'antibiorésistance sous la
présence d'antibiotique

Programmes codés :
   1. MODELE : EDO et fonction de sensibilité
   2. Génération de données synthétiques
   3. Critère et gradient (par fonctions de sensibilité)
   4. Algorithme de Levenberg-Marquardt
   5. Intervalle de confiance
   6. Protocole expérimental
   7. MAIN : éxecution du script et affichage


"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Pour savoir où les fichiers se sont sauvegardés
SCRIPT_DIR = os.getcwd()
print(f"Sauvegarde dans : {SCRIPT_DIR}")


# 1. MODELE : EDO et fonction de sensibilité


def edo(x, t, p, u_fonction):
    """
    Calcule les dérivées temporelles du systeme étendu :
        x = [S, R, dS/da, dR/da, dS/db, dR/db, dS/dc, dR/dc]
    soit 2 variables d'etat + 6 fonctions de sensibilite.
    """
    S, R = x[0], x[1]
    sSa,sRa = x[2], x[3]   # sensibilités initiales par rapport à a
    sSb, sRb = x[4], x[5]   # sensibilités initiales par rapport à b
    sSc, sRc=x[6],x[7]   # sensibilités initiales par rapport à c
    a, b, c = p[0], p[1], p[2]
    u = u_fonction(t)           # entrée au temps t

    # Equations dynamiques
    dSdt = -(a + b*u)*S
    dRdt = (a+ b*u)*S - c*R

    # Sensibilités par rapport à a
    dsSadt = -(a + b*u)*sSa - S
    dsRadt = (a + b*u)*sSa + S - c*sRa

    # Sensibilités par rapport à b
    dsSbdt = -(a + b*u)*sSb - u*S
    dsRbdt = (a + b*u)*sSb + u*S - c*sRb

    # Sensibilités par rapport à c
    dsScdt = -(a + b*u)*sSc
    dsRcdt = (a + b*u)*sSc - c*sRc - R

    return np.array([dSdt, dRdt,
                     dsSadt, dsRadt,
                     dsSbdt, dsRbdt,
                     dsScdt, dsRcdt])


def euler_integrate(fonction, x0, t_tot, p, u_fonction, M=50):
    """
    Intégration par Euler explicite avec M sous-intervalles entre deux instants
    de mesure (pour la précision).
    """
    N = len(t_tot)
    dim = len(x0)
    X = np.zeros((N, dim))
    X[0] = x0
    x = x0.copy()

    for k in range(N - 1):
        dt = (t_tot[k + 1] - t_tot[k]) / M
        t = t_tot[k]
        for _ in range(M):
            t_actuel = t_tot[k]
            '''
            Le temps ne sert qu'à la fonction u_fonction(t), donc si l'on bloque t à t_tot[k],
            on a une approximation de u constante sur le pas de mesure,
            ce qui est cohérent avec les données disponibles.
            '''
            x = x + dt*fonction(x, t_actuel, p, u_fonction)
        X[k + 1] =x

    return X



# 2. Génération de données synthétiques


def generer_donnees(p_vrai, t_mes, u_fonction, S0, R0, sigma_bruit, seed=42):
    """
    Génère des mesures simulées y_obs = y_vrai + bruit_gaussien.
    """
    np.random.seed(seed) # Pour que les résultats de random soient toujours les mêmes
    x0 = np.array([S0, R0, 0, 0, 0, 0, 0, 0])  # CI : sensibilités nulles
    X = euler_integrate(edo, x0, t_mes, p_vrai, u_fonction, M=50)
    S = X[:, 0]
    R = X[:, 1]
    y_vrai = R/(S + R)

    y_obs = y_vrai + sigma_bruit*np.random.randn(len(t_mes))
    y_obs = np.clip(y_obs, 0.0, 1.0) # On ne garde que les valeurs dans [0, 1] (c'est une proportion)

    return y_obs, y_vrai



# 3. Critère et gradient (par fonctions de sensibilité)


def critere_et_gradient(p, t_mes, y_obs, u_fonction, S0, R0):
    """
    Calcul de c(p) = (1/2) * sum (y_obs[k] - y_m(p, t_k))^2
    et son gradient analytique g(p) = dc/dp en utilisant les sensibilités.
    """
    x0 = np.array([S0, R0, 0, 0, 0, 0, 0, 0])
    X = euler_integrate(edo, x0, t_mes, p, u_fonction, M=50)

    S, R = X[:, 0], X[:, 1]
    N_tot = S + R
    y_m = R/N_tot

    r = y_m -y_obs # Residus
    c = 0.5*np.sum(r ** 2) # Critere

    # Sensibilités de y_m par rapport à (a, b, c)
    sSa, sRa = X[:, 2], X[:, 3]
    sSb, sRb = X[:, 4], X[:, 5]
    sSc, sRc = X[:, 6], X[:, 7]

    dy_da = (sRa*N_tot - R*(sSa + sRa)) / (N_tot**2)
    dy_db = (sRb*N_tot - R*(sSb + sRb)) / (N_tot**2)
    dy_dc = (sRc*N_tot - R*(sSc +sRc)) / (N_tot**2)

    J = np.column_stack([dy_da, dy_db, dy_dc]) # Jacobienne (N x 3)
    g = J.T @ r # Gradient du critère

    return c, g, J, r



# 4. Algorithme de Levenberg-Marquardt


def gauss_newton_amorti(p0, t_mes, y_obs, u_fonction, S0, R0,
                        max_iter=50, tol=1e-8, mu0=1e-3):
    """
    Algorithme de Gauss-Newton avec amortissement Levenberg-Marquardt.
    """
    p = p0.copy().astype(float)
    mu = mu0
    historique = {'p': [p.copy()], 'c': [], 'g_norm': []}

    for it in range(max_iter):
        c, g, J, r = critere_et_gradient(p, t_mes, y_obs, u_fonction, S0, R0)
        historique['c'].append(c)
        historique['g_norm'].append(np.linalg.norm(g))

        if np.linalg.norm(g) < tol:
            print(f"  Convergence apres {it} iterations (|g| < {tol}).")
            break

        # Pas de Levenberg-Marquardt : (J^T J + mu I) delta p = -J^T r
        try:
            delta = np.linalg.solve(J.T @ J + mu*np.eye(3), -g)
        except np.linalg.LinAlgError:
            print(f"Approximation de la Hessienne mal conditionnée, on augmente mu.")
            mu *= 10
            continue

        p_candidat = p + delta

        # Contrainte : paramètres positifs (sens physique)
        if p_candidat[0] < 0 or p_candidat[1] < 0 or p_candidat[2] < 0:
            mu = mu*10
            continue

        # Mise à jour de mu
        c_candidat, _, _, _ = critere_et_gradient(p_candidat, t_mes, y_obs, u_fonction, S0, R0)
        if c_candidat < c:
            p = p_candidat
            mu = max(mu / 10, 1e-9)  # Succès : on diminue mu
            historique['p'].append(p.copy())
        else:
            mu = mu*10  # Echec : on augmente mu (se rapproche du gradient)

    return p, historique



# 5. Intervalle de confiance


def intervalle_confiance(p, t_mes, y_obs, u_fonction, S0, R0, sigma):
    """
    Calcule un intervalle de confiance a 95% pour chaque parametre.
    """
    _, _, J, _ = critere_et_gradient(p, t_mes, y_obs, u_fonction, S0, R0)
    try:
        Cov = sigma**2*np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        print("Attention : J^T J singulière, intervalle non calculable.")
        return None

    sigma_p = np.sqrt(np.diag(Cov))
    borne_inf = p - 1.96*sigma_p
    borne_sup = p +1.96*sigma_p

    return Cov, sigma_p, borne_inf, borne_sup



# 6. Protocol expérimental


def u_creneau(t):
    """u = 1 si t in [10, 20], sinon 0."""
    return 1 if 10 <= t <= 20 else 0

def u_zero(t):
    """u = 0 (pas d'antibiotique)."""
    return 0



# 7. MAIN : éxecution du script et affichage


def main():
    print(" IDENTIFICATION DES PARAMETRES D'ANTIBIORESISTANCE (Gauss-Newton Amorti) ")

    p_vrai = np.array([0.01, 0.5, 0.3])
    S0, R0 = 1.0e6, 1.0e3
    sigma = 0.01
    t_mes = np.arange(0, 31.0, 1)

    print(f"\nParamètres vrais : a* = {p_vrai[0]}, b* = {p_vrai[1]}, c* = {p_vrai[2]}")

    # Génération des données
    y_obs_creneau, y_vrai_creneau = generer_donnees(p_vrai, t_mes, u_creneau, S0, R0, sigma, seed=1)

    with open(os.path.join(SCRIPT_DIR, 'donnees_simulees.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_jour', 'u_creneau', 'y_obs_creneau'])
        for i, t in enumerate(t_mes):
            w.writerow([t, u_creneau(t), y_obs_creneau[i]])

    # Estimation (experience creneau)
    print("\n Estimation par Gauss-Newton Amorti (LM) ")
    p0 = np.array([0.1, 5, 3])  # Initialisation volontairement éloignée de la vérité
    p_opt, hist_opt = gauss_newton_amorti(p0, t_mes, y_obs_creneau, u_creneau, S0, R0, max_iter=30)

    err_rel = np.abs(p_opt - p_vrai) / p_vrai * 100
    print(f"\n  Estimee finale  : a = {p_opt[0]:.6f}, b = {p_opt[1]:.6f}, c = {p_opt[2]:.6f}")
    print(f"  Erreur relative : {err_rel[0]:.1f}% / {err_rel[1]:.1f}% / {err_rel[2]:.1f}%")

    # Intervalle de confiance
    print("\n Intervalle de confiance asymptotique 95% (Fisher) ")
    intervalle = intervalle_confiance(p_opt, t_mes, y_obs_creneau, u_creneau, S0, R0, sigma)
    if intervalle is not None:
        _, sigma_p, ic_inf, ic_sup = intervalle
        for i, nom in enumerate(['a', 'b', 'c']):
            print(f"  {nom} = {p_opt[i]:.5f}  IC95 = [{ic_inf[i]:.5f}, {ic_sup[i]:.5f}]")

    plot_resultats(t_mes, y_obs_creneau, y_vrai_creneau, p_opt, u_creneau, S0, R0, hist_opt, p_vrai)


def plot_resultats(t_mes, y_obs, y_vrai, p, u_fonction, S0, R0, hist_opt, p_vrai):
    """
    Génère une figure individuelle par graphique (7 fichiers PNG).
    Chaque figure est sauvegardée avec un nom explicite.
     - fig1_ajustement.png : Ajustement du modèle aux mesures
     - fig2_dynamique_SR.png : Dynamique interne des populations S et R
     - fig3_convergence.png : Convergence de l'algorithme
     - fig4_trajectoire_parametres.png : Trajectoire des paramètres normalisés
     - fig5_sensibilites.png : Fonctions de sensibilité
     - fig6_residus.png : Résidus de l'ajustement
     - fig7_histogramme_residus.png : Histogramme des résidus VS loi normale

     Utilisation de l'IA pour cette fonction
    """
    x0 = np.array([S0, R0, 0, 0, 0, 0, 0, 0])
    X_mes = euler_integrate(edo, x0, t_mes, p, u_fonction, M=50)
    y_mes = X_mes[:, 1] / (X_mes[:, 0] + X_mes[:, 1])

    # Figure 1 : Ajustement du modèle aux mesures
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_mes, y_vrai, 'g--', label='Réalité', linewidth=2)
    ax.plot(t_mes, y_obs, 'b.', label='Mesures bruitées', markersize=8)
    ax.plot(t_mes, y_mes, '-', color='purple', label='Modèle ajusté', linewidth=2)
    ax.axvspan(10, 20, alpha=0.2, color='orange', label='Antibiotique u=1')
    ax.set_xlabel('Temps (jours)')
    ax.set_ylabel('Proportion de cellules résistantes')
    ax.set_title('Ajustement du modèle aux mesures')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig1_ajustement.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2 : Dynamique interne des populations S et R
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t_mes, X_mes[:, 0], 'b-', label='S(t) (sensibles)', linewidth=2)
    ax.semilogy(t_mes, X_mes[:, 1], 'r-', label='R(t) (résistantes)', linewidth=2)
    ax.axvspan(10, 20, alpha=0.2, color='orange')
    ax.set_xlabel('Temps (jours)')
    ax.set_ylabel('Population (cellules)')
    ax.set_title('Dynamique interne des populations S et R')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig2_dynamique_SR.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3 : Convergence de l'algorithme
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(hist_opt['c'], '-o', color='tab:pink', label='Levenberg-Marquardt', markersize=5)
    ax.set_xlabel('Itérations')
    ax.set_ylabel('Critère c(p)')
    ax.set_title("Convergence de l'algorithme")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig3_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4 : Trajectoire des paramètres normalisés
    fig, ax = plt.subplots(figsize=(8, 5))
    p_arr = np.array(hist_opt['p'])
    for i, (nom, vrai) in enumerate(zip(['a', 'b', 'c'], p_vrai)):
        ax.plot(p_arr[:, i] / vrai, label=f'{nom}/{nom}*', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Itérations')
    ax.set_ylabel('Paramètres normalisés')
    ax.set_title('Trajectoire des paramètres normalisés')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig4_trajectoire_parametres.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5 : Fonctions de sensibilité
    S, R = X_mes[:, 0], X_mes[:, 1]
    Nt = S + R
    sSa, sRa = X_mes[:, 2], X_mes[:, 3]
    sSb, sRb = X_mes[:, 4], X_mes[:, 5]
    sSc, sRc = X_mes[:, 6], X_mes[:, 7]

    sy_a = (sRa * Nt - R * (sSa + sRa)) / (Nt**2)
    sy_b = (sRb * Nt - R * (sSb + sRb)) / (Nt**2)
    sy_c = (sRc * Nt - R * (sSc + sRc)) / (Nt**2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_mes, sy_a, 'b-', label=r'$\partial y_m / \partial a$', linewidth=2)
    ax.plot(t_mes, sy_b, 'r-', label=r'$\partial y_m / \partial b$', linewidth=2)
    ax.plot(t_mes, sy_c, 'g-', label=r'$\partial y_m / \partial c$', linewidth=2)
    ax.axvspan(10, 20, alpha=0.2, color='pink', label='u=1')
    ax.set_xlabel('Temps (jours)')
    ax.set_title('Fonctions de sensibilité')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig5_sensibilites.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 6 : Résidus de l'ajustement
    residus = y_obs - y_mes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(t_mes, residus, basefmt=' ')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.fill_between(t_mes, -2 * 0.01, 2 * 0.01, alpha=0.2, color='gray', label=r'$\pm 2\sigma$ (95\%)')
    ax.set_xlabel('Temps (jours)')
    ax.set_title("Résidus de l'ajustement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig6_residus.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 7 : Histogramme des résidus VS loi normale
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residus, bins=12, alpha=0.6, color='steelblue', edgecolor='black', density=True)
    x_range = np.linspace(residus.min(), residus.max(), 100)
    gauss = (1 / (0.01 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / 0.01)**2)
    ax.plot(x_range, gauss, 'r-', linewidth=2, label=r'$\mathcal{N}(0,\sigma^2)$')
    ax.set_xlabel('Résidus')
    ax.set_title('Distribution des résidus VS loi normale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'fig7_histogramme_residus.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("7 figures sauvegardées : fig1_ajustement.png à fig7_histogramme_residus.png")


if __name__ == '__main__':
    main()