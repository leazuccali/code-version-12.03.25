###
from __future__ import division

from trajectoire_pas_i import *
import math

eps = 1e-14  #


# Match input variable order as closely as possible
# def calc_mogi(x,y,xoff=0,yoff=0,d=3e3,dV=1e6,nu=0.25,output='cyl'):
# def forward(E,N,DEPTH,STRIKE,DIP,LENGTH,WIDTH,RAKE,SLIP,OPEN):
def forward(x, y, depth, length, width, dip, opening, strike,  slip = 0, rake = 0, nu=0.25):
    '''
    Calculate surface displacements for Okada85 dislocation model
    '''

    e = x
    n = y
    #print(strike)
    strike = np.deg2rad(strike)

    #dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    L = length
    W = width

    U1 = np.cos(rake) * slip
    U2 = np.sin(rake) * slip
    U3 = opening


    d = depth + np.sin(dip) * W / 2
    ec = e + np.cos(strike) * np.cos(dip) * W / 2
    nc = n - np.sin(strike) * np.cos(dip) * W / 2
    x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2
    y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W
    p = y * np.cos(dip) + d * np.sin(dip)
    q = y * np.sin(dip) - d * np.cos(dip)

    ux = - U1 / (2 * np.pi) * chinnery(ux_ss, x, p, L, W, q, dip, nu) - \
         U2 / (2 * np.pi) * chinnery(ux_ds, x, p, L, W, q, dip, nu) + \
         U3 / (2 * np.pi) * chinnery(ux_tf, x, p, L, W, q, dip, nu)
    uy =  -U1 / (2* np.pi) * chinnery(uy_ss, x, p, L, W, q, dip, nu) - \
          U2 / (2*np.pi) * chinnery(uy_ds, x, p, L, W, q, dip, nu) + \
          U3 / (2*np.pi) * chinnery(uy_tf, x, p, L, W, q, dip, nu)

    uz = - U1 / (2 * np.pi) * chinnery(uz_ss, x, p, L, W, q, dip, nu) - \
         U2 / (2 * np.pi) * chinnery(uz_ds, x, p, L, W, q, dip, nu) + \
         U3 / (2 * np.pi) * chinnery(uz_tf, x, p, L, W, q, dip, nu)

    ue = np.sin(strike) * ux - np.cos(strike) * uy
    un = np.cos(strike) * ux + np.sin(strike) * uy

    return ue, un, uz

#essai_froward = forward()

'''
% Notes for I... and K... subfunctions:
%
%	1. original formulas use Lame's parameters as mu/(mu+lambda) which
%	   depends only on the Poisson's ratio = 1 - 2*nu
%	2. tests for cos(dip) == 0 are made with "cos(dip) > eps"
%	   because cos(90*np.pi/180) is not zero but = 6.1232e-17 (!)
%	   NOTE: don't use cosd and sind because of incompatibility
%	   with Matlab v6 and earlier...
'''


def chinnery(f, x, p, L, W, q, dip, nu):
    ''' % Chinnery's notation [equation (24) p. 1143]'''
    u = (f(x, p, q, dip, nu) -
         f(x, p - W, q, dip, nu) -
         f(x - L, p, q, dip, nu) +
         f(x - L, p - W, q, dip, nu))
    return u


'''
% Displacement subfunctions
% strike-slip displacement subfunctions [equation (25) p. 1144]
'''


def ux_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = xi * q / (R * (R + eta)) + \
        I1(xi, eta, q, dip, nu, R) * np.sin(dip)
    k = (q != 0)
    # u[k] = u[k] + np.arctan2( xi[k] * (eta[k]) , (q[k] * (R[k])))
    u[k] = u[k] + np.arctan((xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def uy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + \
        q * np.cos(dip) / (R + eta) + \
        I2(eta, q, dip, nu, R) * np.sin(dip)
    return u


def uz_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + eta)) + \
        q * np.sin(dip) / (R + eta) + \
        I4(db, eta, q, dip, nu, R) * np.sin(dip)
    return u


def ux_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = q / R - \
        I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
    return u


def uy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = ((eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) -
         I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    k = (q != 0)
    u[k] = u[k] + np.cos(dip) * np.arctan((xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def uz_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (db * q / (R * (R + xi)) -
         I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip))
    k = (q != 0)
    # u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
    u[k] = u[k] + np.sin(dip) * np.arctan((xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def ux_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = q ** 2 / (R * (R + eta)) - \
        I3(eta, q, dip, nu, R) * (np.sin(dip) ** 2)
    return u


def uy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = - (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - \
        np.sin(dip) * xi * q / (R * (R + eta)) - \
        I1(xi, eta, q, dip, nu, R) * (np.sin(dip) ** 2)
    k = (q != 0)
    # u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
    u[k] = u[k] + np.sin(dip) * np.arctan((xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def uz_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + \
        np.cos(dip) * xi * q / (R * (R + eta)) - \
        I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) ** 2
    k = (q != 0)  # not at depth=0?
    u[k] = u[k] - np.cos(dip) * np.arctan((xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def I1(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * (- xi / (np.cos(dip) * (R + db))) - \
            np.sin(dip) / np.cos(dip) * \
            I5(xi, eta, q, dip, nu, R, db)
    else:
        I = -(1 - 2 * nu) / 2 * xi * q / (R + db) ** 2
    return I


def I2(eta, q, dip, nu, R):
    I = (1 - 2 * nu) * (-np.log(R + eta)) - \
        I3(eta, q, dip, nu, R)
    return I


def I3(eta, q, dip, nu, R):
    yb = eta * np.cos(dip) + q * np.sin(dip)
    #print(yb)
    db = eta * np.sin(dip) - q * np.cos(dip)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + \
            np.sin(dip) / np.cos(dip) * \
            I4(db, eta, q, dip, nu, R)
    else:
        I = (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))
    return I


def I4(db, eta, q, dip, nu, R):
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * 1.0 / np.cos(dip) * \
            (np.log(R + db) - np.sin(dip) * np.log(R + eta))
    else:
        I = - (1 - 2 * nu) * q / (R + db)
    return I


def I5(xi, eta, q, dip, nu, R, db):
    X = np.sqrt(xi ** 2 + q ** 2)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * 2 / np.cos(dip) * \
            np.arctan((eta * (X + q * np.cos(dip)) + X * (R + X) * np.sin(dip)) /
                      (xi * (R + X) * np.cos(dip)))
        I[xi == 0] = 0
    else:
        I = -(1 - 2 * nu) * xi * np.sin(dip) / (R + db)
    return I


def calcul_parametres_effectifs_okada(longueur, ouverture, distance_parcourue, G, E, P_load):

    if distance_parcourue < longueur:
        longueur_okada = distance_parcourue
        vol_eff = (((1 - 0.25**2)*G*abs(P_load)) * distance_parcourue**3 ) / E
        print(vol_eff)
        ouverture_okada = (math.pi * vol_eff) / (2 * longueur_okada**2)
    else:
        longueur_okada = longueur
        ouverture_okada = ouverture

    return longueur_okada, ouverture_okada





def liaison_modele_okada(vec_X_haut,vec_X_bas,vec_Z_haut, vec_Z_bas,pas_okada, gmin_x,gmax_x, longueur, opening):   # ,vec_temps, vec_distance, vec_dist_cumu,L):

    grille_x = np.arange(gmin_x, gmax_x, pas_okada)
    grille_y = np.arange(0, pas_okada, pas_okada)
    mX, mY = np.meshgrid(grille_x, grille_y)

    ### Calculs du centre entre le haut et la bas de la trajectoire en chaque temps
    vec_x_c = (vec_X_haut + vec_X_bas)/2
    #print('vecxc', vec_x_c)
    vec_z_c = -(vec_Z_haut + vec_Z_bas)/2
    #print('veczc', vec_z_c)
    vec_z_c_abs = abs(vec_z_c)

    #nb_points = len(vec_X_haut)

    # Calcul du strike
    if vec_X_bas >= 0:
        strike = 180
    else:
        strike = 0

    ##Calcul du dip pour chaque temps
    val_tan = (vec_Z_haut - vec_Z_bas) / (vec_X_haut - vec_X_bas)
    vec_dip = np.arctan(abs(val_tan))

    #Calcul des vecteurs de déplacement ue, un et uz
    ux = []
    uy = []
    uz = []

    ux, uy, uz = forward((mX - vec_x_c), mY, vec_z_c_abs, longueur, 1000, vec_dip, opening, strike)

    return ux, uy, uz, grille_x, grille_y, mX, mY


def une_particule_deplacement_pas_i(X_haut, Z_haut, X_bas, Z_bas, longueur, ouverture, distance_parcourue_i, R, G, mu, E, vol, P_load, pas_okada, xmin, xmax):

    #longueur_okada, ouverture_okada = calcul_parametres_effectifs_okada(longueur, ouverture, distance_parcourue_i, G, E, P_load)

    #ux, uy, uz, x, y, mX, mY = liaison_modele_okada(X_haut, X_bas, Z_haut, Z_bas, pas_okada, xmin, xmax,
    #                                                longueur_okada, ouverture_okada)


    if distance_parcourue_i < longueur:
        longueur_okada = distance_parcourue_i
        vol_eff = (((1 - 0.25**2)*G*abs(P_load)) * distance_parcourue_i**3 ) / E
        #print(vol_eff)
        ouverture_okada = (math.pi * vol_eff) / (2 * longueur_okada**2)
    else:
        longueur_okada = longueur
        ouverture_okada = ouverture

    #longueur_okada, ouverture_okada

    grille_x = np.arange(xmin, xmax, pas_okada)
    grille_y = np.arange(0, pas_okada, pas_okada)
    mX, mY = np.meshgrid(grille_x, grille_y)

    ### Calculs du centre entre le haut et la bas de la trajectoire en chaque temps
    vec_x_c = (X_haut + X_bas) / 2
    # print('vecxc', vec_x_c)
    vec_z_c = -(Z_haut + Z_bas) / 2
    # print('veczc', vec_z_c)
    vec_z_c_abs = abs(vec_z_c)

    # nb_points = len(vec_X_haut)

    # Calcul du strike
    if X_bas >= 0:
        strike = 180
    else:
        strike = 0

    ##Calcul du dip pour chaque temps
    val_tan = (Z_haut - Z_bas) / (X_haut - X_bas)
    vec_dip = np.arctan(abs(val_tan))

    # Calcul des vecteurs de déplacement ue, un et uz
    ux = []
    uy = []
    uz = []

    ux, uy, uz = forward((mX - vec_x_c), mY, vec_z_c_abs, longueur_okada, 1000, vec_dip, ouverture_okada, strike)



    # fig3, ax3 = plt.subplots(figsize=(12, 10))
    # indice_0_ref = np.nonzero(mY == 0)
    # ux_val_ref = ux
    # uz_val_ref = uz
    # ux_p_ref = ux_val_ref[indice_0_ref]
    # uz_p_ref = uz_val_ref[indice_0_ref]
    #
    # plt.plot(grille_x / 1000, uz_p_ref * 1000, 'k-',
    #          label=f"Particule de référence. R = {R:2.2f}. G = {G:2.2f}. mu = {mu} Pas. E = {(E / 1000000000) :2.2f} GPa. Vol = {vol / 1000000000} km³.")
    # plt.title("Profile of vertical displacement", fontsize=20, fontweight='bold')
    # #plt.xlabel('X (km)')
    # #plt.ylabel(r'$U_x$ (mm)')
    # #plt.legend()
    # plt.xlabel('X (m)', fontweight='bold', fontsize=20)
    # plt.xticks(fontsize=20, fontweight='bold')
    # plt.xlim(-30, 30)
    #
    # plt.yticks(fontsize=20, fontweight='bold')
    # plt.ylabel(r'$U_x$ (mm)', fontweight='bold', fontsize=20)
    #
    # plt.show()
    #
    # fig4, ax4 = plt.subplots(figsize=(12, 10))
    # indice_0_ref = np.nonzero(mY == 0)
    # ux_val_ref = ux
    # uz_val_ref = uz
    # ux_p_ref = ux_val_ref[indice_0_ref]
    # uz_p_ref = uz_val_ref[indice_0_ref]
    #
    # plt.plot(grille_x / 1000, ux_p_ref * 1000, 'k-',
    #          label=f"Particule de référence. R = {R:2.2f}. G = {G:2.2f}. mu = {mu} Pas. E = {(E / 1000000000) :2.2f} GPa. Vol = {vol / 1000000000} km³.")
    # plt.title("Profile of horizontal displacement", fontsize=20, fontweight='bold')
    # #plt.xlabel('X (km)')
    # #plt.ylabel(r'$U_x$ (mm)')
    # #plt.legend()
    # plt.xlabel('X (m)', fontweight='bold', fontsize=20)
    # plt.xticks(fontsize=20, fontweight='bold')
    # plt.xlim(-30, 30)
    # plt.ylabel(r'$U_z$ (mm)', fontweight='bold', fontsize=20)
    # plt.yticks(fontsize=20, fontweight='bold')
    #
    # plt.show()


    return ux, uz, grille_x, mY, vec_x_c, vec_z_c, longueur_okada, ouverture_okada, vec_dip, strike





if __name__ == "__main__":
    xmin = -30000  # m
    xmax = 30000  # m
    zmin = -11000  # m
    zmax = -1  # m
    pas_trajectoire = 100
    pas_vect = 2000
    P_load = -15000000  # MPa
    rayon_load = 10000  # m
    norme = 1  # Si norme = 1, alors l'unité est le mètre. Si norme = 1000, alors l'unité est le km.
    pas_okada = 100
    temps_i = 0

    R = 0.3
    G = 0.2
    mu = 100
    E = 10 ** 9
    vol = 10 ** 7
    start_point = np.array([[8000, -8000]])
    pas_temps = 60
    temps_voulu = 10000

    ############Trajectoire

    extension, longueur, vitesse, ouverture = calcul_parametres(R, G, mu, E, vol, P_load)
    # print("La longueur de la remontée magmatique est de :", longueur, "m.")

    mesh_X, mesh_Z, sig_xx, sig_zz, sig_xz = fonction_watanabe(xmin, xmax, zmin, zmax, pas_trajectoire, P_load,
                                                               rayon_load, extension)

    # aff1 = affichage_champs(x_min, x_max, z_min, z_max, mesh_X, mesh_Z, sig_xx, sig_zz, sig_xz, norme)

    sig_1, sig_3, u_sig_1, v_sig_1, u_sig_1_eff, v_sig_1_eff, u_sig_3, v_sig_3, mesh_vec = calcul_sig1_sig3(
        mesh_X, mesh_Z, sig_xx, sig_zz, sig_xz, pas_vect, G)

    vec_Xt, vec_Xt_eff, vec_Zt, long_magma, vec_temps_eff= streamplot_trajectoire(
        mesh_X, mesh_Z, u_sig_1_eff, v_sig_1_eff, start_point, norme, R, G, vitesse, temps_i)

    # Etat de la trajectoire au pas i

    nouveau_pas = temps_i + pas_temps

    X_haut, Z_haut, new_vec_Xt, new_vec_Zt, new_vec_temps = nouvelle_valeur(vec_Xt_eff, vec_Zt, vec_temps_eff,
                                                                            nouveau_pas)

    vec_Xt_drop_haut, vec_Zt_drop_haut, vec_temps_drop_haut, vec_distance_drop_haut, vec_distance_parc_drop,  distance_parcourue_i = vecteurs_temps_i(
        new_vec_Xt, new_vec_Zt, new_vec_temps, nouveau_pas)

    X_bas, Z_bas = point_bas(X_haut, 0, vec_Xt_drop_haut, vec_Zt_drop_haut, vec_distance_drop_haut,
                             vec_distance_parc_drop, longueur)

    vecXt = list(vec_Xt_drop_haut)
    vecZt = list(vec_Zt_drop_haut)

    trajectoire_x_i, trajectoire_z_i = une_trajectoire_haute_basse(vecXt, vecZt, X_bas, Z_bas, X_haut)


    #distance_parcourue = vec_distance_parc_drop[-1]

    # Déplacements
    # Valeurs différentes que les valeurs théoriques


    '''longueur_okada, ouverture_okada = calcul_parametres_effectifs_okada(longueur, distance_parcourue_i, G, E, P_load)



    ux, uy, uz, x, y, mX, mY = liaison_modele_okada(X_haut, X_bas, Z_haut, Z_bas, pas_okada, xmin, xmax,
                                                    longueur_okada, ouverture_okada)

    fig3, ax3 = plt.subplots(figsize=(12, 10))
    indice_0_ref = np.nonzero(mY == 0)
    ux_val_ref = ux
    uz_val_ref = uz
    ux_p_ref = ux_val_ref[indice_0_ref]
    uz_p_ref = uz_val_ref[indice_0_ref]

    plt.plot(x / 1000, uz_p_ref * 1000, 'r--',
             label=f"Particule de référence. R = {R:2.2f}. G = {G:2.2f}. mu = {mu} Pas. E = {(E / 1000000000) :2.2f} GPa. Vol = {vol / 1000000000} km³.")
    plt.title("Profils de déplacements verticaux")
    plt.xlabel('X (km)')
    plt.ylabel(r'$U_x$ (mm)')
    plt.legend()
    plt.show()'''

    #test = une_particule_deplacement_pas_i(X_haut, Z_haut, X_bas, Z_bas, longueur, distance_parcourue_i, G, E, P_load, pas_okada, xmin, xmax)
    test = une_particule_deplacement_pas_i(X_haut, Z_haut, X_bas, Z_bas, longueur, ouverture, distance_parcourue_i, R, G, mu, E, vol, P_load, pas_okada, xmin, xmax)

