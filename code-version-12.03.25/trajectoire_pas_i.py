import numpy as np

from trajectoire_particules import *
import bisect
def nouvelle_valeur(vec_Xt, vec_Zt, vec_temps, valeur_temps):
    '''
    Calcule le point situé sur la ligne entre deux points déjà calculés, selon la distance à laquelle le point se situe,
    pour un pas de temps défini.
    :param vec_Xt: Vecteur des valeurs X
    :param vec_Zt: Vecteur des valeurs Z
    :param vec_temps: Vecteur du temps auquelles correspondent les coordonnées (X,Z)
    :param valeur_temps: Valeur du temps pour laquelle les coordonnées sont recherchées.
    :return: nouveau_x, nouveau_z : les coordonnées (X,Z) d'un point à un temps donné.
    '''
    index_temps = bisect.bisect(vec_temps, valeur_temps)
    #print("Avant insertion")
    #print(vec_Xt[0:5])
    #print(vec_Zt[0:5])
    #print(vec_temps[0:5])

    if (index_temps == 0):
        nouveau_x = vec_Xt[index_temps]
        nouveau_z = vec_Zt[index_temps]

        insertion_x = vec_Xt.insert(index_temps, nouveau_x)
        insertion_z = vec_Zt.insert(index_temps, nouveau_z)
        insertion_temps = vec_temps.insert(index_temps, valeur_temps)

    elif (index_temps == len(vec_temps)):
        nouveau_x = vec_Xt[index_temps - 1]
        nouveau_z = vec_Zt[index_temps - 1]

        insertion_x = vec_Xt.insert(index_temps, nouveau_x)
        insertion_z = vec_Zt.insert(index_temps, nouveau_z)
        insertion_temps = vec_temps.insert(index_temps, valeur_temps)

    else:
        index_moins = index_temps - 1
        index_plus = index_temps + 1

        diff_moins = valeur_temps - vec_temps[index_moins]
        diff_plus = vec_temps[index_plus - 1] - valeur_temps
        diff_totale = vec_temps[index_plus - 1] - vec_temps[index_moins]

        coef1 = (diff_totale - diff_moins) / diff_totale
        coef2 = (diff_totale - diff_plus) / diff_totale

        nouveau_x = coef1 * vec_Xt[index_moins] + coef2 * vec_Xt[index_plus - 1]
        nouveau_z = coef1 * vec_Zt[index_moins] + coef2 * vec_Zt[index_plus - 1]

        insertion_x = vec_Xt.insert(index_temps, nouveau_x)
        insertion_z = vec_Zt.insert(index_temps, nouveau_z)
        insertion_temps = vec_temps.insert(index_temps, valeur_temps)

    #print("X_haut", vec_Xt)
    #print("Z_haut", vec_Zt)
    #print("taille Xhaut après calcul du point haut", len(vec_Xt))

    #new_vec_Xt = list(insertion_x)
    #new_vec_Zt = list(insertion_z)
    #print("Après insertion")
    #print(vec_Xt[0:5])
    #print(vec_Zt[0:5])
    #print(vec_temps[0:5])

    return nouveau_x, nouveau_z, vec_Xt, vec_Zt, vec_temps


def vecteurs_temps_i(vec_Xt, vec_Zt, vec_temps, nouveau_pas):

    #Calcul des distances entre chaque point et des distance cumulées
    vec_distance_i = (np.diff(vec_Xt) ** 2 + np.diff(vec_Zt) ** 2) ** 0.5
    vec_distance_pas_i = np.insert(vec_distance_i, 0, 0)

    vec_DC_pas_i = np.cumsum(vec_distance_pas_i)
    distance_parcourue = np.max(vec_DC_pas_i)

    #print("La distance parcourue au temps i est ", distance_parcourue)

    # Insertion du point haut dans vec_Xt, drop ce qui est au dessus et calcul des distance cumulées jusqu'à temps_i + pas
    new_tab1 = [vec_Xt, vec_Zt, vec_temps, vec_distance_pas_i, vec_DC_pas_i]
    new_tab = np.transpose(new_tab1)
    df_new_tab = pd.DataFrame(new_tab)
    df_new_tab_drop_haut = df_new_tab.drop(df_new_tab[(df_new_tab[2] > nouveau_pas)].index)
    #print(df_new_tab_drop_haut)

    vec_new_tab_drop_haut = df_new_tab_drop_haut.to_numpy()
    vec_tab_drop_haut = np.transpose(vec_new_tab_drop_haut)

    vec_Xt_drop_haut = vec_tab_drop_haut[0]
    vec_Zt_drop_haut = vec_tab_drop_haut[1]
    vec_temps_drop_haut = vec_tab_drop_haut[2]
    vec_distance_drop_haut = vec_tab_drop_haut[3]
    vec_distance_parc_drop = vec_tab_drop_haut[4]
    #print("vec_distance parc", vec_distance_parc_drop)

    # print("vec Xt drop haut", vec_Xt_drop_haut)
    # print("vec Zt", vec_Zt_drop_haut)
    # print("vec temps", vec_temps_drop_haut)

    distance_parcourue_i = vec_distance_parc_drop[-1]
    #print("La distance parcourue par le magma au temps", nouveau_pas, "est de ", distance_parcourue_i, "m.")

    #print("Vecteur position Xt coupé ", vec_Xt_drop_haut)
    #print("Vecteur temps coupé : ", vec_temps_drop_haut)


    return vec_Xt_drop_haut, vec_Zt_drop_haut, vec_temps_drop_haut, vec_distance_drop_haut, vec_distance_parc_drop, distance_parcourue_i



def trajectoire_pas_temps(Xt_eff, Zt, vec_temps_eff, pas_temps, nombre_points, temps_i):
    '''
    Calcul de l'ensemble des coordonnées de n points espacés d'un pas de temps régulier.

    :param Xt_eff: Vecteurs des valeurs de X
    :param Zt: Vecteur des valeurs de Z
    :param vec_temps_eff: Vecteur du temps auquelles correspondent les coordonnées (X,Z)
    :param pas_temps: Pas de temps.
    :param nombre_points: Nombre de points (pas de temps) considérés
    :return: vec_X_pas, vec_Z_pas, vec_temps_pas, vec_distance_pas, vec_dist_cumu_pas
    Coordonnées (X,Z) de chaque nouveau point au tempt t. Vecteurs distance entre 2 points et distances cumulées aux points
    considérés.
    '''


    #Transformer arrays en listes
    Xt_eff_list = list(Xt_eff)
    Zt_list = list(Zt)
    vect_list = list(vec_temps_eff)
    #Définition du vecteur temps
    #vec_temps_pas = np.arange(temps_i, temps_i + 1 * pas_temps, pas_temps)
    vec_temps_pas = [temps_i, temps_i + pas_temps]
    #print(vec_temps)
    #Nouveaux vecteurs vides X et Z
    vec_X_pas = []
    vec_Z_pas = []

    for val in vec_temps_pas:   #ou vec_temps_eff
        X_pas, Z_pas, valeur_temps = nouvelle_valeur(Xt_eff_list, Zt_list, vect_list, val)
        vec_X_pas.append(X_pas)
        vec_Z_pas.append(Z_pas)

    #plt.figure()
    #plt.plot(vec_X_pas, vec_Z_pas, 'ro')
    #plt.title("Point de la trajectoire selon pas de temps")
    #plt.show()

    vec_dist = (np.diff(vec_X_pas)**2 + np.diff(vec_Z_pas)**2)**0.5
    vec_distance_pas = np.insert(vec_dist, 0, 0)

    vec_dist_cumu_pas = np.cumsum(vec_distance_pas)
    long_magma = np.max(vec_dist_cumu_pas)
    #print("La longueur de la trajectoire du magma est de", long_magma, "m.")

    #print("vecteur X pour chaque pas", vec_X_pas)
    #print("taille vecteur X pour chaque pas", len(vec_X_pas))
    #print("vecteur temps pour chaque pas", vec_temps_pas)

    return vec_X_pas, vec_Z_pas, vec_temps_pas, vec_distance_pas, vec_dist_cumu_pas


def point_bas(x, z, vec_X, vec_Z, vec_D, vec_DC, L):
    '''
    Calcul du point minimal d'une trajectoire selon les coordonnées (x,z) du front et selon la longueur L de la remontée
    magmatique.
    :param x: Coordonnée X du front.
    :param z: Coordonnée Z du front
    :param vec_X: Vecteur des coordonées X de la trajectoire
    :param vec_Z: Vecteur des coodonnées Z de la trajectoire
    :param vec_D: Vecteur des distances entre deux points de la trajectoire
    :param vec_DC: Vecteur des distances cumulées de la trajectoire
    :param L: Longueur de la remontée magmatique
    :return: (x_b, z_b) : coordonnées basses de la remontée magmatique.
    '''
    #i = vec_X.index(x)
    #i_z = vec_Z.index(z)
    #print(i)


    if vec_DC[-1] <= L:
        x_b = vec_X[0]
        z_b = vec_Z[0]

    else:
        k = vec_DC[-1] - L    # k différence entre Lon cumulée et L. Distance cumulée " basse " DCB

        #On intercale DCB sur la trajectoire vec_X vec_Z.
        #Pour cela on la place déjà sur vec_DC le vecteur des trajectoires cumuluées
        index_k = bisect.bisect(vec_DC, k)

        indice_moins = index_k - 1
        indice_plus = index_k

        reste_a_parcourir = vec_DC[indice_plus] - k

        coef = reste_a_parcourir / vec_D[indice_plus]

        x_b = coef * vec_X[indice_moins] + (1 - coef) * vec_X[indice_plus]
        z_b = coef * vec_Z[indice_moins] + (1 - coef) * vec_Z[indice_plus]


    return x_b,z_b


def une_trajectoire_haute_basse(vec_X_haut, vec_Z_haut, vec_X_bas, vec_Z_bas, x):
    '''
    Extraction des coordonnées d'une remontée magmatique à partir des coordonnées de son front et de son point le plus bas.
    :param vec_X_haut: Coordonnées X de la trajectoire - du front.
    :param vec_Z_haut: Coordonnées Z de la trajectoire - du front.
    :param vec_X_bas: Coordonnées X du bas de la remontée.
    :param vec_Z_bas: Coordonnées Z du bas de la remontée.
    :param x: Coordonnée X du front considéré.
    :return: vecX_coupe, vecZ_coupe : Coordonnées X et Z de la remontée, comprises entre le front et le bas.
    '''
    # Indice de la valeur Xhaute que l'on considère.
    indice_X_haut = vec_X_haut.index(x)
    #print('indice haut', indice_X_haut)
    #Recupération des valeurs Xbasse et Zbasse correspondantes
    #X_bas = vec_X_bas[indice_X_haut]
    #Z_bas = vec_Z_bas[indice_X_haut]

    X_bas = vec_X_bas
    Z_bas = vec_Z_bas

    #print('val Xbas', X_bas)
    #print(Z_bas)
    #Attribution d'une place pour la valeur Xbasse dans le vecteur trajectoire

    if x >=0:

        indice_X_bas = bisect.bisect(vec_X_haut, X_bas)
        #print('nouveau indice pour valeur Xbas', indice_X_bas)
        #création d'un vecteur avec la valeur Xbasse considérée et les autres valeurs Xhautes
        vec_X_HB_ar = np.insert(vec_X_haut, indice_X_bas, X_bas)
        vec_Z_HB_ar = np.insert(vec_Z_haut, indice_X_bas, Z_bas)

        vec_X_HB = list(vec_X_HB_ar)
        vec_Z_HB = list(vec_Z_HB_ar)
        #print(vec_X_HB)

        vecX_coupe = []
        vec_indice_X_coupe = []
        for valX in vec_X_HB:
            if valX >= X_bas and valX <= x:
                vecX_coupe.append(valX)
                indiceX = vec_X_HB.index(valX)
                vec_indice_X_coupe.append(indiceX)

        #print(vecX_coupe)
        #print(vec_indice_X_coupe)

        vecZ_coupe = []
        for indice in vec_indice_X_coupe:
            val = vec_Z_HB[indice]
            vecZ_coupe.append(val)

        #print(vecZ_coupe)

    else:
        ##### Pour x négatif
        vec_X_haut_liste = list(vec_X_haut)
        #print(vec_X_haut_liste)
        vec_X_haut_croissant = sorted(vec_X_haut_liste)
        #print(vec_X_haut_croissant)
        vec_X_haut_croissant_tri = np.array(vec_X_haut_croissant)
        #print(vec_X_haut_croissant_tri)

        indice_X_bas_dans_X_haut = bisect.bisect(vec_X_haut_croissant_tri, X_bas)
        #print(indice_X_bas_dans_X_haut)

        vec_X_haut_croissant_tri_ar = np.insert(vec_X_haut_croissant_tri, indice_X_bas_dans_X_haut, X_bas)
        #print(vec_X_haut_croissant_tri_ar)
        vec_X_HB_croissant = list(vec_X_haut_croissant_tri_ar)
        #print(vec_X_HB_croissant)
        vec_X_HB = sorted(vec_X_HB_croissant, reverse=True)

        indice_X_HB = len(vec_X_HB) - indice_X_bas_dans_X_haut - 1
        #print(indice_X_HB)
        vec_Z_HB_ar = np.insert(vec_Z_haut, indice_X_HB, Z_bas)
        vec_Z_HB = list(vec_Z_HB_ar)

        #print(vec_X_HB)
        #print(len(vec_X_HB))
        #print(vec_Z_HB)
        #print(len(vec_Z_HB))

        vecX_coupe = []
        vec_indice_X_coupe = []
        for valX in vec_X_HB:
            if valX >= x and valX <= X_bas:
                vecX_coupe.append(valX)
                indiceX = vec_X_HB.index(valX)
                vec_indice_X_coupe.append(indiceX)

        # print(vecX_coupe)
        # print(vec_indice_X_coupe)

        vecZ_coupe = []
        for indice in vec_indice_X_coupe:
            val = vec_Z_HB[indice]
            vecZ_coupe.append(val)




    #return vecX_coupe, vecZ_coupe
    return vecX_coupe, vecZ_coupe




def une_trajectoire_pas_i(vec_Xt_eff, vec_Zt, vec_temps_eff, temps_courant, pas_temps, longueur):
    nouveau_pas = temps_courant + pas_temps
    #nouveau_pas = temps_courant
    #print("On calcul pour le pas à ", nouveau_pas, "secondes")

    # vec_Xt_eff_l = list(vec_Xt_eff)
    # vec_Zt_l = list(vec_Zt)
    # vec_temps_l = list(vec_temps_eff)
    #print(len(vec_temps_eff))

    X_haut, Z_haut, new_vec_Xt, new_vec_Zt, new_vec_temps = nouvelle_valeur(vec_Xt_eff, vec_Zt, vec_temps_eff,
                                                                            nouveau_pas)

    #print("X_haut", X_haut)
    #print("Z_haut", Z_haut)
    #print("new vec Xt", len(new_vec_Xt))
    #print("new vex Zt", len(new_vec_Zt))
    #print("new vec temps", len(new_vec_temps))


    vec_Xt_drop_haut, vec_Zt_drop_haut, vec_temps_drop_haut, vec_distance_drop_haut, vec_distance_parc_drop, distance_parcourue_i = vecteurs_temps_i(
        new_vec_Xt, new_vec_Zt, new_vec_temps, nouveau_pas)

    ## Maintenant calcul du point bas à au temps considéré

    X_bas, Z_bas = point_bas(X_haut, 0, vec_Xt_drop_haut, vec_Zt_drop_haut, vec_distance_drop_haut,
                             vec_distance_parc_drop, longueur)
    #print("X bas", X_bas)
    #print("Z bas", Z_bas)


    ## Maintenant trajectoire : tous les points de la trajectoire au moment voulu. Il faut intercaler xbas zbas

    vecXt = list(vec_Xt_drop_haut)
    vecZt = list(vec_Zt_drop_haut)

    trajectoire_x_i, trajectoire_z_i = une_trajectoire_haute_basse(vecXt, vecZt, X_bas, Z_bas, X_haut)

    #print("dim trajectoire X", len(trajectoire_x_i))
    #print("dim trajectoire Z", len(trajectoire_z_i))


    #print("Trajectoire pas i", trajectoire_x_i)


    return vecXt, vecZt, trajectoire_x_i, trajectoire_z_i, X_haut, Z_haut, X_bas, Z_bas, distance_parcourue_i


#G, E, P_load, pas_okada, xmin, xmax)








if __name__ == '__main__':
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

    vec_Xt, vec_Xt_eff, vec_Zt, long_magma, vec_temps_eff, = streamplot_trajectoire(
        mesh_X, mesh_Z, u_sig_1_eff, v_sig_1_eff, start_point, norme, R, G, vitesse, temps_i)

    # Pas i
    # Fonction qui cherche la position du point au pas i et qui enlève les points au dessous/dessus, qui intercale ce point et qui enlève les valeurs en dessous/au dessus de temps_i   temps_i+pas
    # Point haut + vecteur distance cumulées

    nouveau_pas = temps_i + pas_temps

    #vec_Xt_eff_l = list(vec_Xt_eff)
    #vec_Zt_l = list(vec_Zt)
    #vec_temps_l = list(vec_temps_eff)

    X_haut, Z_haut, new_vec_Xt, new_vec_Zt, new_vec_temps = nouvelle_valeur(vec_Xt_eff, vec_Zt, vec_temps_eff, nouveau_pas)
    print("X_haut", X_haut)
    print("Z_haut", Z_haut)
    #print("taille Xhaut après calcul du point haut", len(new_vec_Xt))
    print(len(new_vec_Xt))
    print(len(new_vec_Zt))
    print(len(new_vec_Xt))


    vec_Xt_drop_haut, vec_Zt_drop_haut, vec_temps_drop_haut, vec_distance_drop_haut, vec_distance_parc_drop, distance_parcourue_i = vecteurs_temps_i(new_vec_Xt, new_vec_Zt, new_vec_temps, nouveau_pas)


    ## Maintenant calcul du point bas à au temps considéré

    """X_bas, Z_bas = point_bas(X_haut, 0, vec_Xt_drop_haut, vec_Zt_drop_haut, vec_distance_drop_haut, vec_distance_parc_drop, longueur )
    print("X bas", X_bas)
    print("Z bas", Z_bas)"""

    ## Maintenant trajectoire : tous les points de la trajectoire au moment voulu. Il faut intercaler xbas zbas

    """vecXt = list(vec_Xt_drop_haut)
    vecZt = list(vec_Zt_drop_haut)

    trajectoire_x_i, trajectoire_z_i = une_trajectoire_haute_basse(vecXt, vecZt, X_bas, Z_bas, X_haut)

    print("Trajectoire pas i", trajectoire_x_i)"""


    #vecXt, vecZt, trajectoire_x_i, trajectoire_z_i, X_haut, Z_haut, X_bas, Z_bas, longueur, distance_parcourue_i, G, E, P_load, pas_okada, xmin, xmax = une_trajectoire_pas_i(vec_Xt_eff, vec_Zt, vec_temps_eff, temps_i ,pas_temps, G, E, P_load, pas_okada, xmin, xmax)


    '''fig1, ax1 = plt.subplots(figsize=(12, 10))
    plt.plot(vecXt, vecZt, 'bo-', markersize=6, label='Points de la trajectoire')
    #plt.plot(start_point[0][0] / norme, start_point[0][1] / norme, 'ko', markersize=12, label="Point de départ")
    plt.plot(trajectoire_x_i, trajectoire_z_i, 'ko', label= 'Trajectoire au temps donné')

    plt.pcolormesh(mesh_X / norme, mesh_Z / norme, sig_3, cmap='YlOrRd', shading="gouraud")
    plt.colorbar(location="bottom", label=r"Amplitude de $\sigma_{3}$")
    plt.quiver(mesh_X[mesh_vec] / norme, mesh_Z[mesh_vec] / norme, u_sig_1[mesh_vec], v_sig_1[mesh_vec],
               label=r"Direction $\sigma_1$")
    plt.quiver(mesh_X[mesh_vec] / norme, mesh_Z[mesh_vec] / norme, u_sig_1_eff[mesh_vec], v_sig_1_eff[mesh_vec],
               label=r"Direction $\sigma_1$ effectif", color='gray')
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.legend()
    # plt.title("Trajectoire du magma depuis le point de départ x=%d" %start_point)
    plt.title(
        "Trajectoire du magma depuis le point de depart x={}".format(start_point[0][0] / norme) + " \n et z={}".format(
            start_point[0][1] / norme) + ". Charge={}".format(P_load) + r" MPa. R ={}".format(
            R) + r" et $\gamma$ = {}".format(G))
    # plt.axis('equal')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()'''







