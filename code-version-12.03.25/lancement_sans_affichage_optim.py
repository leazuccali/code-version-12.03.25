import shutil
from deplacement_pas_i import *
from calcul_vraisemblance_poids import *
from algorithme_SUS import *
from collections import Counter
#from prediction_mille_m import *
from joblib import Parallel, delayed
import os
import datetime
from trajectoire_pas_i import *
# Sauvegarde des données
def sauvegarder_json(filepath, data):
    with open(filepath, 'w') as fichier:
        json.dump(data, fichier, indent=4)


def charger_json(filepath):
    with open(filepath, 'r') as fichier:
        return json.load(fichier)



####### Foncttion joblib pour copier les données d'initialisation dans le dossier step initial
def copier_fichiers_particule(p, i):
    # Copier le fichier paramètre
    src_param = f'data_initialisation/particule_{p}/param_init_particule_{p}.json'
    dst_dir_param = f'output_data_and_figures/step_{i}/particule_{p}'
    dst_param = f'{dst_dir_param}/param_particules_{p}_step_{i}.json'
    os.makedirs(dst_dir_param, exist_ok=True)
    shutil.copy2(src_param, dst_param)

    # Copier le fichier trajectoire
    src_trajectoire = f'data_initialisation/particule_{p}/trajectoire_init_particule_{p}.json'
    dst_dir_trajectoire = f'output_data_and_figures/step_{i}/particule_{p}'
    dst_trajectoire = f'{dst_dir_trajectoire}/trajectoire_particules_{p}_step_{i}.json'
    os.makedirs(dst_dir_trajectoire, exist_ok=True)
    shutil.copy2(src_trajectoire, dst_trajectoire)


# Fonction pour propager une particule
def propager_particule(p, i, step_directory, temps_courant, pas_temps):
    print("Particule", p, "au pas i =", i)

    # Récupération des données stockées dans les fichiers d'initialisation
    particule_directory = os.path.join(step_directory, f'particule_{p}')
    data_filename = os.path.join(particule_directory, f'trajectoire_particules_{p}_step_{i}.json')
    with open(data_filename, 'r') as fichier:
        sauvegarde_trajectoire_p = json.load(fichier)

    vec_Xt_eff_p = sauvegarde_trajectoire_p['X']
    vec_Zt_p = sauvegarde_trajectoire_p['Z']
    vec_temps_eff_p = sauvegarde_trajectoire_p['Temps effectif']
    longueur_p = sauvegarde_trajectoire_p['Longueur remontee']
    ouverture_p = sauvegarde_trajectoire_p['Ouverture']
    vitesse_p = sauvegarde_trajectoire_p['Vitesse']

    # Propagation
    vecXt_p, vecZt_p, trajectoire_x_p, trajectoire_z_p, X_haut_p, Z_haut_p, X_bas_p, Z_bas_p, distance_parcourue_p = une_trajectoire_pas_i(
        vec_Xt_eff_p, vec_Zt_p, vec_temps_eff_p, temps_courant, pas_temps, longueur_p)

    # Construction du dictionnaire pour cette particule
    dico_trajectoire_pas_i = {
        'Numero': p,
        'Trajectoire X': trajectoire_x_p,
        'Trajectoire Z': trajectoire_z_p,
        'Point haut X': X_haut_p,
        'Point bas X': X_bas_p,
        'Point haut Z': Z_haut_p,
        'Point bas Z': Z_bas_p,
        'Memoire X': vecXt_p,
        'Memoire Z': vecZt_p,
    }

    return dico_trajectoire_pas_i, vitesse_p

def copy_files_for_particle(p, i, vec_selection, Nb_propag_init, step_directory, directory_sauvegarde):
    dst_dir = f'output_data_and_figures/step_{i}/particule_{p}'
    os.makedirs(dst_dir, exist_ok=True)

    if i == Nb_propag_init:
        # Cas initial : récupération depuis `data_initialisation`
        src_param = f'data_initialisation/particule_{p}/param_init_particule_{p}.json'
        dst_param = f'{dst_dir}/param_particules_{p}_step_{i}_avant.json'
        shutil.copy2(src_param, dst_param)

        src_traj = f'data_initialisation/particule_{p}/trajectoire_init_particule_{p}.json'
        dst_traj = f'{dst_dir}/trajectoire_particules_{p}_step_{i}_avant.json'
        shutil.copy2(src_traj, dst_traj)
    else:
        # Cas général : récupération depuis le step précédent
        ind_pas = vec_selection.index(i)
        ind_av = ind_pas - 1
        i_av = vec_selection[ind_av]

        src_param = f'output_data_and_figures/step_{i_av}/particule_{p}/param_particules_{p}_step_{i_av}_apres.json'
        dst_param = f'{dst_dir}/param_particules_{p}_step_{i}_avant.json'
        shutil.copy2(src_param, dst_param)

        src_traj = f'output_data_and_figures/step_{i_av}/particule_{p}/trajectoire_particules_{p}_step_{i_av}_apres.json'
        dst_traj = f'{dst_dir}/trajectoire_particules_{p}_step_{i}_avant.json'
        shutil.copy2(src_traj, dst_traj)


def propag_deplac_particule(p, i, step_directory, temps_courant, pas_temps, pas_okada, P_load, xmin, xmax):
    print(f"Pas {i}, phase de sélection pour la particule {p}")
    print(f"Le temps courant de propagation est {temps_courant} secondes.")

    particule_directory = os.path.join(step_directory, f'particule_{p}')

    # Lecture des paramètres de la particule
    data_filename = os.path.join(particule_directory, f'param_particules_{p}_step_{i}_avant.json')
    with open(data_filename, 'r') as fichier:
        sauvegarde_param_p = json.load(fichier)

    x_0_p = sauvegarde_param_p['x_0']
    z_0_p = sauvegarde_param_p['z_0']
    R_0_p = sauvegarde_param_p['R_0']
    G_0_p = sauvegarde_param_p['G_0']
    mu_0_p = sauvegarde_param_p['mu_0']
    E_0_p = sauvegarde_param_p['E_0']
    vol_0_p = sauvegarde_param_p['vol_0']

    # Lecture de la trajectoire
    data_filename = os.path.join(particule_directory, f'trajectoire_particules_{p}_step_{i}_avant.json')
    with open(data_filename, 'r') as fichier:
        sauvegarde_trajectoire_p = json.load(fichier)

    vec_Xt_eff_p = sauvegarde_trajectoire_p['X']
    vec_Zt_p = sauvegarde_trajectoire_p['Z']
    vec_temps_eff_p = sauvegarde_trajectoire_p['Temps effectif']
    longueur_p = sauvegarde_trajectoire_p['Longueur remontee']
    ouverture_p = sauvegarde_trajectoire_p['Ouverture']
    vitesse_p = sauvegarde_trajectoire_p['Vitesse']

    # Calcul du temps final, position finale
    vec_temps_final = vec_temps_eff_p[-1]
    vec_x_final = vec_Xt_eff_p[-1]
    vec_z_final = vec_Zt_p[-1]


    # Calcul des trajectoires
    (vecXt_p, vecZt_p, trajectoire_x_p, trajectoire_z_p,
     X_haut_p, Z_haut_p, X_bas_p, Z_bas_p, distance_parcourue_p) = une_trajectoire_pas_i(
        vec_Xt_eff_p, vec_Zt_p, vec_temps_eff_p, temps_courant, pas_temps, longueur_p)


    is_surface = False
    if Z_haut_p >= -100:
        is_surface = True


    # Calcul des déplacements
    ux_p, uz_p, x_p, mY_p, centre_x_p, centre_z_p, l_okada_p, ouv_okada_p, dip_p, strike_p = une_particule_deplacement_pas_i(
        X_haut_p, Z_haut_p, X_bas_p, Z_bas_p,
        longueur_p, ouverture_p, distance_parcourue_p,
        R_0_p, G_0_p, mu_0_p, E_0_p, vol_0_p,
        P_load, pas_okada, xmin, xmax)

    # Sauvegarde des résultats pour cette particule
    dico_trajectoire_pas_i = {
        'Numero': p,
        'Trajectoire X': trajectoire_x_p,
        'Trajectoire Z': trajectoire_z_p,
        'Point haut X': X_haut_p,
        'Point bas X': X_bas_p,
        'Point haut Z': Z_haut_p,
        'Point bas Z': Z_bas_p,
        'Memoire X': vecXt_p,
        'Memoire Z': vecZt_p,
    }

    dico_deplacements_pas_i = {
        'Numero': p,
        'ux': np.ravel(ux_p).tolist(),
        'uz': np.ravel(uz_p).tolist(),
        'x': x_p.tolist(),
        'mY': np.ravel(mY_p).tolist(),
        'centre_x': centre_x_p,
        'centre_z': centre_z_p,
        'l_okada': l_okada_p,
        'ouv_okada': ouv_okada_p,
        'dip': dip_p,
        'strike': strike_p,
    }

    return {
        'trajectoire': dico_trajectoire_pas_i,
        'deplacements': dico_deplacements_pas_i,
        'temps_final': vec_temps_final,
        'x_final': vec_x_final,
        'z_final': vec_z_final,
        'vitesse': vitesse_p,
        'is_surface': is_surface,
        'param_physiques': {
            'x_0': x_0_p, 'z_0': z_0_p, 'R': R_0_p, 'G': G_0_p, 'mu': mu_0_p, 'E': E_0_p, 'vol': vol_0_p
        }
    }

# Fonction pour traiter les déplacements pour une particule
def traiter_deplacement(p, sauvegarde_deplacements):
    dep_x = np.ravel(sauvegarde_deplacements[p]['ux'])
    dep_z = np.ravel(sauvegarde_deplacements[p]['uz'])
    dep_mY = np.ravel(sauvegarde_deplacements[p]['mY'])

    centre_x = sauvegarde_deplacements[p]['centre_x']
    centre_z = sauvegarde_deplacements[p]['centre_z']
    l_okadas = sauvegarde_deplacements[p]['l_okada']
    ouv_okadas = sauvegarde_deplacements[p]['ouv_okada']
    dip_ps = sauvegarde_deplacements[p]['dip']
    strikes = sauvegarde_deplacements[p]['strike']

    return dep_x, dep_z, dep_mY, centre_x, centre_z, l_okadas, ouv_okadas, dip_ps, strikes


def donnes_particule_avant_selec(p, i, step_directory, liste_vraisemblance_i):
    """
    Fonction pour traiter une particule et retourner son dictionnaire.
    """
    particule_directory = os.path.join(step_directory, f'particule_{p}')
    data_filename = os.path.join(particule_directory, f'param_particules_{p}_step_{i}_avant.json')

    with open(data_filename, 'r') as fichier:
        sauvegarde_param_p = json.load(fichier)

    dico_donnes_p = {
        'Numero': p,
        'Position x': sauvegarde_param_p['x_0'],
        'Position z': sauvegarde_param_p['z_0'],
        'R_0': sauvegarde_param_p['R_0'],
        'G_0': sauvegarde_param_p['G_0'],
        'mu_0': sauvegarde_param_p['mu_0'],
        'E_0': sauvegarde_param_p['E_0'],
        'vol_0': sauvegarde_param_p['vol_0'],
        'Vraisemblance': liste_vraisemblance_i[p]
    }

    # Ajout des clés rétropropagation si elles existent
    if 'Start x retropropag' in sauvegarde_param_p:
        dico_donnes_p.update({
            'Retro x_0': sauvegarde_param_p['Start x retropropag'],
            'Retro z_0': sauvegarde_param_p['Start z retropropag']
        })

    return dico_donnes_p


def donnees_particule_apres_selec(p, i, step_directory, vec_index_particules):
    """
    Fonction pour traiter une particule et retourner son dictionnaire après traitement.
    """
    particule_directory = os.path.join(step_directory, f'particule_{p}')
    data_filename = os.path.join(particule_directory, f'param_particules_{p}_step_{i}_apres.json')

    with open(data_filename, 'r') as fichier:
        sauvegarde_physiques = json.load(fichier)

    dico_donnes_ap_p = {
        'Numero': p,
        'Position x': sauvegarde_physiques['x_0'],
        'Position z': sauvegarde_physiques['z_0'],
        'R_0': sauvegarde_physiques['R_0'],
        'G_0': sauvegarde_physiques['G_0'],
        'mu_0': sauvegarde_physiques['mu_0'],
        'E_0': sauvegarde_physiques['E_0'],
        'vol_0': sauvegarde_physiques['vol_0'],
        'Nouvel index': vec_index_particules[p]
    }

    # Ajout des clés rétropropagation si elles existent
    if 'Start x retropropag' in sauvegarde_physiques:
        dico_donnes_ap_p.update({
            'Retro x_0': sauvegarde_physiques['Start x retropropag'],
            'Retro z_0': sauvegarde_physiques['Start z retropropag']
        })

    return dico_donnes_ap_p


def propag_deplac_particule_apres(p, i, step_directory, temps_courant, pas_temps, pas_okada, P_load, xmin, xmax):
    print(f"Pas {i}, phase de sélection pour la particule {p}")
    print(f"Le temps courant de propagation est {temps_courant} secondes.")

    particule_directory = os.path.join(step_directory, f'particule_{p}')

    # Lecture des paramètres de la particule
    data_filename = os.path.join(particule_directory, f'param_particules_{p}_step_{i}_apres.json')
    with open(data_filename, 'r') as fichier:
        sauvegarde_param_p = json.load(fichier)

    x_0_p = sauvegarde_param_p['x_0']
    z_0_p = sauvegarde_param_p['z_0']
    R_0_p = sauvegarde_param_p['R_0']
    G_0_p = sauvegarde_param_p['G_0']
    mu_0_p = sauvegarde_param_p['mu_0']
    E_0_p = sauvegarde_param_p['E_0']
    vol_0_p = sauvegarde_param_p['vol_0']

    # Lecture de la trajectoire
    data_filename = os.path.join(particule_directory, f'trajectoire_particules_{p}_step_{i}_apres.json')
    with open(data_filename, 'r') as fichier:
        sauvegarde_trajectoire_p = json.load(fichier)

    vec_Xt_eff_p = sauvegarde_trajectoire_p['X']
    vec_Zt_p = sauvegarde_trajectoire_p['Z']
    vec_temps_eff_p = sauvegarde_trajectoire_p['Temps effectif']
    longueur_p = sauvegarde_trajectoire_p['Longueur remontee']
    ouverture_p = sauvegarde_trajectoire_p['Ouverture']
    vitesse_p = sauvegarde_trajectoire_p['Vitesse']

    # Calcul du temps final, position finale
    vec_temps_final = vec_temps_eff_p[-1]
    vec_x_final = vec_Xt_eff_p[-1]
    vec_z_final = vec_Zt_p[-1]


    # Calcul des trajectoires
    (vecXt_p, vecZt_p, trajectoire_x_p, trajectoire_z_p,
     X_haut_p, Z_haut_p, X_bas_p, Z_bas_p, distance_parcourue_p) = une_trajectoire_pas_i(
        vec_Xt_eff_p, vec_Zt_p, vec_temps_eff_p, temps_courant, pas_temps, longueur_p)

    # Calcul des déplacements
    ux_p, uz_p, x_p, mY_p, centre_x_p, centre_z_p, l_okada_p, ouv_okada_p, dip_p, strike_p = une_particule_deplacement_pas_i(
        X_haut_p, Z_haut_p, X_bas_p, Z_bas_p,
        longueur_p, ouverture_p, distance_parcourue_p,
        R_0_p, G_0_p, mu_0_p, E_0_p, vol_0_p,
        P_load, pas_okada, xmin, xmax)

    # Sauvegarde des résultats pour cette particule
    dico_trajectoire_pas_i = {
        'Numero': p,
        'Trajectoire X': trajectoire_x_p,
        'Trajectoire Z': trajectoire_z_p,
        'Point haut X': X_haut_p,
        'Point bas X': X_bas_p,
        'Point haut Z': Z_haut_p,
        'Point bas Z': Z_bas_p,
        'Memoire X': vecXt_p,
        'Memoire Z': vecZt_p,
    }

    dico_deplacements_pas_i = {
        'Numero': p,
        'ux': np.ravel(ux_p).tolist(),
        'uz': np.ravel(uz_p).tolist(),
        'x': x_p.tolist(),
        'mY': np.ravel(mY_p).tolist(),
        'centre_x': centre_x_p,
        'centre_z': centre_z_p,
        'l_okada': l_okada_p,
        'ouv_okada': ouv_okada_p,
        'dip': dip_p,
        'strike': strike_p,
    }

    return {
        'trajectoire': dico_trajectoire_pas_i,
        'deplacements': dico_deplacements_pas_i,
        'temps_final': vec_temps_final,
        'x_final': vec_x_final,
        'z_final': vec_z_final,
        'vitesse': vitesse_p,
        'param_physiques': {
            'x_0': x_0_p, 'z_0': z_0_p, 'R': R_0_p, 'G': G_0_p, 'mu': mu_0_p, 'E': E_0_p, 'vol': vol_0_p
        }
    }



####################################################################################################################
# FONCTION PRINCIPALE
####################################################################################################################
def lancement_propag_selec(Nb_particules, pas_temps, Nb_propag_init, Nb_pas_max, int_selec, int_aff):

    # 1. Récupérer l'heure de début
    heure_debut = datetime.datetime.now()
    print(f"Début du calcul : {heure_debut.strftime('%Y-%m-%d %H:%M:%S')}")

    ## Paramètres de la grille
    xmin = -30000  # m
    xmax = 30000  # m
    zmin = -15000  # m
    zmax = -1  # m
    pas_trajectoire = 100
    pas_vect = 2000
    P_load = -15000000  # MPa
    rayon_load = 10000  # m
    norme = 1  # Si norme = 1, alors l'unité est le mètre. Si norme = 1000, alors l'unité est le km.
    pas_okada = 100
    ## Paramètres de la particule de référence et ouverture du fichier de reférence
    x_0_ref = 2000
    z_0_ref = -8000
    x_ref = x_0_ref
    z_ref = z_0_ref
    R_ref = 0.5
    G_ref = 0.5
    mu_ref = 100
    E_ref = 5000000000
    vol_ref = 100000000
    vitesse_ref = 0.25



    ## Définition du dossier principal
    directory = "output_data_and_figures"
    os.makedirs(directory, exist_ok=True)


    #directory_sauvegarde = "/home/zuccalil/WS1-NAS/zuccalil/output_sauvegarde_plots_11.03.25-m"
    directory_sauvegarde = "output_sauvegarde_plots"
    os.makedirs(directory, exist_ok=True)


    #directory_etape1000 = "output_step_1000"
    #os.makedirs(directory_etape1000, exist_ok=True)
    #etape_1000m = False

    ###############################
    #### Phase de propagation libre jusqu'à  Nb_propag_init - 1
    ###############################
    i = Nb_propag_init - 1

    #Définition du dossier d'étape
    step_directory = os.path.join(directory, f'step_{i}')
    os.makedirs(step_directory, exist_ok=True)

    print("Pas ", i, "phase de propagation")
    temps_courant = pas_temps * i
    print("Le temps courant de propagation est", temps_courant, "secondes.")

    step_directory_sauvegarde = os.path.join(directory_sauvegarde, f'step_{i}')
    os.makedirs(step_directory_sauvegarde, exist_ok=True)



    ####### Récupération et copie des données de la phase d'initialisation
    #Reference

    ### Copie des données dans le dossier d'étape

    #Reference
    source_dir_ref_init = 'data_initialisation/sauvegarde_trajectoire_ref.json'
    destination_dir_ref_init = os.path.join(step_directory, 'sauvegarde_trajectoire_ref.json')
    shutil.copy2(source_dir_ref_init, destination_dir_ref_init)

    # Parallélisation de la copie pour les particules
    Parallel(n_jobs=-1)(delayed(copier_fichiers_particule)(p, i) for p in range(Nb_particules))


    ######################################################################

    #Propagation de la particule de référence
    data_filename = os.path.join(step_directory, 'sauvegarde_trajectoire_ref.json')
    with open(data_filename, 'r') as fichier:
        sauvegarde_trajectoire_ref = json.load(fichier)

    # Données référence
    vec_Xt_ref = sauvegarde_trajectoire_ref['X']
    vec_Zt_ref = sauvegarde_trajectoire_ref['Z']
    vec_temps_ref = sauvegarde_trajectoire_ref['Temps effectif']
    longueur_ref = sauvegarde_trajectoire_ref['Longueur remontee']
    ouverture_ref = sauvegarde_trajectoire_ref['Ouverture']
    vitesse_ref = sauvegarde_trajectoire_ref['Vitesse']

    # Propagation de la particule de référence
    # Calcul de la trajectoire et des déplacements de référence au pas i
    vecXt_ref, vecZt_ref, trajectoire_x_ref, trajectoire_z_ref, X_haut_ref, Z_haut_ref, X_bas_ref, Z_bas_ref, distance_parcourue_ref = une_trajectoire_pas_i(
        vec_Xt_ref, vec_Zt_ref, vec_temps_ref, temps_courant, pas_temps, longueur_ref)

    # Dictionnaire données pas i particule de référence
    dico_ref_pas_i = {}
    dico_ref_pas_i['Numero'] = 'ref'
    dico_ref_pas_i['Trajectoire X'] = trajectoire_x_ref
    dico_ref_pas_i['Trajectoire Z'] = trajectoire_z_ref
    dico_ref_pas_i['Point haut X'] = X_haut_ref
    dico_ref_pas_i['Point bas X'] = X_bas_ref
    dico_ref_pas_i['Point haut Z'] = Z_haut_ref
    dico_ref_pas_i['Point bas Z'] = Z_bas_ref
    dico_ref_pas_i['Memoire X'] = vecXt_ref
    dico_ref_pas_i['Memoire Z'] = vecZt_ref

    # Création du fichier de sauvegarde de la trajectoire de reference empruntée à l'issue de la phase de propagation libre
    data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_traj_ref_pas_{i}.json')
    with open(data_filename, 'w') as fichier:
        json.dump(dico_ref_pas_i, fichier, indent=4)




    #Propagation des partcules
    #### Vecteur de sauvegarde des des trajectoires au pas i (dernier pas de la propagation libre)

    resultats = Parallel(n_jobs=-1)(
        delayed(propager_particule)(p, i, step_directory, temps_courant, pas_temps) for p in range(Nb_particules))

    sauvegarde_trajectoires_pas_i = []
    sauvegarde_vitesses_pas_i = []
    # Séparation des résultats en trajectoires et vitesses
    sauvegarde_trajectoires_pas_i, sauvegarde_vitesses_pas_i = zip(*resultats)

    # Convertir en listes (car zip renvoie des tuples)
    sauvegarde_trajectoires_pas_i = list(sauvegarde_trajectoires_pas_i)
    sauvegarde_vitesses_pas_i = list(sauvegarde_vitesses_pas_i)

    # Sauvegarde des trajectoires dans un fichier .json
    data_filename_traj = os.path.join(step_directory, f'sauvegarde_traj_avant_pas_{i}.json')
    with open(data_filename_traj, 'w', encoding='utf-8') as fichier:  # Ajout de l'encodage
        json.dump(sauvegarde_trajectoires_pas_i, fichier, indent=4)

    # Sauvegarde des vitesses dans un fichier .json
    dico_vitesses_pas_i = {
        'Vitesse ref': vitesse_ref,
        'Vitesses particules': sauvegarde_vitesses_pas_i
    }

    data_filename_vit = os.path.join(step_directory_sauvegarde, f'sauvegarde_vitesses_pas_{i}.json')
    with open(data_filename_vit, 'w', encoding='utf-8') as fichier:  # Ajout de l'encodage
        json.dump(dico_vitesses_pas_i, fichier, indent=4)








    ##############################################
    ################## SELECTIONS
    ##############################################
    # Définition du vecteur de sélection
    i = Nb_propag_init
    i_max = Nb_pas_max + 1

    vec_selection = list(range(i, i_max, int_selec))
    print("Vecteur pas de selection", vec_selection)

    #step_directory_sauvegarde = os.path.join(directory_sauvegarde, f'step_{i}')
    #os.makedirs(step_directory_sauvegarde, exist_ok=True)




    vec_pas_selection = []

    for i in vec_selection:

        # Definition du temps
        temps_courant = i * pas_temps
        vec_pas_selection.append(i)
        print("i = ", i, "Temps courant :", temps_courant)
        # Définition du répertoire de direction
        step_directory = os.path.join(directory, f'step_{i}')
        os.makedirs(step_directory, exist_ok=True)
        print("Pas de selection", i)

        step_directory_sauvegarde = os.path.join(directory_sauvegarde, f'step_{i}')
        os.makedirs(step_directory_sauvegarde, exist_ok=True)


        # Récupération et copie des données de référence
        source_dir_ref_init = 'data_initialisation/sauvegarde_trajectoire_ref.json'
        destination_dir_ref_init = os.path.join(step_directory, 'sauvegarde_trajectoire_ref.json')
        shutil.copy2(source_dir_ref_init, destination_dir_ref_init)



        # Récupération et copie des données de la phase précédente  (job
        Parallel(n_jobs=-1)(
            delayed(copy_files_for_particle)(
                p, i, vec_selection, Nb_propag_init, step_directory, directory_sauvegarde
            )
            for p in range(Nb_particules)
        )


        if i % int_selec == 0:
            ############### PROPAGATION ET RECUPERATION DES DONNEES
            print("Pas", i, "phase de selection pour la particule de reference")


            ###########################################################
            ######## Récupération des données trajectoires de référence
            ###########################################################
            data_filename = os.path.join(step_directory, 'sauvegarde_trajectoire_ref.json')
            with open(data_filename, 'r') as fichier:
                sauvegarde_trajectoire_ref = json.load(fichier)


            vec_Xt_ref = sauvegarde_trajectoire_ref['X']
            vec_Zt_ref = sauvegarde_trajectoire_ref['Z']
            vec_temps_ref = sauvegarde_trajectoire_ref['Temps effectif']
            longueur_ref = sauvegarde_trajectoire_ref['Longueur remontee']
            ouverture_ref = sauvegarde_trajectoire_ref['Ouverture']
            vitesse_ref = sauvegarde_trajectoire_ref['Vitesse']

            #tems_final_ref = vec_temps_ref[-1]

            # Calcul de la trajectoire et des déplacements de référence au pas i
            vecXt_ref, vecZt_ref, trajectoire_x_ref, trajectoire_z_ref, X_haut_ref, Z_haut_ref, X_bas_ref, Z_bas_ref, distance_parcourue_ref = une_trajectoire_pas_i(
                vec_Xt_ref, vec_Zt_ref, vec_temps_ref, temps_courant, pas_temps, longueur_ref)

            ux_ref, uz_ref, x_ref, mY_ref, centre_x_ref, centre_z_ref, l_okada_ref, ouv_okada_ref, dip_ref, strike_ref = une_particule_deplacement_pas_i(
                X_haut_ref, Z_haut_ref, X_bas_ref, Z_bas_ref,
                longueur_ref, ouverture_ref, distance_parcourue_ref,
                R_ref, G_ref, mu_ref, E_ref, vol_ref,
                P_load, pas_okada, xmin, xmax)

            dico_ref_pas_i = {}
            dico_ref_pas_i['Numero'] = 'ref'
            dico_ref_pas_i['Trajectoire X'] = trajectoire_x_ref
            dico_ref_pas_i['Trajectoire Z'] = trajectoire_z_ref
            dico_ref_pas_i['Point haut X'] = X_haut_ref
            dico_ref_pas_i['Point bas X'] = X_bas_ref
            dico_ref_pas_i['Point haut Z'] = Z_haut_ref
            dico_ref_pas_i['Point bas Z'] = Z_bas_ref
            dico_ref_pas_i['Memoire X'] = vecXt_ref
            dico_ref_pas_i['Memoire Z'] = vecZt_ref

            # if not etape_1000m and Z_haut_ref > -1000:
            #     data_filename = os.path.join(directory_etape1000, 'num_step_1000m.txt')
            #     with open(data_filename, 'w') as fichier:
            #         fichier.write(f"Le front atteint {Z_haut_ref} au pas {i}")
            #
            #     etape_1000m = True




            dico_deplacements_ref_pas_i = {}
            dico_deplacements_ref_pas_i['Numero'] = 'Ref'

            ux_ref_ligne = np.ravel(ux_ref)
            dico_deplacements_ref_pas_i['ux'] = ux_ref_ligne.tolist()

            uz_ref_ligne = np.ravel(uz_ref)
            dico_deplacements_ref_pas_i['uz'] = uz_ref_ligne.tolist()

            dico_deplacements_ref_pas_i['x'] = x_ref.tolist()
            mY_ref_ligne = np.ravel(mY_ref)
            dico_deplacements_ref_pas_i['mY'] = mY_ref_ligne.tolist()

            dico_deplacements_ref_pas_i['centre_x'] = centre_x_ref
            dico_deplacements_ref_pas_i['centre_z'] = centre_z_ref
            dico_deplacements_ref_pas_i['l_okada'] = l_okada_ref
            dico_deplacements_ref_pas_i['ouv_okada'] = ouv_okada_ref
            dico_deplacements_ref_pas_i['dip'] = dip_ref
            dico_deplacements_ref_pas_i['strike'] = strike_ref






            data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_traj_ref_pas_{i}.json')
            with open(data_filename, 'w') as fichier:
                json.dump(dico_ref_pas_i, fichier, indent=4)

            data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_deplacements_ref_pas_{i}.json')
            with open(data_filename, 'w') as fichier:
                json.dump(dico_deplacements_ref_pas_i, fichier, indent=4)








            ###########################################################################################################
            # Propagation et déplacements des particules
            ###########################################################################################################

            # Parallélisation du processus pour chaque particule
            results = Parallel(n_jobs=-1)(delayed(propag_deplac_particule)(
                p, i, step_directory, temps_courant, pas_temps, pas_okada, P_load, xmin, xmax
            ) for p in range(Nb_particules))

            # Consolidation des résultats
            sauvegarde_trajectoires_pas_i = [r['trajectoire'] for r in results]
            sauvegarde_deplacements_i = [r['deplacements'] for r in results]
            vec_temps_final = [r['temps_final'] for r in results]
            vec_x_final = [r['x_final'] for r in results]
            vec_z_final = [r['z_final'] for r in results]
            #vec_x_1000 = [r['x_1000'] for r in results]
            #vec_temps_1000 = [r['temps_1000'] for r in results]
            sauvegarde_vitesses_pas_i_avant = [r['vitesse'] for r in results]
            vec_index_p_surface = [r['trajectoire']['Numero'] for r in results if r['is_surface']]

            # Paramètres physiques
            sauvegarde_x_i = [r['param_physiques']['x_0'] for r in results]
            sauvegarde_z_i = [r['param_physiques']['z_0'] for r in results]
            sauvegarde_R_i = [r['param_physiques']['R'] for r in results]
            sauvegarde_G_i = [r['param_physiques']['G'] for r in results]
            sauvegarde_mu_i = [r['param_physiques']['mu'] for r in results]
            sauvegarde_E_i = [r['param_physiques']['E'] for r in results]
            sauvegarde_vol_i = [r['param_physiques']['vol'] for r in results]

            # Sauvegarde des données

            sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_traj_avant_pas_{i}.json'),
                             sauvegarde_trajectoires_pas_i)
            sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_deplacements_avant_pas_{i}.json'),
                             sauvegarde_deplacements_i)

            dico_predictions_avant = {
                'Location': vec_x_final,
                'Temps': vec_temps_final,
            }
            sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_predictions_avant_pas_{i}.json'),
                             dico_predictions_avant)

            dico_vitesses_pas_i_avant = {
                'Vitesse ref': vitesse_ref,
                'Vitesses particules': sauvegarde_vitesses_pas_i_avant,
            }
            sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_vitesses_pas_avant_{i}.json'),
                             dico_vitesses_pas_i_avant)

            dico_parametres_physiques_avant = {
                'X_0': sauvegarde_x_i, 'Z_0': sauvegarde_z_i, 'R': sauvegarde_R_i,
                'G': sauvegarde_G_i, 'mu': sauvegarde_mu_i, 'E': sauvegarde_E_i, 'vol': sauvegarde_vol_i,
            }
            sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_parametres_physiques_avant_{i}.json'),
                             dico_parametres_physiques_avant)





            ####################################################
            # Calcul des vraisemblances
            ####################################################

            #######Récupération des déplacements pour calculer les vraisemblances
            data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_deplacements_ref_pas_{i}.json')
            with open(data_filename, 'r') as fichier:
                sauvegarde_deplacements_ref = json.load(fichier)

            data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_deplacements_avant_pas_{i}.json')
            with open(data_filename, 'r') as fichier:
                sauvegarde_deplacements = json.load(fichier)

            # Traitement en parallèle
            results = Parallel(n_jobs=-1)(
                delayed(traiter_deplacement)(p, sauvegarde_deplacements) for p in range(Nb_particules)
            )

            # Séparation des résultats en listes
            (deplacements_ux_i, deplacements_uz_i, deplacements_mY_i,
             point_centraux_x_i, point_centraux_z_i,
             longueurs_okada_i, ouvertures_okada_i, dips_i, strikes_i) = map(list, zip(*results))




            # Pour l'instant on prend les valeurs des erreurs


            #erreur_ux = 0.01
            #erreur_uz = 0.1

            #########################
            erreur_ux = 0.1   ## cm
            erreur_uz = 1   # cm
            #########################

            incertitudes = np.array([erreur_ux, erreur_uz])      # plus de précision pour la composante horizontale donc comptera plus

            ux_p_ref = sauvegarde_deplacements_ref['ux']
            uz_p_ref = sauvegarde_deplacements_ref['uz']

            ## Insertion des données dans les fonctions poids et vraisemblance

            vecteur_vraisemblance_i, vecteur_poids_i = fonction_poids(Nb_particules, ux_p_ref, uz_p_ref, deplacements_ux_i, deplacements_uz_i, deplacements_mY_i, incertitudes, vec_index_p_surface)
            print("#####vec vraisemblance i", vecteur_vraisemblance_i, len(vecteur_vraisemblance_i))
            print("#####vec poids i", vecteur_poids_i)

            sauvegarde_donnes_avant_i = Parallel(n_jobs=-1)(
                delayed(donnes_particule_avant_selec)(p, i, step_directory, vecteur_vraisemblance_i) for p in range(Nb_particules))

            # Écriture des données dans un fichier JSON
            data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_donnees_avant_pas_{i}.json')
            with open(data_filename, 'w') as fichier:
                json.dump(sauvegarde_donnes_avant_i, fichier, indent=4)












        ################ Etape de selection
        vec_index_particules = fonction_selection_SUS(Nb_particules, vecteur_poids_i, vec_index_p_surface)

        points_centraux_x_i_re, points_centraux_z_i_re, longueurs_okada_i_re, ouvertures_okada_i_re, dips_i_re, strikes_i_re = resampling_SUS_bruit(Nb_particules, vec_index_particules,
                                                                                                    point_centraux_x_i, point_centraux_z_i, longueurs_okada_i, ouvertures_okada_i, dips_i, strikes_i)



        sauv_parametres, sauv_trajectoires = okada_vers_physique_SUS(Nb_particules, vec_index_particules, points_centraux_x_i_re,
                        points_centraux_z_i_re, longueurs_okada_i_re, ouvertures_okada_i_re,
                        dips_i_re, strikes_i_re, temps_courant, i)




        # Utilisation de Joblib pour paralléliser le traitement des particules
        sauvegarde_donnes_apres_i = Parallel(n_jobs=-1)(
            delayed(donnees_particule_apres_selec)(p, i, step_directory, vec_index_particules) for p in range(Nb_particules))

        # Écriture des données dans un fichier JSON
        data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_donnees_apres_pas_{i}.json')
        with open(data_filename, 'w') as fichier:
            json.dump(sauvegarde_donnes_apres_i, fichier, indent=4)



        dico_donnees_reechantillonnage = {}

        #dico_donnees_reechantillonnage['Index particule de reech'] = index_p_ref.tolist()

        dico_donnees_reechantillonnage['Nouvel index'] = vec_index_particules  # Pour SUS

        liste_bool_reech = [val == i for i, val in enumerate(vec_index_particules)]
        dico_donnees_reechantillonnage['Valeur bool'] = liste_bool_reech

        count = Counter(vec_index_particules)
        num_duplicates = sum(1 for elem in count.values() if elem > 1)
        dico_donnees_reechantillonnage['Nb de particules reech'] = Nb_particules - num_duplicates

        dico_donnees_reechantillonnage['Surface'] = vec_index_p_surface
        dico_donnees_reechantillonnage['Nb Surface'] = len(vec_index_p_surface)



        data_filename = os.path.join(step_directory_sauvegarde, f'sauvegarde_donnees_selection_{i}.json')
        with open(data_filename, 'w') as fichier:
            json.dump(dico_donnees_reechantillonnage, fichier, indent=4)








        ############################################## Propagation des nouvelles particules


        # Parallélisation du processus pour chaque particule
        results = Parallel(n_jobs=-1)(delayed(propag_deplac_particule_apres)(
                p, i, step_directory, temps_courant, pas_temps, pas_okada, P_load, xmin, xmax
            ) for p in range(Nb_particules))

        # Consolidation des résultats
        sauvegarde_trajectoires_apres_pas_i = [r['trajectoire'] for r in results]
        sauvegarde_deplacements_apres_pas_i = [r['deplacements'] for r in results]
        vec_temps_final_apres = [r['temps_final'] for r in results]
        vec_x_final_apres = [r['x_final'] for r in results]
        vec_z_final_apres = [r['z_final'] for r in results]
        #vec_x_1000_apres = [r['x_1000'] for r in results]
        #vec_temps_1000_apres = [r['temps_1000'] for r in results]
        sauvegarde_vitesses_pas_i_apres = [r['vitesse'] for r in results]

        # Paramètres physiques
        sauvegarde_x_i_ap = [r['param_physiques']['x_0'] for r in results]
        sauvegarde_z_i_ap = [r['param_physiques']['z_0'] for r in results]
        sauvegarde_R_i_ap = [r['param_physiques']['R'] for r in results]
        sauvegarde_G_i_ap = [r['param_physiques']['G'] for r in results]
        sauvegarde_mu_i_ap = [r['param_physiques']['mu'] for r in results]
        sauvegarde_E_i_ap = [r['param_physiques']['E'] for r in results]
        sauvegarde_vol_i_ap = [r['param_physiques']['vol'] for r in results]

        sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_traj_apres_pas_{i}.json'),
                         sauvegarde_trajectoires_apres_pas_i)
        sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_deplacements_apres_pas_{i}.json'),
                         sauvegarde_deplacements_apres_pas_i)

        dico_predictions_apres = {
            'Location': vec_x_final_apres,
            'Temps': vec_temps_final_apres,
        }
        sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_predictions_apres_pas_{i}.json'),
                         dico_predictions_apres)

        dico_vitesses_pas_i_apres = {
            'Vitesse ref': vitesse_ref,
            'Vitesses particules': sauvegarde_vitesses_pas_i_apres,
        }
        sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_vitesses_pas_apres_{i}.json'),
                         dico_vitesses_pas_i_apres)

        dico_parametres_physiques_apres = {
            'X_0': sauvegarde_x_i_ap, 'Z_0': sauvegarde_z_i_ap, 'R': sauvegarde_R_i_ap,
            'G': sauvegarde_G_i_ap, 'mu': sauvegarde_mu_i_ap, 'E': sauvegarde_E_i_ap, 'vol': sauvegarde_vol_i_ap,
        }
        sauvegarder_json(os.path.join(step_directory_sauvegarde, f'sauvegarde_parametres_physiques_apres_{i}.json'),
                         dico_parametres_physiques_apres)

        # 2. Fin du calcul
        heure_fin = datetime.datetime.now()
        print(f"Fin du calcul : {heure_fin.strftime('%Y-%m-%d %H:%M:%S')}")

        # 3. Calcul de la durée du calcul
        duree_calcul = heure_fin - heure_debut
        print(f"Durée totale du calcul : {duree_calcul}")



if __name__ == '__main__':
    lancement_propag_selec(100, 60, 60, 550, 5, 5)    #60   540  60   60


