import numpy as np
#from pymc3 import logsumexp
from scipy.special import logsumexp

def calcul_vraisemblances(Nb_particules, ux_ref, uz_ref, deplacements_ux, deplacements_uz, deplacements_mY, incertitudes, vec_index_p_surface):



    vraisemblances_particules = []
    #erreurs_particules = []
    Nb_points_x = len(ux_ref)
    Nb_points_z = len(uz_ref)

    print("nb deplacements ux", len(deplacements_ux))
    print("nb deplacements uz", len(deplacements_uz))


    #### Construction de la matrice de covariance
    erreur_ux = incertitudes[0]
    erreur_uz = incertitudes[1]

    diago = [erreur_ux]*Nb_points_x + [erreur_uz]*Nb_points_z
    mat_coefs = np.diag(diago)
    mat_cov = np.dot(mat_coefs, mat_coefs)
    print(mat_cov)

    cov_inv = np.linalg.inv(mat_cov)  # Inverse de la matrice de covariance
    cov_det = np.linalg.det(mat_cov)  # Déterminant pour la normalisation




    #coefficient = round(1 / ((2 * np.pi) ** (Nb_points) * cov_det ** (Nb_points / 2)), 5)
    #coefficient = 1 / (np.sqrt((2* np.pi)**2  * cov_det))
    for p in range(Nb_particules):
        if p in vec_index_p_surface:
            vraisemb = 0
            #vraisemb = np.array([0])
            vraisemblances_particules.append(vraisemb)
        else:
            ind_0 = np.nonzero(deplacements_mY[p] == 0)
            dep_x = deplacements_ux[p]
            dep_x_p = dep_x[ind_0]
            dep_z = deplacements_uz[p]
            dep_z_p = dep_z[ind_0]


            #Gaussienne normalisee carree
            ux_ref_ar = np.array(ux_ref)
            #print("ux ref m", ux_ref[:6])
            print("ux ref cm", ux_ref_ar[:6])
            uz_ref_ar = np.array(uz_ref)
            #print("uz ref m", uz_ref[:6])
            print("uz ref cm", uz_ref_ar[:6])
            vec_dep_ref = np.concatenate((ux_ref_ar, uz_ref_ar))
            print("vec concatenate", vec_dep_ref[:6])

            dep_x_ar = np.array(dep_x_p)
            print("dep ux m", dep_x_p[:6])
            print("dep ux cm", dep_x_ar[:6])
            dep_z_ar = np.array(dep_z_p)
            print("dep uz cm", dep_z_ar[:6])
            vec_dep_pred = np.concatenate((dep_x_ar, dep_z_ar))
            print("vec dep prediction particule", vec_dep_pred[:6])
            erreur = vec_dep_ref - vec_dep_pred
            print("erreur", erreur[:6])

            erreur_ponderee = np.dot(erreur, cov_inv)
            print('erreur * mat cov', erreur_ponderee[:6])
            erreur_carree = np.dot(erreur_ponderee, erreur.T)
            print('erreur * mat_cov * erreur.T', erreur_carree)
            #vraisemblance = np.exp(-0.5 * erreur_carree)
            #print('vraisemblance : -0.5* ...', vraisemblance)
            #############
            vraisemblance = np.exp(-0.5 * erreur_carree)
            vraisemblances_particules.append(vraisemblance)


    print("Le vecteur des vraisemblances des particules est ", vraisemblances_particules)

    return vraisemblances_particules



def calcul_poids(Nb_particules, vraisemblance_particules, vec_index_p_surface):
    somme_totale = np.sum(vraisemblance_particules)

    #somme_particules_en_jeu = somme_totale - len(vec_index_p_surface)


    poids_particules = []
    for p in range(Nb_particules):
        poids_p = vraisemblance_particules[p] / somme_totale
        poids_particules.append(poids_p)

    print("Somme du vec poids particules", sum(poids_particules))

    return poids_particules


def fonction_poids(Nb_particules, ux_ref, uz_ref, deplacements_ux, deplacements_uz, deplacements_mY, matrice_covariance, vec_index_p_surface):

    vec_vraisemblances = calcul_vraisemblances(Nb_particules, ux_ref, uz_ref, deplacements_ux, deplacements_uz, deplacements_mY, matrice_covariance, vec_index_p_surface)
    vecteur_poids = calcul_poids(Nb_particules, vec_vraisemblances, vec_index_p_surface)

    return vec_vraisemblances, vecteur_poids








































# def calcul_vraisemblances(Nb_particules, ux_ref, uz_ref, deplacements_ux, deplacements_uz, deplacements_mY, matrice_covariance, vec_index_p_surface):
#
#     vraisemblances_particules = []
#     Nb_points = len(ux_ref)
#
#     for p in range(Nb_particules):
#         ind_0 = np.nonzero(deplacements_mY[p] == 0)
#         dep_x = deplacements_ux[p]
#         dep_x_p = dep_x[ind_0]
#         dep_z = deplacements_uz[p]
#         dep_z_p = dep_z[ind_0]
#
#         vraisemb = 0
#         for i in range(Nb_points):
#             mat_dep_points = np.array([dep_x_p[i], dep_z_p[i]])
#             #print("mat dep_points", mat_dep_points)
#             mat_dep_ref = np.array([ux_ref[i], uz_ref[i]])
#             #print("mat_dep_ref", mat_dep_ref)
#
#             diff = mat_dep_points - mat_dep_ref
#             diff_transpose = [[diff[0]], [diff[1]]]
#             #print("diff", diff)
#             #print(diff_transpose)
#
#             vraisemb_point = np.dot(diff, matrice_covariance)
#             #print("vraisemb point", vraisemb_point)
#             vraisemb_point2 = np.dot(vraisemb_point, diff_transpose)
#             #print("vraisemb 2", vraisemb_point2)
#             #vraisemb_point_z = np.transpose((dep_z_p[i] - uz_ref[i])) * matrice_covariance * (dep_z_p[i] - uz_ref[i])
#             vraisemb = vraisemb + vraisemb_point2
#             #vraisemb_exp = np.exp(- vraisemb)
#             #print("vraisemb finale", vraisemb)
#
#
#         #vraisemb = np.transpose((dep_x_p - ux_ref)) * matrice_covariance * (dep_x_p - ux_ref)
#         vraisemb_exp = np.exp(-vraisemb)
#         vraisemblances_particules.append(vraisemb_exp)
#
#     print("Le vecteur des vraisemblances des particules est ", vraisemblances_particules)
#
#     return vraisemblances_particules
#
#
# def calcul_poids(Nb_particules, vraisemblance_particules):
#     somme_totale = np.sum(vraisemblance_particules)
#
#
#     poids_particules = []
#     for p in range(Nb_particules):
#         poids_p = vraisemblance_particules[p] / somme_totale
#         poids_particules.append(poids_p)
#
#     return poids_particules
#
#
# def fonction_poids(Nb_particules, ux_ref, uz_ref, deplacements_ux, deplacements_uz, deplacements_mY, matrice_covariance):
#
#     vecteur_vraisemblance = calcul_vraisemblances(Nb_particules, ux_ref, uz_ref, deplacements_ux, deplacements_uz, deplacements_mY, matrice_covariance)
#
#     vecteur_poids = calcul_poids(Nb_particules, vecteur_vraisemblance)
#
#     return vecteur_vraisemblance, vecteur_poids