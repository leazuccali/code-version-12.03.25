# ux ref cm [-0.06732989 -0.06774889 -0.0681716  -0.06859807 -0.06902834 -0.06946246]
# uz ref cm [0.00878161 0.00889611 0.00901218 0.00912984 0.00924913 0.00937006]
# vec concatenate [-0.06732989 -0.06774889 -0.0681716  -0.06859807 -0.06902834 -0.06946246]
# dep ux cm [-0.00131569 -0.00131853 -0.00132127 -0.0013239  -0.00132641 -0.00132881]
# dep uz cm [0.00044973 0.00045394 0.00045816 0.00046238 0.0004666  0.00047081]
# vec dep prediction particule [-0.00131569 -0.00131853 -0.00132127 -0.0013239  -0.00132641 -0.00132881]
# erreur [-0.0660142  -0.06643035 -0.06685033 -0.06727417 -0.06770193 -0.06813365]
# erreur * mat cov [-6.60142018 -6.64303548 -6.68503302 -6.72741733 -6.77019304 -6.81336485]
# erreur * mat_cov * erreur.T 2375.5817560454416
# vraisemblance : -0.5* ... 0.0

# import numpy as np
#
# Nb_points_x = 6
# incertitude_ux = 0.1
# incertitude_uz = 1
# diago = [incertitude_ux]*Nb_points_x + [incertitude_uz]*Nb_points_x
# mat_coefs = np.diag(diago)
# mat_cov = np.dot(mat_coefs, mat_coefs)
# #print(mat_cov)
# cov_inv = np.linalg.inv(mat_cov)  # Inverse de la matrice de covariance
# cov_det = np.linalg.det(mat_cov)  # Déterminant pour la normalisation
# print(cov_inv)
#
#
# ux_ref = np.array([-0.06732989,-0.06774889, -0.0681716,  -0.06859807, -0.06902834, -0.06946246])
# uz_ref = np.array([0.00878161, 0.00889611, 0.00901218, 0.00912984, 0.00924913, 0.00937006])
# dep_ux = np.array([-0.00131569, -0.00131853, -0.00132127, -0.0013239,  -0.00132641, -0.00132881])
# dep_uz = np.array([0.00044973, 0.00045394, 0.00045816, 0.00046238, 0.0004666, 0.00047081])
# dep_ref = np.concatenate((ux_ref, ux_ref))
# dep_pred = np.concatenate((dep_ux, dep_uz))
# print('ref', dep_ref)
# print('pred', dep_pred)
#
#
# ## calculs
# #
# print("Erreur")
# erreur = dep_ref - dep_pred
# print('erreur', erreur)
# tmp1 = np.dot(erreur, mat_coefs)
# print("e \sigma = ", tmp1)
# tmp2 = np.dot(tmp1, erreur.T)
# print("e \sigma e^T = ", tmp2)
# res1 = np.exp(-0.5* tmp2)
# print("Exp(-0.5x) : ", res1 )
#
#
# print('###########')
# print("Erreur normalisee")
# erreur_normalisee = (dep_ref - dep_pred)/dep_ref
# print('erreur_normalisee', erreur_normalisee)
# tmp3 = np.dot(erreur_normalisee, mat_coefs)
# print("e \sigma = ", tmp3)
# tmp4 = np.dot(tmp3, erreur_normalisee.T)
# print("e \sigma e^T = ", tmp4)
# res2 = np.exp(-0.5*tmp4)
# print("Exp(-0.5*x) : ", res2)
#
#
# print('###########')
# print("Poids classique")
# vraisemb = np.array([res1, res2])
# poids_classiques = vraisemb/np.sum(vraisemb)
# print("poids_classiques = ", poids_classiques)
#
#
#
#
#
# def logsumexp(x):
#     c = x.max()
#     return c + np.log(np.sum(np.exp(x - c)))
#
#
#
# ############## Calcul de la log vraisemblance
# log_vr1 = np.log(res1)
# print('log_vr1', log_vr1)
# log_vr2 = np.log(res2)
# print('log_vr2', log_vr2)
#
#
# ## Maintenant comment comparer log_vr1 et log_vr2 ???
# vraisb = np.array([log_vr1, log_vr2])
#
#
# print(logsumexp(vraisb))
# print(np.exp(vraisb - logsumexp(vraisb)))
#
#
#
# #####################################################
# #
# v = np.array([1000000000, 6000000, 80])    ## les différentes valeurs d'erreurs
# v2 = np.exp(-0.5*v)                   ## passage à l'exponentielle inverse
# print('v2', v2)
#
# # v3 = np.log(np.exp(-0.5*v))
# # print(v3)
# # print(np.exp(-0.5*80))
# # print(np.exp(-40))
# # print(np.log(np.exp(-40)))
#
#
#
#
# print("nouvel essai")
# # Erreurs pour chaque particule
# #v = np.array([1000000000, 6000000, 80])
# voo2 = np.array([10000, 6000, 800, 800, 900, 500, 500])
# print('v')
# # Calcul des log-vraisemblances pour éviter l'underflow
# log_vraisemblances = -0.5 * voo2
# print('log_vraisemblances', log_vraisemblances)
# # Normalisation avec le trick log-sum-exp
# log_weights = log_vraisemblances - logsumexp(log_vraisemblances)
# print('log_weights', log_weights)
# # Convertir en poids classiques (exponentiation)
# weights = np.exp(log_weights)
#
# print("Poids normalisés 1:", weights)
#
# ###########
# print('###########')
# print("essai 2")
# # Erreurs pour chaque particule
# v = np.array([1000000000, 6000000, 80])
# voo = np.array([10000, 6000, 800, 800, 900, 500, 500])
# # Calcul des log-vraisemblances
# log_vraisemblances = -0.5 * voo
# print('log_vraisemblances', log_vraisemblances)
# # Recentrer en soustrayant le max (astuce pour éviter l'underflow)
# log_vraisemblances -= np.max(log_vraisemblances)
# print('log_vraisemblances', log_vraisemblances)
# # Calcul des poids normalisés
# log_weights = log_vraisemblances - logsumexp(log_vraisemblances)
# print('log_weights', log_weights)
# weights = np.exp(log_weights)
# print("Poids normalisés :", weights)
# print(np.sum(weights))














# print('###########')
# # Erreurs pour chaque particule
# v = np.array([1000000000, 6000000, 80])
# voo = np.array([10000, 6000, 80])
#
# # Ajustement d'échelle : diviser v par sa moyenne ou son écart-type
# v_scaled = voo / np.max(v)  # Normalisation simple
#
# # Calcul des log-vraisemblances
# log_vraisemblances = -0.5 * v_scaled
#
# # Recentrer en soustrayant le max (astuce pour éviter l'underflow)
# log_vraisemblances -= np.max(log_vraisemblances)
#
# # Calcul des poids normalisés
# log_weights = log_vraisemblances - logsumexp(log_vraisemblances)
# weights = np.exp(log_weights)
#
# print("Poids normalisés :", weights)
#
#
#
# import numpy as np
# from scipy.special import logsumexp
#
# # Erreurs initiales
# voo = np.array([10000, 6000, 80])
#
# # Fixer un seuil maximal (par exemple 10^4)
# v_clipped = np.clip(voo, 0, 10000)  # Tronque les valeurs trop grandes
#
# # Calcul des log-vraisemblances
# log_vraisemblances = -0.5 * v_clipped
#
# # Recentrer en soustrayant le max (log-sum-exp trick)
# log_vraisemblances -= np.max(log_vraisemblances)
#
# # Calcul des poids normalisés
# log_weights = log_vraisemblances - logsumexp(log_vraisemblances)
# weights = np.exp(log_weights)
#
# print("Poids normalisés :", weights)














