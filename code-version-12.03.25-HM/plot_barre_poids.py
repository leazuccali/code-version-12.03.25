import matplotlib.pyplot as plt
import numpy as np


import os
import json
Nb_particules = 100
pas = 130
with open(f"output_sauvegarde_plots/step_{pas}/sauvegarde_donnees_avant_pas_{pas}.json", "r") as f:
    donnes = json.load(f)

print(donnes[0])

vec_vraisemblance_pas = []
for p in range(Nb_particules):
    donnees_p = donnes[p]
    vraisemblance_p = donnees_p["Vraisemblance"]
    vec_vraisemblance_pas.append(vraisemblance_p)
print(vec_vraisemblance_pas)



#Normalisation
# Calcul du poids total
poids_total = np.sum(vec_vraisemblance_pas)

# Normalisation des poids pour que la somme des longueurs soit égale à 1
proportions = vec_vraisemblance_pas / poids_total
print(proportions)
# Initialisation de la figure
fig, ax = plt.subplots(figsize=(20, 3))

# Positionnement des portions sur la droite
position = 0
for i, prop in enumerate(proportions):
    # Affichage des segments
    ax.plot([position, position + prop], [0.5, 0.5], lw=10, label=f"Particule {i + 1}")

    # Affichage du numéro en dessous de la droite
    ax.text(position + prop / 2, 0.3, f"{i + 1}", color='red', ha='center', va='center', fontsize=12)

    # Ajout d'une flèche pour relier le numéro à la portion
    ax.annotate("", xy=(position + prop / 2, 0.5), xytext=(position + prop / 2, 0.3),
                arrowprops=dict(arrowstyle="->", color='black', lw=1))

    position += prop  # Mise à jour de la position pour la prochaine particule

# Ajustement de l'affichage
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')  # Enlever les axes pour ne garder que la droite
plt.title(f"Répartition des poids des particules au pas {pas}")

directory = 'affichage_evolutions'
os.makedirs(directory, exist_ok=True)
plot_filename = os.path.join(directory, f'droite_poids_{pas}.png')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

plt.show()