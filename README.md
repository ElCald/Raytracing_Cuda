# Projet de CHPS0802 (Ray tracing)

<hr></hr>

## Installer ffmpeg (nécessaire au bon fonctionnement du projet)

Sous un système Unix, faites :

```bash
sudo apt install ffmpeg
```

<hr></hr>

## Comment compiler et exécuter le projet

Le projet est composé de deux répertoires à la racine intitulés `Sequential/` et `GPU/`. Comme leurs noms l'indiquent, l'un correspond à la version dite "séquentielle" du ray tracing et l'autre à la version CUDA du ray tracing.

Ces deux versions présentent une architecture similaire. Vous pouvez retrouver ces répertoires à la racine de ces derniers :
- `bash/` (Contient les scripts bash pour exécuter le main ainsi que les tests unitaires)
- `build/` (Nécessaire au bon fonctionnement de CMake. S'il n'existe pas, il sera créé ultérieurement par les fichiers bash)
- `GeometricsObjects/` (Répertoire contenant l'ensemble des fichiers permettant la représentation des formes de notre ray tracing)
- `Geometry/` (Répertoire contenant l'ensemble des fichiers permettant la représentation géométrique de notre ray tracing, c'est-à-dire : points, vecteurs, rayons)
- `src/` (Répertoire contenant le main)
- `tests/` (Répertoire contenant les tests unitaires)
- `Utils/` (Répertoire contenant l'ensemble des fichiers relatifs à la représentation de la scène, c'est-à-dire : lumière, caméra, et la scène elle-même)

Dans `bash/`, en fonction de ce que vous souhaitez exécuter, faites :


```bash
cd bash/
sh main.sh a b c
```
Pour exécuter le main, où `a` représente le nombre de secondes de la simulation, `b` le nombre de fps souhaités et `c` le nombre de tours effectués par les objets de la scène.


```bash
cd bash/
sh tests.sh
```
Pour exécuter les tests unitaires.

Pour consulter la vidéo générée, il suffit d'aller dans `build/` et d'ouvrir le fichier nommé `output.mp4`.
