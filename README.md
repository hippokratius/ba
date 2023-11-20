# Git-Repository zur Bachelorarbeit "Datenschutzkonforme Texturierung von 3D Stadtmodellen durch RGB-Punktwolken"

In diesem Repository sind alles Skripte, welche im Zusammenhang mit meiner BA entstanden sind festgehalten. Da die 
dazugehörigen Punktwolken Dateien zu groß für das Repository sind, liegen sie in der Nextcloud der Informatik der 
Universität Rostock (https://nextcloud.informatik.uni-rostock.de/s/dpRiBkeasT3xsgm). Es wird empfohlen nach dem 
Klonen des Repositorys die Ordner `ply`, `pickleddata` und `las` zu erstellen und die Inhalte der Nextcloud 
entsprechend der Dateien in `ply` bzw. `las` abzulegen.

# Vorbereitung
Um die Arbeit zu reproduzieren werden folgende weitere Tools benötigt:
- Blender in Version 3.5.1 (Version 3.6 funktioniert auch, jedoch muss dann die Versionsnr. in der `__init__.py` 
  angepasst werden.)
- CityJSONEditor (https://github.com/rostock/CityJSONEditor) in Version 2.0, falls noch nicht veröffentlicht kann 
  der `rework`-Branch genutzt werden.
- (Pointcloud Visualizer kurz PCV, wird für die Ausrichtung der Punktwolke und für das Umwandeln von `.las` in `.ply` 
  genutzt. Die umgewandelten Dateien befinden sich aber in der Nextcloud und die Ausrichtung weiter unten in der 
  Dokumentation.)

Für die Python Dateien wurden folgende Bibliotheken verwendet (siehe `requirements.txt`)
- numpy
- pandas
- matplotlib
- laspy
- pyntcloud
- colormath
- scikit-learn

# Workflow
**Information:** Wenn große Strukturen wie Punktwolken, in mehreren Programmstarts genutzt werden sollen, beitet es 
sich an diese Dateien beim ersten Ausführen einmal zu 'picklen' und als `.pkl` Datei in den Ordner `pickleddata` zu 
speichern. Entsprechende Funktionen sind in der `main.py` zu finden.

