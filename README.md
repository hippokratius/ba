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

1. Zu Beginn muss ein neues Blender Projekt erstellt werden. Im neuen Projekt muss nun die Datei 
   `rathaus_upscaled_colors.ply` über den experimentellen Stanford PLY Import. Als nächstes sollte die 
   CityJSON-Datei `rathaus_mit_semantischen_flaechen.json` in Blender importiert werden (`CityJSONEditor` muss 
   installiert sein, im Import Dialog Häkchen für Texturimport entfernen). Sollte die Punktwolke nicht passend zum 
   CityJSON Objekt ausgerichtet sein, 
   muss folgende 
   Werte 
   für die Punktwolke gesetzt werden. Location: X=-25,06 Y=-63,25 Z=1,58; Rotation: Z=-86,27. Die Werte müssen über 
   `Òbject > Apply > All Transforms` gespeichert werden.
2. Als nächstes muss das Python Skript `erstesskript.py` über den Scripting Bereich von Blender geöffnet werden. Im 
   Skript sollten die Objekt-Bezeichnungen der Importierten Objekte überprüft werden (Variable `pc` und `rathaus`). 
   Danach kann durch ausführen des Skripts ein Matching erzeugt werden. Um das Matching mittels `pickle` zu speichern, 
   müssen die entsprechenden Zeile zum Ende des Skripts auskommentiert werden.
3. Das Python Skript `main.py` in einer IDE öffnen. Zielen zum Importieren der Punktwolke zu Beginn des Abschnittes 
   'Hauptprogramm' auskommentieren (entweder für `.ply` oder `pkl`). Danach den gewünschten Funktionsaufruf für 
   einen Farbberechnungsmethode auskommentieren und das Skript ausführen. Um das `color_matching` auszuwerten, unter 
   den Methoden den entsprechenden Teil auskommentieren. Zum Exportieren des `color_matching` die entsprechenden 
   Zeilen zum Ende des Skripts auskommentieren.
4. Um die Flächen des CityJSON Objektes zu färben müssen die entsprechenden Zeilen am Ende von `erstesskript.py` 
   exportiert werden. Danach das Skript in Blender ausführen.