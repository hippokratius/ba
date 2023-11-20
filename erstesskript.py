import bpy
import bmesh
import datetime
import time
import math
import csv
import pickle

import matplotlib.pyplot as plt

from mathutils import Vector
from bmesh.types import BMVert
from bpy.types import Mesh, MeshPolygon, MeshVertex, Object
from multiprocessing import Pool, cpu_count

# -----------------------------------------------------------------------------
# Schnellzugriffsvariablen
# -----------------------------------------------------------------------------
name = 'Buffer'
namet = 'Cube'
objects = bpy.data.objects
#active_obj = bpy.context.active_object.data
scene = bpy.context.scene
# Pointcloud
pc = objects['test1']
pcMesh = pc.data
# Rathausmodell
rathaus = objects['DEMVAL03000hEgtA']
rathausMesh = rathaus.data
factor = 0.1


# -----------------------------------------------------------------------------
# Funktionen zur Vorbereitung einer neuen Rathaus Datei
# -----------------------------------------------------------------------------

def transformRathausPW():
    """
    Richtet neue Rathauspunktwolke automatisch richtig aus.
    """
    bpy.context.object.location[0] = -25.06
    bpy.context.object.location[1] = -63.25
    bpy.context.object.location[2] = 1.58
    bpy.context.object.rotation_euler[2] = -86.27
    bpy.ops.object.transform_apply()


def genVertInDistanceToObject(pcMesh: Mesh, obj: Object, dist):
    """
    Generator, welcher angibt ob der jeweils den nächsten Punkt einer Punktwolke
    innerhalb einer Distanz zu einem Objekt liegt.
    pcMesh: Mesh der Punktwolke
    obj: Objekt zu dem Distanz gemessen werden soll
    dist: maximale Distanz
    """
    for vert in pcMesh.vertices:
        result, location, normal, findex = obj.closest_point_on_mesh(
                list(vert.co.to_tuple()),
                distance=dist)
        yield result


def clearUpPointCloud(pc: Object, obj: Object, dist=1):
    """
    Entferne überflüssige Punkte aus der Punktwolke.
    """
    if not pc.get('cityJSONType'):
        inDistance = genVertInDistanceToObject(pc.data, obj, dist)
        bm = bmesh.new()
        bm.from_mesh(pc.data)
        bm.verts.ensure_lookup_table()
        for vert in bm.verts:
            b = next(inDistance)
            if not b:
                bm.verts.remove(vert)
        bm.to_mesh(pc.data)
        bm.free()


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------

def vert(x, y, z):
    """
    Erzeugt ein Tuple der gegebenen Koordinaten. Zur besseren Erkennung, dass
    dass es sich um einen Vertex handelt.
    """
    return (x, y, z)


def getColorValueList(vlist, pc: Mesh, value: int, srgb=True):
    """
    Gibt eine Liste mit den Farbwerten der Punkte aus der RGB-Punktwolke zurück.
    vlist: Liste von Vertex-Indizes aus der Punktwolke.
    pc: Punktwolke als Mesh
    value: 0=R-Wert, 1=G-Wert, 2=B-Wert, 3=A/S-Wert
    """
    cplist = []
    for vert in vlist:
        if srgb:
            cplist.append(pc.color_attributes['Col'].data[vert].color_srgb[value])
        else:
            cplist.append(pc.color_attributes['Col'].data[vert].color[value])
    return cplist


def calcNormalColorValues(l):
    """
    Rechnet Blender-Farbwerte in normale Farbwerte um.
    """
    for i in range(len(l)):
        l[i] = l[i] * 255
    return l


def boxPlot(rlist, glist, blist, prefix='', suffix=''):
    """
    Boxplot erstellen
    rlist: sortierte Liste der R-Werte
    glist: sortierte Liste der G-werte
    blist: sortierte Liste der B-Werte
    """
    # Daten vorbereiten
    labels = ['R', 'G', 'B']
    data = [
        calcNormalColorValues(rlist),
        calcNormalColorValues(glist),
        calcNormalColorValues(blist)]
    
    for element in data:
        print(element)
    
    # BoxPlot Objekt erzeugen und mit Daten füllen und beschriften
    fig, ax = plt.subplots()
    bplot = ax.boxplot(data, vert=True, patch_artist=True, labels=labels)
    ax.set_title('Verteilung der gemessenen RGB-Werte')
    
    # Boxen mit Farben füllen
    colors = ['red', 'green', 'blue']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Horizintale Linien einzeichnen
    ax.yaxis.grid(True)
    
    # Plot anzeigen -> funktioniert nicht
    #plt.show()
    
    plt.savefig(f'/home/steffi/Dokumente/Projekte/BA/boxplot/{prefix}_boxplot_{suffix}.png')
    
    
def setColor(objMesh: Mesh, p_index, r, g, b, a = 1):
    """
    Setze Material Farbe eines Polygons.
    """
    mat = objMesh.materials[objMesh.polygons[p_index].material_index]
    principled_BSDF = mat.node_tree.nodes.get('Principled BSDF')
    principled_BSDF.inputs['Base Color'].default_value = (r, g, b, a)


def removeNullValues(l: list):
    """
    Entfernt alles 0-Werte aus einer sortierten Listel.
    """
    i = 0
    for e in l:
        if e == 0:
            i += 1
        else:
            break
    l.reverse()
    for j in range(i):
        l.pop()
    l.reverse()
    return l


def printSelectedFaceID(objMesh):
    current_mode = bpy.context.active_object.mode
    bpy.ops.object.mode_set()   # setzt Object Mode
    for polygon in objMesh.polygons:
        if polygon.select:
            i = polygon.index
            print(f'Ausgewähltes Polygon: {i}')
    bpy.ops.object.mode_set(mode=current_mode)

# -----------------------------------------------------------------------------
# Punkt-Flächen Matching
# -----------------------------------------------------------------------------
class Matcher:
    def __init__(self, pcMesh, obj, dist, matching: dict = {}):
        self.obj = obj
        self.dist = dist
        self.pcMesh = pcMesh
        self.matching = matching
    
    def findMatch(self, vert):
        result, location, normal, findex = self.obj.closest_point_on_mesh(
            list(vert.co.to_tuple()),
            distance=self.dist)
        if result:
            if findex in self.matching:
                self.matching[findex].append(vert.index)
            else:
                self.matching[findex] = [vert.index]


def multiprocessingMatching(pcMesh: Mesh, obj: Object, dist, processes=4):
    p = Pool(processes=processes)
    m = Matcher(pcMesh, obj, dist)
    p.map(m.findMatch, pcMesh.vertices)
    time.sleep(15)
    return m.matching
    

def fastPointSurfaceMatching(pcMesh: Mesh, obj: Object, dist):
    """
    pc: Pointcloud als Mesh
    obj: Objekt, dessen Polygone fürs Matching genutzt werden sollen
    Factor
    dist: int oder float, max. Matching-Abstand zw. Punkt und Polygon
    """
    matching = {}
    for vert in pcMesh.vertices:
        result, location, normal, findex = obj.closest_point_on_mesh(
            list(vert.co.to_tuple()),
            distance=dist)
        if result:
            if findex in matching:
                matching[findex].append(vert.index)
            else:
                matching[findex] = [vert.index]
    return matching


# -----------------------------------------------------------------------------
# Färbungsverfahren
# -----------------------------------------------------------------------------

def setObjectColorsAverage(matching: dict, objMesh: Mesh, pc: Mesh):
    """
    Berechnet und setzt die Farben von Polygonen eines Objektes auf den durchschnittlichen
    Farbwert der dazugehörigen Punkte einer RGB-Punktwolke.
    matching: dict mit Polygonindex als Keys und Liste von Punktindizes als Value
    obj: 
    """
    for polygon, vertexlist in matching.items():
        r, g, b, a = [], [], [], []
        for vertex in vertexlist:
            r.append(pc.color_attributes['Col'].data[vertex].color_srgb[0])
            g.append(pc.color_attributes['Col'].data[vertex].color_srgb[1])
            b.append(pc.color_attributes['Col'].data[vertex].color_srgb[2])
        r = sum(r) / len(r)
        g = sum(g) / len(g)
        b = sum(b) / len(b)
        setColor(objMesh, polygon, r, g, b)


def setObjectColorsWithMedian(matching: dict, objMesh: Mesh, pc: Mesh):
    """
    Setz Farbe auf Meidan einer sortierten Listes der RGB-Werte
    """
    for polygon, vertexlist in matching.items():
        r = getColorValueList(vertexlist, pc, value=0)
        g = getColorValueList(vertexlist, pc, value=1)
        b = getColorValueList(vertexlist, pc, value=2)
        r.sort()
        g.sort()
        b.sort()
        r = removeNullValues(r)
        g = removeNullValues(g)
        b = removeNullValues(b)
        if r != None and g != None and b != None:
            length = len(r)
            if length % 2 != 0:
                # falls Anz. der Farben ungerade, setze Median für Farbwerte auf das
                # mittlere Element der Liste, int() rundet zur Vorkommazahl
                r_median = r[int(length / 2)]
                g_median = g[int(length / 2)]
                b_median = b[int(length / 2)]
            else:
                # falls Anz. der Farben gerade, setze den Median auf den Durchschnitt
                # aus den beiden mittleren Elementen
                r_median = (r[math.floor(length / 2)] + r[math.ceil(length / 2)]) / 2
                g_median = (g[math.floor(length / 2)] + g[math.ceil(length / 2)]) / 2
                b_median = (b[math.floor(length / 2)] + b[math.ceil(length / 2)]) / 2
            # setze Farbwert des Polygons anhand der Median Werte
            setColor(objMesh, polygon, r_median, g_median, b_median)


def setObjectColorsWithBoxPlot(matching: dict, objMesh: Mesh, pcMesh: Mesh):
    for polygon, vertexlist in matching.items():
        r = getColorValueList(vertexlist, pcMesh, value=0)
        g = getColorValueList(vertexlist, pcMesh, value=1)
        b = getColorValueList(vertexlist, pcMesh, value=2)
        length = len(r)
        if length > 7:
            r.sort()
            g.sort()
            b.sort()
            if length % 2 != 0:
                median = int(length / 2) 
                quartilUnten = int(median - (median / 2))
                quartilOben = int(median + (median / 2))
            else:
                median = length / 2
                quartilUnten = int(median - (median / 2))
                quartilOben = int(median + (median / 2) + 1)
            r_sum, g_sum, b_sum = 0, 0, 0
            for i in range(quartilUnten, quartilOben):
                r_sum += r[i]
                g_sum += g[i]
                b_sum += b[i]
            r = r_sum / (quartilOben - quartilUnten)
            g = g_sum / (quartilOben - quartilUnten)
            b = b_sum / (quartilOben - quartilUnten)
        else:
            r = sum(r) / length
            g = sum(g) / length
            b = sum(b) / length
        setColor(objMesh, polygon, r, g, b)

def setObjectColosWithMostCommonValue(matching: dict, objMesh: Mesh, pcMesh: Mesh):
    """
    Setzt Objektfarben auf den häufigsten R-, G-, B-Wert.
    """
    for polygon, vertexlist in matching.items():
        r = getColorValueList(vertexlist, pcMesh, value=0)
        g = getColorValueList(vertexlist, pcMesh, value=1)
        b = getColorValueList(vertexlist, pcMesh, value=2)
        col = [r, g, b]
        anzahl = {}
        for e in col:
            for elem in e:
                if elem in anzahl:
                    anzahl[elem] += 1
                else:
                    anzahl[elem] = 1
            col[e.index] = max(anzahl)
        setColor(objMesh, polygon, col[0], col[1], col[2])
        print(f'most common value: {r}, {g}, {b}')


# -----------------------------------------------------------------------------
# Analysefunktionen
# -----------------------------------------------------------------------------

def getBoxPoltForSelectedPolygons(pcMesh, matching):
    """
    Erstellt für die im Edit Mode ausgewählten Flächen ein BoxPlot, falls ein
    Matching für sie existiert.
    """
    current_mode = bpy.context.active_object.mode
    suffix = 'srgb'
    # setzt Object Mode
    bpy.ops.object.mode_set()
    
    for polygon in obj.data.polygons:
        if polygon.select:
            r = getColorValueList(matching[polygon.index], pcMesh, value=0)
            g = getColorValueList(matching[polygon.index], pcMesh, value=1)
            b = getColorValueList(matching[polygon.index], pcMesh, value=2)
            r.sort()
            g.sort()
            b.sort()
            r = removeNullValues(r)
            g = removeNullValues(g)
            b = removeNullValues(b)
            print(polygon.index)
            print('r = ', r)
            print('g = ', g)
            print('b = ', b)
            boxPlot(r, g, b, prefix=polygon.index, suffix=suffix)
    
    bpy.ops.object.mode_set(mode=current_mode)


def getBoxPlotForAllPolygonsInMatching(pcMesh, matching):
    """
    Erstellt für alle Flächen eines Matching ein BoxPlot.
    """
    for pIndex, vertexlist in matching.items():
        r = getColorValueList(vertexlist, pcMesh, value=0)
        g = getColorValueList(vertexlist, pcMesh, value=1)
        b = getColorValueList(vertexlist, pcMesh, value=2)
        if r == None:
            print('r ist None')
        if g == None:
            print('g ist None')
        if b == None:
            print('b ist None')
        r.sort()
        g.sort()
        b.sort()
        r = removeNullValues(r)
        g = removeNullValues(g)
        b = removeNullValues(b)
        if r != None and g != None and b != None:
            boxPlot(r, g, b, prefix=pIndex)


# -----------------------------------------------------------------------------------
#                                 Hauptprogramm
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# Funktionen zur Vorbereitung eines neuen Rathaus-Projekts
# -----------------------------------------------------------------------------------

# Ausrichten der Rathaus Punktwolke
#transformRathausPW()

# überflüssige Punkte entfernen
#clearUpPointCloud(pc, rathaus, bfactor)


# -----------------------------------------------------------------------------------
# Vorbereitung
# -----------------------------------------------------------------------------------

print("Bereite Berechnungen vor...")

# setzt Object Mode, falls nicht gesetzt.
bpy.ops.object.mode_set()
print("Object Mode gesetzt.")

# Matching von Flächen und Punkten berechnen
print("Berechne Matching...")
#m = fastPointSurfaceMatching(pcMesh=pcMesh, obj=rathaus, dist=factor)
#m = multiprocessingMatching(pcMesh=pcMesh, obj=rathaus, dist=factor)

printSelectedFaceID(rathausMesh)

# -----------------------------------------------------------------------------------
# Färbungsverfahren
# -----------------------------------------------------------------------------------

# Farben mit Durchschnitswerten setzten
#setObjectColorsAverage(m, rathausMesh, pcMesh)

# Farben auf Median der Farbwerte setzen
#setObjectColorsWithMedian(m, rathausMesh, pcMesh)

# Farben auf Durchschnitt des Interquartilsabstand setzen
#setObjectColorsWithBoxPlot(m, rathausMesh, pcMesh)

# Farben auf häufigsten R-, G-, B-Wert setzen.
#setObjectColorWithMostCommonValue(m, rathausMesh, pcMesh)


# -----------------------------------------------------------------------------------
# Aufruf der Analysefunktionen
# -----------------------------------------------------------------------------------

# BoxPlot für ausgewählte Flächen
#getBoxPoltForSelectedPolygons(pcMesh=pcMesh,  matching=m)

# BoxPlot für alle Flächen erzeugen
#getBoxPlotForAllPolygonsInMatching(pcMesh=pcMesh, matching=m)


# -----------------------------------------------------------------------------------
# Matching 'picklen' für externe Bearbeitung
# -----------------------------------------------------------------------------------

# Datei erzeugen und öffnen
#with open(
#    '/home/steffi/Dokumente/Projekte/BA/skriptefuerdieba/pickleddata/matching.pkl',
#    'wb') as output:
    # Matching in Datei schrieben
#    pickle.dump(m, output)
#Datei schließen
#output.close()