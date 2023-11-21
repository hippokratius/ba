import math
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyntcloud import PyntCloud
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------------------


def calc_hsl_values(pc: PyntCloud):
    """
    Doch nicht benötigt. Pyntcloud tut ähnliches mit `add_scalar_field('hsv)`
    :param pc:
    :return:
    """
    hsl_colors = {'hue': [], 'saturation': [], 'lightness': []}
    for element in pc.points.get(['red', 'green', 'blue']).values:
        hsl_colors['hue'].append(element[0])
        hsl_colors['saturation'].append(element[1])
        hsl_colors['lightness'].append(element[2])
    df = pd.DataFrame(hsl_colors)
    pc.points.join(df)



# ------------------------------------------------------------------------------
# Funktionen zum Importieren
# ------------------------------------------------------------------------------


def import_pointcloud(file: str):
    """
    Punktwolke aus PLY Datei importieren.
    :return: Punktwolke als PyntCloud
    """
    #print('Lade Punktwolke...')
    start = time.time()
    cloud: PyntCloud = PyntCloud.from_file(file)
    count_of_points = len(cloud.points)
    elapsed_time = time.time() - start
    #print(f'Punktwolke geladen. {count_of_points} in {elapsed_time}s.')
    return cloud


def import_pointcloud_from_pickle(file: str = 'pickledata/pointcloud.pkl'):
    """
    Importiere Punktwolke aus Pickle-Datei.
    :return:
    """
    #print('Pointcloud laden...')
    start = time.time()
    picklefile = open(file, 'rb')
    cloud: PyntCloud = pickle.load(picklefile)
    count_of_points = len(cloud.points)
    picklefile.close()
    elapsed_time = time.time() - start
    #print(f'Punktwolke geladen. {count_of_points} in {elapsed_time}s.')
    return cloud



def import_matching():
    """
    Matching importieren
    :return:
    """
    #print('Matching laden...')
    start = time.time()
    picklefile = open('pickleddata/matching.pkl', 'rb')
    matching = pickle.load(picklefile)
    picklefile.close()
    elapsed_time = time.time() - start
    #print(f'Matching geladen. {elapsed_time}s.')
    return matching


# ------------------------------------------------------------------------------
# Funktionen zum Exportieren
# ------------------------------------------------------------------------------


def export_pointcloud_as_pickle(
        cloud: PyntCloud,
        file: str = 'pickleddata/pointcloud.pkl'):
    """
    Speichert eine Pyntcloud als mit python-pickle ab.
    :param cloud: Punktwolke, welche gespeichert werden soll.
    :param file: Dateiname, ggf. mit Pfad
    :return:
    """
    #print('Punktwolke als Pickle exportieren...')
    start = time.time()
    output = open(file, 'wb')
    pickle.dump(cloud, output)
    output.close()
    elapsed_time = time.time() - start
    #print(f'Punktwolke exportiert. {elapsed_time}s.')


# ------------------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------------------

def giveNextValue(iterable):
    for element in iterable:
        yield element


def upscale_color_r(a: float):

    color = sRGBColor(
        rgb_r=a,
        rgb_g=0,
        rgb_b=0,
        is_upscaled=False
    )
    (r, g, b) = color.get_upscaled_value_tuple()
    return r


def upscale_color_g(a: float):
    color = sRGBColor(
        rgb_r=0,
        rgb_g=a,
        rgb_b=0,
        is_upscaled=False
    )
    (r, g, b) = color.get_upscaled_value_tuple()
    return g


def upscale_color_b(a:float):
    color = sRGBColor(
        rgb_r=0,
        rgb_g=0,
        rgb_b=a,
        is_upscaled=False
    )
    (r, g, b) = color.get_upscaled_value_tuple()
    return b


# ------------------------------------------------------------------------------
# Methoden zur Farbberechnung
# ------------------------------------------------------------------------------


def calc_colors_with_rgb_average(matching: dict, cloud: PyntCloud):
    """
    Berechnet Flächenfarben anhand des Durschnitts der RGB-Werte pro Fläche.
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als Pyntcloud
    :return: color-matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        r_values, g_values, b_values = [], [], []
        for v in vertexlist:
            r_values.append(cloud.points['red'].loc[v])
            g_values.append(cloud.points['green'].loc[v])
            b_values.append(cloud.points['blue'].loc[v])
        r = sum(r_values) / len(r_values)
        g = sum(g_values) / len(g_values)
        b = sum(b_values) / len(b_values)
        color_matching[polygon] = (r, g, b)
    return color_matching


def calc_colors_with_hsv_average(matching: dict,
                                 cloud: PyntCloud,
                                 is_upscaled: bool=False):
    """
    Berechnet Farben in den HSV Farbraum um und berechnet den durchschnittlichen
    Farbwert und rechnet ihn in sRGB um.
    :param matching: Matching
    :param cloud: PyntCloud
    :param is_upscaled: bool, sRGB Werte der Punktwolke als 8-bit int?
    :return: ColorMatching
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        h_values, s_values, v_values = [], [], []
        for v in vertexlist:
            srgb = sRGBColor(
                rgb_r=cloud.points['red'].loc[v],
                rgb_g=cloud.points['green'].loc[v],
                rgb_b=cloud.points['blue'].loc[v],
                is_upscaled=is_upscaled
            )
            hsv: HSVColor = convert_color(srgb, HSVColor)
            h_values.append(hsv.hsv_h)
            s_values.append(hsv.hsv_s)
            v_values.append(hsv.hsv_v)
        hsv = HSVColor(
            hsv_h=sum(h_values) / len(h_values),
            hsv_s=sum(s_values) / len(s_values),
            hsv_v=sum(v_values) / len(v_values)
        )
        srgb: sRGBColor = convert_color(hsv, sRGBColor)
        if is_upscaled:
            color_matching[polygon] = srgb.get_upscaled_value_tuple()
        else:
            color_matching[polygon] = srgb.get_value_tuple()
    return color_matching


def calc_colors_with_boxplot_rgb(matching: dict, cloud: PyntCloud):
    """
    Berechnung von Farbwerten pro Fläche durch den Durchschnitt des
    Interquartilsabstand.
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als PyntCloud
    :return: color matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        r_values, g_values, b_values = [], [], []
        for v in vertexlist:
            r_values.append(cloud.points['red'].loc[v])
            g_values.append(cloud.points['green'].loc[v])
            b_values.append(cloud.points['blue'].loc[v])
        length = len(r_values)
        if length > 15:
            r_values.sort()
            g_values.sort()
            b_values.sort()
            quartilUnten = int(length / 4) - 1
            quartilOben = length - int(length / 4)
            r_sum, g_sum, b_sum = 0, 0, 0
            for i in range(quartilUnten, quartilOben):
                r_sum += r_values[i]
                g_sum += g_values[i]
                b_sum += b_values[i]
            r = r_sum / (quartilOben - quartilUnten - 1)
            g = g_sum / (quartilOben - quartilUnten - 1)
            b = b_sum / (quartilOben - quartilUnten - 1)
        else:
            r = sum(r_values) / length
            g = sum(g_values) / length
            b = sum(b_values) / length
        color_matching[polygon] = (r, g, b)
    return color_matching


def calc_colors_with_boxplot_hsv(matching: dict,
                                  cloud: PyntCloud,
                                  is_upscaled: bool=False):
    """
    Berechnung von Farbwerten pro Fläche durch den Durchschnitt des
    Interquartilsabstand.
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als PyntCloud
    :param is_upscaled: bool, sRGB Werte der Punktwolke als 8-bit int?
    :return: color matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        h_values, s_values, v_values = [], [], []
        for v in vertexlist:
            srgb = sRGBColor(
                rgb_r=cloud.points['red'].loc[v],
                rgb_g=cloud.points['green'].loc[v],
                rgb_b=cloud.points['blue'].loc[v],
                is_upscaled=is_upscaled
            )
            hsv: HSVColor = convert_color(srgb, HSVColor)
            h_values.append(hsv.hsv_h)
            s_values.append(hsv.hsv_s)
            v_values.append(hsv.hsv_v)
        length = len(h_values)
        if length > 15:
            h_values.sort()
            s_values.sort()
            v_values.sort()
            quartilUnten = int(length / 4) - 1
            quartilOben = length - int(length / 4)
            h_sum, s_sum, v_sum = 0, 0, 0
            for i in range(quartilUnten, quartilOben):
                h_sum += h_values[i]
                s_sum += s_values[i]
                v_sum += v_values[i]
            h = h_sum / (quartilOben - quartilUnten - 1)
            s = s_sum / (quartilOben - quartilUnten - 1)
            v = v_sum / (quartilOben - quartilUnten - 1)
        else:
            h = sum(h_values) / length
            s = sum(s_values) / length
            v = sum(v_values) / length
        hsv = HSVColor(hsv_h=h, hsv_s=s, hsv_v=v)
        srgb: sRGBColor = convert_color(hsv, sRGBColor)
        if is_upscaled:
            color_matching[polygon] = srgb.get_upscaled_value_tuple()
        else:
            color_matching[polygon] = srgb.get_value_tuple()
    return color_matching


def calc_colors_with_Median(matching: dict, cloud: PyntCloud):
    """
    Berechnung von Flächenfarben über den Median (im sRGB-Farbraum).
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als PyntCloud
    :return: color matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        r_values, g_values, b_values = [], [], []
        for v in vertexlist:
            r_values.append(cloud.points['red'].loc[v])
            g_values.append(cloud.points['green'].loc[v])
            b_values.append(cloud.points['blue'].loc[v])
        length = len(r_values)
        r_values.sort()
        g_values.sort()
        b_values.sort()
        if length % 2 != 0:
            r_median = r_values[int(length / 2)]
            g_median = g_values[int(length / 2)]
            b_median = b_values[int(length / 2)]
        else:
            r_median = (r_values[math.floor(length / 2)] + r_values[
                math.ceil(length / 2)]) / 2
            g_median = (g_values[math.floor(length / 2)] + g_values[
                math.ceil(length / 2)]) / 2
            b_median = (b_values[math.floor(length / 2)] + b_values[
                math.ceil(length / 2)]) / 2
        color_matching[polygon] = (r_median, g_median, b_median)
    return color_matching


def calc_colors_with_Median_HSV(matching: dict,
                                cloud: PyntCloud,
                                is_upscaled: bool=False):
    """
    Berechnung von Flächenfarben über den Median (im sRGB-Farbraum).
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als PyntCloud
    :param is_upscaled: bool, sRGB Werte der Punktwolke als 8-bit int?
    :return: color matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        h_values, s_values, v_values = [], [], []
        for v in vertexlist:
            rgb = sRGBColor(
                rgb_r=cloud.points['red'].loc[v],
                rgb_g=cloud.points['green'].loc[v],
                rgb_b=cloud.points['blue'].loc[v],
                is_upscaled=is_upscaled
            )
            hsv: HSVColor = convert_color(rgb, HSVColor)
            h_values.append(hsv.hsv_h)
            s_values.append(hsv.hsv_s)
            v_values.append(hsv.hsv_v)
        h_values.sort()
        s_values.sort()
        v_values.sort()
        length = len(h_values)
        if length % 2 != 0:
            h_median = h_values[int(length / 2)]
            s_median = s_values[int(length / 2)]
            v_median = v_values[int(length / 2)]
        else:
            h_median = (h_values[math.floor(length / 2)] + h_values[
                math.ceil(length / 2)]) / 2
            s_median = (s_values[math.floor(length / 2)] + s_values[
                math.ceil(length / 2)]) / 2
            v_median = (v_values[math.floor(length / 2)] + v_values[
                math.ceil(length / 2)]) / 2
        hsv = HSVColor(hsv_h=h_median, hsv_s=s_median, hsv_v=v_median)
        srgb: sRGBColor = convert_color(hsv, sRGBColor)
        if is_upscaled:
            color_matching[polygon] = srgb.get_upscaled_value_tuple()
        else:
            color_matching[polygon] = srgb.get_value_tuple()
    return color_matching

def calc_colors_with_kmeans_rgb(matching: dict,
                                cloud: PyntCloud,
                                k: int = 8,
                                random_state: int = None):
    """
    Bestimmt Flächenfarben durch eine k-means Clusterbildung. Die Flächenfarbe
    wird das Clusterzentrum des größten Clusters gebildet
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als PyntCloud
    :param k: Anzahl der zu bildenden Cluster
    :param random_state: int, um Ergebnisse reproduzierbar zu machen.
    :return: color matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        colors = cloud.points.get(['red', 'green', 'blue']).loc[vertexlist].values
        if len(colors) >= 5*k:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=random_state,
                n_init='auto'
            ).fit(colors)
            centroids = list(kmeans.cluster_centers_)
            labels = kmeans.labels_
            counts = list(np.bincount(labels))
            max_index = counts.index(max(counts))
            maximum = centroids[max_index]
            color_matching[polygon] = (maximum[0], maximum[1], maximum[2])
    return color_matching


def calc_colors_with_kmeans_hsv(matching: dict,
                                cloud: PyntCloud,
                                k: int = 8,
                                random_state: int = None,
                                is_upscaled: bool = False):
    """
    Bestimmt Flächenfarben durch eine k-means Clusterbildung. Die Flächenfarbe
    wird das Clusterzentrum des größten Clusters gebildet
    :param matching: Matching von Flächen und Punkten aus der Punktwolke
    :param cloud: Punktwolke als PyntCloud
    :param k: Anzahl der zu bildenden Cluster
    :param random_state: int, um Ergebnisse reproduzierbar zu machen.
    :param is_upscaled: bool, sRGB Werte der Punktwolke als 8-bit int?
    :return: color matching mit Flächen als Keys und (r, g, b)-Tuple als Values
    """
    color_matching = {}
    for polygon, vertexlist in matching.items():
        colors = cloud.points.get(['red', 'green', 'blue']).loc[vertexlist].values
        if len(colors) >= 5*k:
            hsv_colors = []
            for color in colors:
                srgb = sRGBColor(
                    rgb_r=color[0],
                    rgb_g=color[1],
                    rgb_b=color[2],
                    is_upscaled=is_upscaled
                )
                hsv: HSVColor = convert_color(srgb, HSVColor)
                hsv_colors.append([hsv.hsv_h, hsv.hsv_s, hsv.hsv_v])
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=random_state,
                n_init='auto'
            ).fit(hsv_colors)
            centroids = list(kmeans.cluster_centers_)
            labels = kmeans.labels_
            counts = list(np.bincount(labels))
            max_index = counts.index(max(counts))
            maximum = centroids[max_index]
            hsv = HSVColor(
                hsv_h=maximum[0],
                hsv_s=maximum[1],
                hsv_v=maximum[2]
            )
            srgb: sRGBColor = convert_color(hsv, sRGBColor)
            if is_upscaled:
                color_matching[polygon] = srgb.get_upscaled_value_tuple()
            else:
                color_matching[polygon] = srgb.get_value_tuple()
    return color_matching


# ------------------------------------------------------------------------------
# Funktionen zum kontrollieren der Farbwerte
# ------------------------------------------------------------------------------

def validate_colors(color1: tuple, color2: tuple, is_upscaled: bool = False):
    """
    Berechnet Delta E (76) für zwei sRGB Farben.
    :param color1: Farbe 1 als (r, g, b)-Tuple
    :param color2: Farbe 2 als (r, g, b)-Tuple
    :param is_upscaled: bool, sRGB Werte der Punktwolke als 8-bit int?
    :return: delta_e
    """
    (r1, g1, b1) = color1
    (r2, g2, b2) = color2
    srgb1 = sRGBColor(
        rgb_r=r1,
        rgb_g=g1,
        rgb_b=b1,
        is_upscaled=is_upscaled
    )
    srgb2 = sRGBColor(
        rgb_r=r2,
        rgb_g=g2,
        rgb_b=b2,
        is_upscaled=is_upscaled
    )
    lab1: LabColor = convert_color(srgb1, LabColor)
    lab2: LabColor = convert_color(srgb2, LabColor)
    delta_l = (lab1.lab_l - lab2.lab_l)
    delta_a = (lab1.lab_a - lab2.lab_a)
    delta_b = (lab1.lab_b - lab2.lab_b)
    delta_e = math.sqrt((delta_l ** 2) + (delta_a ** 2) + (delta_b ** 2))
    return delta_e


# ------------------------------------------------------------------------------
# Funktionen zur Erstellung von Histogrammen
# ------------------------------------------------------------------------------


def createRGBHistogram(srgb_colors: dict, show: bool = False, file: str = None):
    # Histogramm erzeugen
    fig, ((ax0)) = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(16, 9)
    )

    ax0.hist(
        x=[srgb_colors['red'], srgb_colors['green'], srgb_colors['blue']],
        bins=256,
        histtype='bar',
        stacked=True,
        color=['red', 'green', 'blue']
    )
    ax0.set_title('RGB')

    plt.ticklabel_format(style='plain')  # ganzen Zahlen verwenden
    fig.tight_layout()

    # Plot ggf. ausgeben und speichern
    if file:
        plt.savefig(file)
    if show:
        plt.show()


def createHSLHistogram(hsl_colors: dict, show: bool = False, file: str = None):
    # Histogramm erzeugen
    fig, ((ax0), (ax1), (ax2)) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(16, 9)
    )

    ax0.hist(
        x=hsl_colors['hue'],
        bins=360,
        histtype='step'
    )
    ax0.set_title('Hue')

    ax1.hist(
        x=hsl_colors['saturation'],
        bins=100,
        histtype='step'
    )
    ax1.set_title('Saturation')

    ax2.hist(
        x=hsl_colors['lightness'],
        bins=100,
        histtype='step'
    )
    ax2.set_title('Lightness')

    plt.ticklabel_format(style='plain')  # ganzen Zahlen verwenden
    fig.tight_layout()

    # Plot ggf. ausgeben und speichern
    if file:
        plt.savefig(file)
    if show:
        plt.show()

# ------------------------------------------------------------------------------
# Hauptprogramm
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Hilfsvariablen
    # --------------------------------------------------------------------------
    rgba_header = ['red', 'green', 'blue', 'alpha']
    rgb_header = ['red', 'green', 'blue']
    clean_header = ['x', 'y', 'z', 'red', 'green', 'blue']
    references = {
        8495: (137, 96, 87),    # Rosa link neben Mitte
        7725: (141, 149, 149),  # Weiße hervorhebung links neben 8495
        7179: (137, 145, 148),  # obere Säulenende Querbalken weiß (mitte)
        7739: (98, 72, 77),     # Linke Ecksäule rosa
        8096: (140, 141, 135),  # grüne Fensterfläche
        6948: (112, 92, 73),    # alte Fassade
        7699: (217, 220, 235),  # Durchgang 8495, rechte Wand
        8468: (141, 153, 164),  # Linker Säulenfuß, front
        8524: (151, 111, 105),  # Links neben 8495
        8527: (160, 125, 124),  # Links neben 8524
    }

    # --------------------------------------------------------------------------
    # Daten importieren
    # --------------------------------------------------------------------------

    # Punktwolke aus Pickle Dateien importieren
    cloud = import_pointcloud_from_pickle(
        # flasche Koordinaten, richtige Farben als float-Werte
        file='pickleddata/rathaus_float_colors.pkl'
    )
    cloud_upscaled = import_pointcloud_from_pickle(
        # richtige Koordinaten, richtige Farben
        file='pickleddata/rathaus_upscaled_colors.pkl'
    )
    cloud_las = import_pointcloud_from_pickle(
        # falsche Koordinaten, richtige Farben, nicht als PLY exportierbar (las)
        file='pickleddata/rathaus_las.pkl'
    )
    cloud_ply = import_pointcloud_from_pickle(
        # richtige Koordinaten, falsche Farben (zu dunkel)
        file='pickleddata/pointcloud_blender_export.pkl'
    )

    # Punktwolke aus PLY Datei importieren
    """
    cloud_upscaled = import_pointcloud(
        file='ply/rathaus_upscaled_colors.ply'
    )
    """

    # Matching importieren
    matching = import_matching()

    # --------------------------------------------------------------------------
    # Daten exportieren
    # --------------------------------------------------------------------------

    # Punktwolke als Pickle exportieren → Pickle kann schneller importieren
    """
    export_pointcloud_as_pickle(
        cloud=cloud,
        file='pickleddata/rathaus_float_colors.pkl'
    )
    """

    # --------------------------------------------------------------------------
    # Weiteres
    # --------------------------------------------------------------------------

    # Ausgabe der Punktwolken zur Überprüfung, ob alle da sind ;)
    #print(cloud_las.points.get(clean_header))
    #print(cloud.points.get(clean_header))
    #print(cloud_ply.points.get(clean_header))
    #print(cloud_upscaled.points.get(clean_header))



    # Color Matching mit Durchschnitt RGB
    """
    color_matching = calc_colors_with_rgb_average(
        matching=matching,
        cloud=cloud_upscaled
    )
    """

    # Color Matching mit Durchschnitt HSV
    """
    color_matching = calc_colors_with_hsv_average(
        matching=matching,
        cloud=cloud_upscaled,
        is_upscaled=True
    )
    """

    # Color Matching mit BoxPlot RGB
    """
    color_matching = calc_colors_with_boxplot_rgb(
        matching=matching,
        cloud=cloud_upscaled
    )
    """

    # Color Matching mit BoxPlot HSV
    """
    color_matching = calc_colors_with_boxplot_hsv(
        matching=matching,
        cloud=cloud_upscaled,
        is_upscaled=True
    )
    """

    # Color Matching mit Median RGB
    """
    color_matching = calc_colors_with_Median(
        matching=matching,
        cloud=cloud_upscaled
    )
    """

    # Color Matching mit Median HSV

    color_matching = calc_colors_with_Median_HSV(
        matching=matching,
        cloud=cloud_upscaled,
        is_upscaled=True
    )


    # Color Matching mit RGB kmeans berechnen
    """
    color_matching = calc_colors_with_kmeans_rgb(
        matching=matching,
        cloud=cloud_upscaled,
        k=7,
        random_state=1,
    )
    """

    # Color Matching mit HSV kmeans berechnen
    """
    color_matching = calc_colors_with_kmeans_hsv(
        matching=matching,
        cloud=cloud_upscaled,
        k=7,
        random_state=1,
        is_upscaled=True
    )
    """

    # Berechnung und Ausgabe der Farbdifferenz für die Referenzflächen für eine
    # Auswertung
    print('FlächenID: Referenzfarbe; Farbdifferenz Delta E; Berechnete Farbe ')
    for polygon, color in references.items():
        (r, g, b) = color_matching[polygon]
        calc_col = (r, g, b)
        delta_e = validate_colors(
            color1=(r, g, b),
            color2=color,
            is_upscaled=True)
        print(f'{polygon}: {color}; {delta_e}; {calc_col}')
        srgb = sRGBColor(r, g, b, True)
        print(srgb.rgb_r, srgb.rgb_g, srgb.rgb_b)
        print('   ')


    # Color Matching mit Durchschnitt
    """
    color_matching = calc_colors_with_average(
        matching=matching,
        cloud=cloud_upscaled
    )
    """

    # RGB-Histogramm pro Fläche erzeugen
    """
    for polygon, vertexlist in matching.items():
        r_values, g_values, b_values = [], [], []
        for v in vertexlist:
            r_values.append(cloud_upscaled.points['red'].loc[v])
            g_values.append(cloud_upscaled.points['green'].loc[v])
            b_values.append(cloud_upscaled.points['blue'].loc[v])
        c = {'red': r_values, 'green': g_values, 'blue': b_values}
        createRGBHistogram(
            srgb_colors=c,
            show=False,
            file=f'diagrams/{polygon}_rgb.png')
    """

    # HSL-Histogramm pro Fläche erzeugen
    """
    for polygon, vertexlist in matching.items():
        h_values, s_values, l_values = [], [], []
        for v in vertexlist:
            rgb = sRGBColor(
                rgb_r=cloud_upscaled.points['red'].loc[v],
                rgb_g=cloud_upscaled.points['green'].loc[v],
                rgb_b=cloud_upscaled.points['blue'].loc[v],
                is_upscaled=True,
            )
            hsl: HSLColor = convert_color(rgb, HSLColor)
            h_values.append(hsl.hsl_h)
            s_values.append(hsl.hsl_s)
            l_values.append(hsl.hsl_l)
        c = {'hue': h_values, 'saturation': s_values, 'lightness': l_values}
        createHSLHistogram(
            hsl_colors=c,
            show=False,
            file=f'diagrams/hist_hsl_per_face/{polygon}_hsl.png'
        )
    """

    # Color-Matching mit Median erzeugen und picklen
    """
    color_matching = calc_colors_with_Median(
        matching=matching,
        cloud=cloud
    )
    print(color_matching)
    print(len(color_matching))

    with open(
        file='pickleddata/color_matching.pkl',
        mode='wb'
    ) as output:
        # Matching in Datei schrieben
        pickle.dump(color_matching, output)
    # Datei schließen
    output.close()
    """


    # upscalen der Farbwerte
    """
    df: pd.DataFrame = cloud.points['red']
    cloud.points['red'] = df.transform(upscale_color_r)
    print('Rot wurde umgerechnet.')
    df: pd.DataFrame = cloud.points['green']
    cloud.points['green'] = df.transform(upscale_color_g)
    print('Grün wurde umgerechnet.')
    df: pd.DataFrame = cloud.points['blue']
    cloud.points['blue'] = df.transform(upscale_color_b)
    print('Blau wurde umgerechnet.')

    print(cloud.points[clean_header])

    # Erzeugen einer reinen neuen Punktwolke ohne zusätzliche scalar-fields
    upscaled_cloud = PyntCloud(
        points=cloud.points[clean_header]
    )
    """

    # Exportiere upscaled_cloud als Pickle und PLY
    """
    export_pointcloud_as_pickle(
        cloud=upscaled_cloud,
        file='pickleddata/rathaus_upscaled_colors.pkl'
    )
    export_cloud.to_file('ply/rathaus_upscaled_colors.ply')
    """

    # Bereite Histogramme vor
    """
    print('Sortiere Farben und wechsle ggf. Farbraum...')

    colors = upscaled_cloud.points[rgb_header]
    srgb_colors = {'red': [], 'green': [], 'blue': []}
    hsl_colors = {'hue': [], 'saturation': [], 'lightness': []}
    lab_colors = {'l': [], 'a': [], 'b': []}

    colorvalues = giveNextValue(colors.values)
    for i in range(len(colors.values)):
        if i % 4 == 0:
            element = next(colorvalues)

            # sRGB Farbwerte
            srgb = sRGBColor(element[0], element[1], element[2], True)
            (r, g, b) = srgb.get_upscaled_value_tuple()
            srgb_colors['red'].append(r)
            srgb_colors['green'].append(g)
            srgb_colors['blue'].append(b)
            

            # HSL Farbwerte
            hsl: HSLColor = convert_color(srgb, HSLColor)
            hsl_colors['hue'].append(hsl.hsl_h)
            hsl_colors['saturation'].append(hsl.hsl_s)
            hsl_colors['lightness'].append(hsl.hsl_l)

            # CIELAB Farbwerte
            lab: LabColor = convert_color(srgb, LabColor)
            lab_colors['l'].append(lab.lab_l)
            lab_colors['a'].append(lab.lab_a)
            lab_colors['b'].append(lab.lab_b)

    print('Farben umgewandelt und sortiert. Erzeuge Histogramme...')
    """
    # Erzeuge RGB Histogramm
    """
    createRGBHistogram(
        srgb_colors=srgb_colors,
        show=True,
        file='diagrams/rgb.png'
    )
    """

    # Erzeuge HSL Histogramm
    """
    createHSLHistogram(
        hsl_colors=hsl_colors,
        show=True,
        file='diagrams/hsl.png'
    )
    """

