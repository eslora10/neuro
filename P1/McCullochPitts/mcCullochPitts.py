# -*- coding: utf-8 -*-
import sys

umbral = 2
red = { "x1": {"valor": 0, "incidencia": {}},
        "x2": {"valor": 0, "incidencia": {}},
        "x3": {"valor": 0, "incidencia": {}},
        "z11": {"valor": 0, "incidencia": {"x1": 2}},
        "z12": {"valor": 0, "incidencia": {"x2": 2}},
        "z13": {"valor": 0, "incidencia": {"x3": 2}},
        "z21": {"valor": 0, "incidencia": {"x3": 1, "z11": 1}},
        "z22": {"valor": 0, "incidencia": {"x1": 1, "z12": 1}},
        "z23": {"valor": 0, "incidencia": {"x2": 1, "z13": 1}},
        "z31": {"valor": 0, "incidencia": {"x2": 1, "z11": 1}},
        "z32": {"valor": 0, "incidencia": {"x3": 1, "z12": 1}},
        "z33": {"valor": 0, "incidencia": {"x1": 1, "z13": 1}},
        "y1": {"valor": 0, "incidencia": {"z21": 2, "z22": 2, "z23": 2}},
        "y2": {"valor": 0, "incidencia": {"z31": 2, "z32": 2, "z33": 2}},
}

def respuesta(red, x1, x2, x3):
    red["x1"]["valor"] = x1
    red["x2"]["valor"] = x2
    red["x3"]["valor"] = x3

    siguientes = ["y1", "y2"]
    while siguientes:
        nodo = siguientes[0]
        incidencia = red[nodo]["incidencia"]
        if nodo != "x1" and nodo != "x2" and nodo != "x3":
            red[nodo]["valor"] = 0
            for conexion, peso in incidencia.items():
                red[nodo]["valor"] += peso*red[conexion]["valor"]
                if conexion not in siguientes:
                    siguientes.append(conexion)
            red[nodo]["valor"] = int(red[nodo]["valor"] >= umbral)
        siguientes.remove(nodo)

    print(red["y1"]["valor"], red["y2"]["valor"])

filename = sys.argv[1]
with open(filename) as entrada:
    for line in entrada:
        parsed = line.strip("\n").split()
        x1 = int(parsed[0])
        x2 = int(parsed[1])
        x3 = int(parsed[2])
        respuesta(red, x1, x2, x3)

    respuesta(red, 0, 0, 0)
