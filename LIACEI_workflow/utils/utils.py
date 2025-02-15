import os
import pandas as pd
import numpy as np
from LIACEI_workflow.data.DinamicaMolecular import DinamicaMolecular
import yaml

def crear_carpeta(ruta_carpeta):
    """
    Crea una carpeta en la ruta especificada si no existe.

    :param ruta_carpeta: Ruta de la carpeta a crear.
    """
    try:
        os.makedirs(ruta_carpeta, exist_ok=True)
        print(f"Carpeta creada o ya existente: {ruta_carpeta}")
    except Exception as e:
        print(f"Error al crear la carpeta {ruta_carpeta}: {e}")

def procesar_y_guardar_csv_a_npz(numero_frames, train=0.81, test=0.1, val=0.09):
    """
    Verifica si existe un archivo .csv en la carpeta 'input', lo procesa, convierte los datos a .npz y los guarda en la misma carpeta.

    :param numero_frames: Número esperado de frames para verificar con los índices del CSV.
    :param train: Porcentaje de datos para el conjunto de entrenamiento.
    :param test: Porcentaje de datos para el conjunto de prueba.
    :param val: Porcentaje de datos para el conjunto de validación.
    """
    # Definir la carpeta de búsqueda
    carpeta_input = os.path.join(os.getcwd(), "input")

    # Verificar si la carpeta 'input' existe
    if not os.path.exists(carpeta_input):
        print("La carpeta 'input' no existe.")
        return

    # Buscar archivos en la carpeta 'input'
    archivos_en_directorio = os.listdir(carpeta_input)
    archivo_csv_nombre = None
    for archivo in archivos_en_directorio:
        if os.path.isfile(os.path.join(carpeta_input, archivo)) and archivo.endswith('.csv'):
            archivo_csv_nombre = archivo
            break

    if archivo_csv_nombre:
        # Cargar el archivo .csv y convertir la columna "ids" a tipo int
        ruta_csv = os.path.join(carpeta_input, archivo_csv_nombre)
        archivo_csv_ = pd.read_csv(ruta_csv)
        archivo_csv_["ids"] = archivo_csv_["ids"].astype(int)

        # Verificar que el número de frames coincida con la longitud de los índices del archivo CSV
        if len(archivo_csv_) != numero_frames:
            print(f"Error: El número de frames ({numero_frames}) no coincide con la cantidad de índices en el archivo CSV ({len(archivo_csv_)}).")
            return

        # Definir los tamaños de los conjuntos
        total_size = len(archivo_csv_)
        test_size = int(total_size * test)
        val_size = int(total_size * val)

        # Crear el diccionario para almacenar los conjuntos
        splits = {
            'idx_test': archivo_csv_["ids"][:test_size].values,
            'idx_val': archivo_csv_["ids"][test_size:test_size+val_size].values,
            'idx_train': archivo_csv_["ids"][test_size+val_size:].values
        }

        # Guardar los arrays en un archivo .npz en la carpeta 'input'
        np.savez(os.path.join(carpeta_input, "splits.npz"), idx_train=splits['idx_train'], idx_val=splits['idx_val'], idx_test=splits['idx_test'])
        print(f"El archivo {archivo_csv_nombre} se transformó a .npz y se guardó en la carpeta 'input'/")
    else:
        print("No se encontró ningún archivo .csv en la carpeta 'input'.")


def cargar_o_generar_npz(md_step, train=0.81, test=0.1, val=0.09):
    """
    Busca un archivo .npz en la carpeta 'input'. Si no lo encuentra, genera los ids para los conjuntos de datos.

    :param md_step: Número total de frames disponibles.
    :param train: Proporción de datos para el conjunto de entrenamiento.
    :param test: Proporción de datos para el conjunto de prueba.
    :param val: Proporción de datos para el conjunto de validación.
    """
    # Validar proporciones
    if not np.isclose(train + test + val, 1.0):
        raise ValueError("Las proporciones deben sumar 1.0")

    # Definir la carpeta de búsqueda
    carpeta_input = os.path.join(os.getcwd(), "input")

    # Verificar si la carpeta 'input' existe
    if not os.path.exists(carpeta_input):
        os.makedirs(carpeta_input)

    # Buscar archivos .npz en la carpeta 'input'
    archivos_en_directorio = os.listdir(carpeta_input)
    archivo_npz_encontrado = any(archivo.endswith('.npz') for archivo in archivos_en_directorio)

    if archivo_npz_encontrado:
        print("Se encontró un archivo .npz en la carpeta 'input'. No se generarán nuevos ids.")
    else:
        # Generar los conjuntos de datos si no se encuentra un archivo .npz
        np.random.seed(42)
        numeros = np.arange(md_step)

        # Permutar aleatoriamente los números
        numeros_aleatorios = np.random.permutation(numeros)

        # Calcular las longitudes de los conjuntos de datos
        total_elementos = len(numeros_aleatorios)
        num_elementos_test = int(test * total_elementos)
        num_elementos_val = int(val * total_elementos)

        # Dividir el array en tres conjuntos según las proporciones especificadas
        idx_test, idx_val, idx_train = np.split(numeros_aleatorios, [num_elementos_test, num_elementos_test + num_elementos_val])

        # Guardar los conjuntos en un archivo .npz
        ruta_npz = os.path.join(carpeta_input, "splits.npz")
        np.savez(ruta_npz, idx_test=idx_test, idx_train=idx_train, idx_val=idx_val)

        print("No se encontró un archivo .npz en la carpeta 'input'. Se generaron los ids de cada conjunto de datos y se guardaron en 'input/splits.npz'.")