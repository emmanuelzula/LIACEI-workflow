import os
import pandas as pd
import numpy as np
from scripts.data import DinamicaMolecular
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
        
def configuracion_TorchMDNet(archivo_configuracion, **kwargs):
    """
    Actualiza o crea un archivo de configuración .yaml con valores predeterminados.
    Si el archivo ya existe, conserva la configuración actual y solo actualiza los parámetros proporcionados.
    """
    # Valores por defecto
    valores_por_defecto = {
        "num_epochs": 300,
        "batch_size": 32,
        "inference_batch_size": None,
        "lr": 1e-4,
        "lr_patience": 10,
        "lr_metric": "val_loss",
        "lr_min": 1e-6,
        "lr_factor": 0.8,
        "lr_warmup_steps": 0,
        "early_stopping_patience": 30,
        "reset_trainer": False,
        "weight_decay": 0.0,
        "ema_alpha_y": 1.0,
        "ema_alpha_neg_dy": 1.0,
        "ngpus": -1,
        "num_nodes": 1,
        "precision": 32,
        "log_dir": "output",
        "splits": "input/splits.npz",
        "train_size": None,
        "val_size": 0.05,
        "test_size": 0.1,
        "test_interval": 10,
        "save_interval": 10,
        "seed": 1,
        "num_workers": 4,
        "redirect": False,
        "gradient_clipping": 0.0,
        "dataset": "HDF5",
        "dataset_root": "input/input_torchmd-net.h5",
        "dataset_arg": None,
        "coord_files": None,
        "embed_files": None,
        "energy_files": None,
        "force_files": None,
        "y_weight": 1.0,
        "neg_dy_weight": 1.0,
        "model": "equivariant-transformer",
        "output_model": "Scalar",
        "prior_model": None,
        "charge": False,
        "spin": False,
        "embedding_dimension": 256,
        "num_layers": 6,
        "num_rbf": 64,
        "activation": "silu",
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "neighbor_embedding": False,
        "aggr": "add",
        "distance_influence": "both",
        "attn_activation": "silu",
        "num_heads": 8,
        "equivariance_invariance_group": "O(3)",
        "derivative": False,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "atom_filter": -1,
        "max_z": 100,
        "max_num_neighbors": 32,
        "standardize": False,
        "reduce_op": "add",
        "wandb_use": False,
        "wandb_name": "training",
        "wandb_project": "training_",
        "wandb_resume_from_id": None,
        "tensorboard_use": False
    }

    # Si el archivo ya existe, cargar su contenido
    if os.path.exists(archivo_configuracion):
        with open(archivo_configuracion, "r") as archivo:
            configuracion_actual = yaml.safe_load(archivo) or {}
    else:
        configuracion_actual = {}

    # Mezclar valores por defecto con la configuración actual
    configuracion = {**valores_por_defecto, **configuracion_actual}

    # Actualizar solo los valores proporcionados en kwargs
    for clave, valor in kwargs.items():
        if valor == "Default":
            configuracion[clave] = valores_por_defecto[clave]
        else:
            configuracion[clave] = valor

    # Guardar la configuración actualizada en el archivo
    with open(archivo_configuracion, "w") as archivo:
        yaml.dump(configuracion, archivo)

def crear_script_entrenamiento(ruta_script, gpus, ruta_conf,ruta_archivo_de_salida):
    """
    Crea un script bash para iniciar un entrenamiento con TorchMD.

    :param ruta_archivo_de_salida: Ruta donde se guardará el script.
    :param gpus: Lista de GPUs a utilizar (e.g., "0,1,2,3").
    :param ruta_conf: Ruta al archivo de configuración.
    """
    # Construir el comando principal
    comando_principal = f"CUDA_VISIBLE_DEVICES={gpus} torchmd-train --conf {ruta_conf}"

    # Contenido del script bash
    texto_script = f"""#!/bin/bash

# Archivo para guardar la información
output_file={ruta_archivo_de_salida}

# Crear directorio de salida si no existe
mkdir -p $(dirname "$output_file")

# Guardar el comando de ejecución y la hora de inicio
echo "Comando de ejecución: {comando_principal}" > "$output_file"
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Inicio del entrenamiento: $start_time" >> "$output_file"

# Ejecutar el comando de entrenamiento
{comando_principal}

# Guardar la hora de finalización
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Fin del entrenamiento: $end_time" >> "$output_file"

# Calcular la duración del entrenamiento
start_seconds=$(date -d "$start_time" +%s)
end_seconds=$(date -d "$end_time" +%s)
duration_seconds=$((end_seconds - start_seconds))

# Guardar la duración en el archivo
echo "Duración del entrenamiento: $((duration_seconds / 60)) minutos y $((duration_seconds % 60)) segundos" >> "$output_file"

echo "Información del entrenamiento guardada en $output_file"
"""

    # Guardar el script bash
    with open(ruta_script, 'w') as archivo:
        archivo.write(texto_script)

    # Hacer el script ejecutable
    os.chmod(ruta_script, 0o755)

    print(f"El script se ha guardado correctamente en: {ruta_script}")
