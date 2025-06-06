from LIACEI_workflow.data.DinamicaMolecular import DinamicaMolecular, Frame
import h5py
import yaml
import os
import numpy as np
import pandas as pd

def guardar_a_TorchMDNet(dinamica_molecular, archivo):
        """
        Guarda una instancia de DinamicaMolecular en un archivo HDF5, agrupando los frames por número de átomos,
        incluyendo un dataset 'n_frame' para conservar los números de frame originales.

        :param dinamica_molecular: Objeto DinamicaMolecular a guardar.
        :param archivo: Nombre del archivo HDF5.
        """
        with h5py.File(archivo, "w") as archivo_h5:
            # Iterar por cada número de átomos único
            for num_atomos in set(frame.numero_atomos for frame in dinamica_molecular.frames.values()):
                # Crear un grupo para este número de átomos
                group = archivo_h5.create_group(f"{num_atomos}_atoms")
                
                # Crear listas para almacenar los datos de los frames correspondientes
                energias = []
                tipos = []
                posiciones = []
                fuerzas = []
                numeros_frames = []  # Lista para guardar los números de frame originales

                # Filtrar frames por el número de átomos
                for numero_frame, frame in dinamica_molecular.frames.items():
                    if frame.numero_atomos == num_atomos:
                        energias.append(frame.energia)
                        tipos.append(frame.elementos)
                        posiciones.append(frame.posiciones)
                        fuerzas.append(frame.fuerzas)
                        numeros_frames.append(numero_frame)

                # Convertir listas a arrays NumPy y guardarlas en el grupo
                group.create_dataset("energy", data=np.array(energias, dtype=np.float32), compression="gzip")
                group.create_dataset("types", data=np.array(tipos, dtype=np.int32), compression="gzip")
                group.create_dataset("pos", data=np.array(posiciones, dtype=np.float32), compression="gzip")
                group.create_dataset("forces", data=np.array(fuerzas, dtype=np.float32), compression="gzip")
                group.create_dataset("n_frame", data=np.array(numeros_frames, dtype=np.int32), compression="gzip")

def cargar_desde_TorchMDNet(archivo):
    """
    Carga una instancia de DinamicaMolecular desde un archivo HDF5, recuperando los frames agrupados por número de átomos.

    :param archivo: Nombre del archivo HDF5.
    :return: Objeto DinamicaMolecular reconstruido.
    """
    # Crear una nueva instancia de DinamicaMolecular
    dinamica_molecular = DinamicaMolecular()

    with h5py.File(archivo, "r") as archivo_h5:
        # Iterar por cada grupo en el archivo HDF5
        for grupo_nombre in archivo_h5:
            grupo = archivo_h5[grupo_nombre]

            # Recuperar los datos de los datasets
            energias = grupo["energy"][:]
            tipos = grupo["types"][:]
            posiciones = grupo["pos"][:]
            fuerzas = grupo["forces"][:]
            numeros_frames = grupo["n_frame"][:]

            # Reconstruir los frames a partir de los datos
            for i, numero_frame in enumerate(numeros_frames):
                frame = Frame(energia=energias[i])
                frame.elementos = tipos[i].tolist()  # Convertir a lista
                frame.posiciones = posiciones[i]
                frame.fuerzas = fuerzas[i]

                # Agregar el frame reconstruido a la DinamicaMolecular
                dinamica_molecular.agregar_frame(frame, numero_frame=numero_frame)

    return dinamica_molecular

def obtener_ruta_modelo_entrenado(directorio='train'):
    import re
    """
    Busca el archivo de modelo entrenado con el mayor número de epoch dentro del directorio especificado.

    Args:
        directorio (str): Ruta del directorio donde buscar los modelos. Por defecto, 'output'.

    Returns:
        str: Ruta del archivo del modelo entrenado con el máximo número de epoch, o None si no se encuentra.
    """
    nombre_max_epoch = max(
        (f for f in os.listdir(directorio) if re.match(r'epoch=(\d+)-val_loss', f)),
        key=lambda x: int(re.search(r'epoch=(\d+)-val_loss', x).group(1)),
        default=None
    )

    return os.path.join(directorio, nombre_max_epoch) if nombre_max_epoch else None

def obtener_indices_test(ruta_splits):
    """
    Carga un archivo .npz y extrae los índices de prueba ('idx_test') ordenados.

    Args:
        ruta_splits (str): Ruta al archivo .npz que contiene los splits de datos.

    Returns:
        numpy.ndarray: Un array con los índices de prueba ordenados, o None si no se encuentra 'idx_test'.
    """
    try:
        splits = np.load(ruta_splits)
        if 'idx_test' in splits:
            return np.sort(splits['idx_test'])
        else:
            raise KeyError("El archivo .npz no contiene la clave 'idx_test'.")
    except Exception as e:
        print(f"Error al cargar los índices de prueba: {e}")
        return None

import torch
import gc
from torchmdnet.models.model import load_model

def generar_inferencias(DinamicaMolecular, modelo_entrenado, gpu=None):
    """
    Realiza inferencias sobre los frames en DinamicaMolecular utilizando un modelo de TorchMD-Net.

    Args:
        DinamicaMolecular: Instancia de la clase que contiene los frames a procesar.
        modelo_entrenado (str): Ruta al modelo preentrenado.
        gpu (str, opcional): ID de la GPU a utilizar (si es None, usa la primera definida en config.yaml).

    Returns:
        DinamicaMolecular: Instancia con los valores de energía y fuerzas actualizados.
    """
    # Cargar la configuración desde el archivo config.yaml
    with open("input/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Obtener la primera GPU del archivo YAML si no se especifica una
# Obtener la primera GPU del archivo YAML si no se especifica una
    if gpu is None:
        if isinstance(config["ngpus"], str):  # Si es string, hacer split
            gpu = int(config["ngpus"].split(",")[0])
        elif isinstance(config["ngpus"], int):  # Si ya es un entero, usarlo directamente
            gpu = config["ngpus"]
        else:
            raise TypeError(f"Error: config['ngpus'] tiene un tipo inesperado: {type(config['ngpus'])}")

    # Si gpu es -1, asignar GPU 0 por defecto
    if gpu == -1:
        gpu = 0  
        
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(modelo_entrenado, derivative=True, device=device)

    for frame in DinamicaMolecular.frames.values():
        torch.cuda.empty_cache()

        # Convertir datos a tensores
        pos = torch.tensor(frame.posiciones, dtype=torch.float32, device=device)
        types = torch.tensor(frame.elementos, dtype=torch.long, device=device)

        n_energy_inferred, n_forces_inferred = model(types, pos)

        # Guardar resultados (asegurar que estén en CPU)
        frame.energia = n_energy_inferred.cpu().item()  # Convertir a float puro
        frame.fuerzas = n_forces_inferred.cpu().detach().numpy()  # Convertir a numpy array sin gradientes

        # Limpiar memoria
        del pos, types, n_energy_inferred, n_forces_inferred
        gc.collect()
        torch.cuda.empty_cache()

    return DinamicaMolecular  # Retorna el objeto actualizado

def guardar_metricas(path_metric_csv, output_path):
    """
    Lee un archivo CSV de métricas, selecciona columnas específicas y guarda en un archivo .data con formato estructurado.

    Args:
        path_metric_csv (str): Ruta del archivo CSV de entrada.
        output_path (str): Ruta del archivo .data de salida.

    Returns:
        None. Guarda los datos en el archivo de salida.
    """
    # Leer el archivo CSV en un DataFrame de pandas
    metrics = pd.read_csv(path_metric_csv)

    # Seleccionar las columnas deseadas
    selected_columns = ['epoch', 'train_loss', 'val_loss']
    train_metrics = metrics[selected_columns]

    # Guardar en un archivo .data con el encabezado estructurado
    with open(output_path, 'w') as file:
        file.write("#Epoch MAE_Train MAE_Val\n")
        train_metrics.to_csv(file, sep=' ', index=False, header=False, float_format='%.6f')


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