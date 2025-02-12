from LIACEI_workflow.DinamicaMolecular import DinamicaMolecular
from LIACEI_workflow.DinamicaMolecular import Frame
import h5py
import yaml

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