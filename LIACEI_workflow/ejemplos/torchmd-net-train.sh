#!/bin/bash

# Activar el entorno virtual de Mamba
source ~/.bashrc  # Asegura que Mamba esté disponible en la sesión
mamba activate torchmd-net

# Verificar si el entorno virtual se activó correctamente
if [[ $? -ne 0 ]]; then
    echo "Error: No se pudo activar el entorno virtual 'torchmd-net'. Verifica la instalación de Mamba."
    exit 1
fi

echo "Entorno virtual 'torchmd-net' activado."

# Ejecutar el script en Python
python - <<EOF
from LIACEI_workflow.data.DinamicaMolecular import cargar_desde_data
from LIACEI_workflow.utils import crear_carpeta
from LIACEI_workflow.codes.torchmd_net.utils import guardar_a_TorchMDNet

# Cargar y convertir datos .data
base_de_datos = cargar_desde_data("input/input.data")

# Transformar .data al formato de TorchMD-Net
crear_carpeta("transform")
guardar_a_TorchMDNet(base_de_datos, "transform/input_torchmd-net.h5")
EOF

if [[ $? -ne 0 ]]; then
    echo "Error en la ejecución de Python."
    mamba deactivate
    exit 1
fi

echo "Conversión de datos completada."

# Ejecutar el comando de entrenamiento con TorchMD-Net y medir tiempo
echo "Iniciando entrenamiento con TorchMD-Net..."

start_time=$(date +%s)  # Captura el tiempo de inicio

torchmd-train --conf input/input.yaml --log-dir train

if [[ $? -ne 0 ]]; then
    echo "Error en el entrenamiento con TorchMD-Net."
    mamba deactivate
    exit 1
fi

end_time=$(date +%s)  # Captura el tiempo de finalización
elapsed_time=$((end_time - start_time))  # Calcula la diferencia en segundos

# Convertir a días, horas, minutos y segundos
days=$((elapsed_time / 86400))
hours=$(( (elapsed_time % 86400) / 3600 ))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

echo "Entrenamiento finalizado correctamente en ${days} días, ${hours} horas, ${minutes} minutos y ${seconds} segundos."

# Desactivar el entorno virtual de Mamba
mamba deactivate
echo "Entorno virtual 'torchmd-net' desactivado."
