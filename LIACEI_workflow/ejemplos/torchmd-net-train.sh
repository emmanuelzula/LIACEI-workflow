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
from LIACEI_workflow.analysis.DinamicaMolecular import Graficas_genericas
from LIACEI_workflow.data.DinamicaMolecular import DinamicaMolecular, ElementoQuimico, Utils
from LIACEI_workflow.utils.utils import crear_carpeta
from LIACEI_workflow.codes.torchmd_net.utils import (
    cargar_desde_TorchMDNet,
    generar_inferencias,
    guardar_a_TorchMDNet,
    guardar_metricas,
    obtener_indices_test,
    obtener_ruta_modelo_entrenado
)

# cargar y convertir datos .data
base_de_datos=DinamicaMolecular.cargar_desde_data("input/input.data")

for frame in base_de_datos.frames.values():
    # Calcular la suma de las energías atómicas de los elementos
    energia_total = sum(ElementoQuimico(e).energia_atomica() for e in frame.elementos)
    
    # Restar la energía atómica total de la energía actual del frame
    frame.energia -= energia_total  # Actualizar directamente el atributo

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

torchmd-train --conf input/config.yaml --log-dir train

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

echo "Iniciando generación de inferencias"

python - <<EOF
ruta_modelo = obtener_ruta_modelo_entrenado("train")
base_de_datos = cargar_desde_TorchMDNet("transform/input_torchmd-net.h5")
test_idx = obtener_indices_test("train/splits.npz")
subconjunto_test = DinamicaMolecular.subconjunto(base_de_datos, test_idx)
inferencias = generar_inferencias(subconjunto_test,ruta_modelo)
DinamicaMolecular.guardar_a_hdf5(inferencias,"train/output_LIACEI.h5")
EOF

echo "Generación de inferencias finalizada correctamente"

echo "Iniciando generación de datos para graficar"

python - <<EOF
# Cargar referencias
referencias = cargar_desde_TorchMDNet("transform/input_torchmd-net.h5")
test_idx = obtener_indices_test("train/splits.npz")
referencias = DinamicaMolecular.subconjunto(referencias, test_idx)
# Cargar inferencias
inferencias = DinamicaMolecular.cargar_desde_hdf5("train/output_LIACEI.h5")
guardar_metricas("train/metrics.csv","analysis/epoch_vs_loss.data")
Utils.comparar_frames(referencias, inferencias)
Utils.calcular_angulos_y_guardar(referencias, inferencias)
Utils.calcular_modulos_fuerza_y_guardar(referencias, inferencias)
Utils.guardar_energia_referencia_vs_inferencia(referencias, inferencias)
EOF

echo "Generación de datos para graficar finalizada correctamente"

echo "Iniciando generación de Graficas"

python - <<EOF
Graficas_genericas.grafica_epoch_vs_loss()
Graficas_genericas.grafica_ref_vs_inf_energy()
Graficas_genericas.grafica_ref_vs_inf_energy_hist_2d()
Graficas_genericas.grafica_ref_vs_inf_energy_hist_3d()
Graficas_genericas.grafica_ref_vs_inf_energy_hist()
Graficas_genericas.grafica_ref_vs_inf_energy_abs_dif()
Graficas_genericas.grafica_ref_vs_inf_forces_angles_abs_dif_hydrogen()
Graficas_genericas.grafica_ref_vs_inf_forces_angles_abs_dif_oxygen()
Graficas_genericas.grafica_ref_vs_inf_forces_angles_abs_dif_aluminium()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_hydrogen()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_hydrogen_hist_2d()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_hydrogen_hist_3d()
Graficas_genericas.graficar_ref_vs_inf_forces_modules_hydrogen_hist()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_oxygen()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_oxygen_hist_2d()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_oxygen_hist_3d()
Graficas_genericas.graficar_ref_vs_inf_forces_modules_oxygen_hist()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_aluminium()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_aluminium_hist_2d()
Graficas_genericas.grafica_ref_vs_inf_forces_modules_aluminium_hist_3d()
Graficas_genericas.graficar_ref_vs_inf_forces_modules_aluminium_hist()
EOF

echo "Generación de graficas finalizada correctamente"

# Desactivar el entorno virtual de Mamba
mamba deactivate
echo "Entorno virtual 'torchmd-net' desactivado."
