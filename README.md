# Proyecto: Refactorización de LIACEI Workflow

## Descripción
Este proyecto implementa una refactorización del workflow de LIACEI, con mejoras en la organización del código y optimización de los procesos de transformación y análisis de datos.

## Estructura del Proyecto
- `01.ipynb`  → Notebook con el la transformación de los datos.
- `02.ipynb`  → Notebook con la configuración de TorchMD-Net.
- `analysis/`  → Carpeta para análisis de datos.
- `input/`  → Directorio donde se almacenan los datos de entrada.
- `LIACEI_workflow/`
  - `codes/torchmd-net/` → Funciones auxiliares para el manejo del codigo TorchMD-Net.
  - `data/` → Clases para el manejo de datos.
  - `utils/` → Funciones auxiliares.
- `train/` → Contiene los resultados del entrenamiento.
- `transform/` → Contiene los datos trasnfromados.

## Requisitos
- Python 3.x
- Dependencias especificadas en `requirements.txt`

## Instalación
1. Clonar el repositorio:
   ```
   git clone https://github.com/emmanuelzula/LIACEI-workflow
   ```
2. Ir a la carpata donde se encuentra setup.py:
   ```
   cd LIACEI_workflow/LIACEI_workflow/
   ```
3. Activar el entorno virtual del codigo a utilizar (ejemplo torchmd-net):
   ```
   mamba activate torchmd-net
   ```

4. Instalar los modulos como un paquete editable:
   ```
   pip install -e LIACEI_workflow/
   ```

## Uso
```

```
