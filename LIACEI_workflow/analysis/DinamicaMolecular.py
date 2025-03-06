import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import glob
from LIACEI_workflow.data.DinamicaMolecular import ElementoQuimico

class Graficas_genericas:
    """
    Clase para gestionar gráficas rápidas de dinámica molecular.
    """

    def __init__(self, output_dir="analysis"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def grafica_epoch_vs_loss(
        path_data_file="analysis/epoch_vs_loss.data", 
        output_path_tif="analysis/epoch_vs_loss.tiff",
        xlim=(0, 500), 
        ylim=(0, 3)
    ):
        os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)
        
        datos = pd.read_csv(
            path_data_file, sep=r'\s+', comment='#', header=None, 
            names=['Epoch', 'MAE_Train', 'MAE_Val'], engine='python'
        )
        
        if datos.shape[1] != 3:
            print("Error: El archivo no tiene 3 columnas (Epoch, MAE_Train, MAE_Val).")
            return
        
        datos[['MAE_Train', 'MAE_Val']] *= 1000.0 / 222
        
        plt.figure(figsize=(6, 4))
        plt.plot(datos['Epoch'], datos['MAE_Train'], label='MAE_Train')
        plt.plot(datos['Epoch'], datos['MAE_Val'], label='MAE_Val')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (meV)')
        plt.title('Epoch vs Mean Squared Error')
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True)
        plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

    def grafica_ref_vs_inf_energy(
        path_data_file="analysis/refence_energy_vs_inference_energy.data",
        output_path_tif="analysis/ref_vs_inf_energy.tiff",
        xlim=None,
        ylim=None
    ):
        os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)
        
        datos = pd.read_csv(
            path_data_file, sep=r'\s+', header=0, engine='python'
        )
        
        if datos.shape[1] != 2:
            print("Error: El archivo no tiene exactamente 2 columnas.")
            return
        
        datos.columns = ['Reference Energy (eV)', 'Inference Energy (eV)']
        
        if xlim is None or ylim is None:
            xy_min = min(datos['Reference Energy (eV)'].min(), datos['Inference Energy (eV)'].min())
            xy_max = max(datos['Reference Energy (eV)'].max(), datos['Inference Energy (eV)'].max())
            xlim = xlim or (xy_min, xy_max)
            ylim = ylim or (xy_min, xy_max)
        
        plt.figure(figsize=(6, 6))
        plt.scatter(datos['Reference Energy (eV)'], datos['Inference Energy (eV)'])
        plt.xlabel('Reference Energy (eV)')
        plt.ylabel('Inference Energy (eV)')
        plt.title('Reference Energy vs Inference Energy')
        plt.grid(True)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

    def grafica_ref_vs_inf_energy_hist_2d(
        path_data_file="analysis/refence_energy_vs_inference_energy.data",
        output_path_tif="analysis/ref_vs_inf_energy_hist_2d.tiff",
        scale_factor=222,
        bins=50,
        xlim=None,
        ylim=None
    ):
        os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)
        
        datos = pd.read_csv(
            path_data_file, sep=r'\s+', header=0, engine='python'
        )
        
        if datos.shape[1] != 2:
            print("Error: El archivo no tiene exactamente 2 columnas.")
            return
        
        datos.columns = ['Reference Energy (eV)', 'Inference Energy (eV)']
        
        datos['Reference Energy (eV)'] /= scale_factor
        datos['Inference Energy (eV)'] /= scale_factor
        
        if xlim is None or ylim is None:
            xy_min = min(datos['Reference Energy (eV)'].min(), datos['Inference Energy (eV)'].min())
            xy_max = max(datos['Reference Energy (eV)'].max(), datos['Inference Energy (eV)'].max())
            xlim = xlim or (xy_min, xy_max)
            ylim = ylim or (xy_min, xy_max)
        
        hist, xedges, yedges = np.histogram2d(
            datos['Reference Energy (eV)'], datos['Inference Energy (eV)'],
            bins=bins, range=[xlim, ylim], density=True
        )
        
        hist_filtered = np.where(hist != 0, hist, np.nan)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(hist_filtered.T, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                cmap='Blues', origin='lower', aspect='auto')
        plt.colorbar(label='Density')
        plt.xlabel('Reference Energy (eV)')
        plt.ylabel('Inference Energy (eV)')
        plt.title('Reference Energy vs Inferred Energy (Density)')
        plt.grid(True)
        plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

    def grafica_ref_vs_inf_energy_hist_3d(
        path_data_file="analysis/refence_energy_vs_inference_energy.data",
        output_path_tif="analysis/ref_vs_inf_energy_hist_3d.tiff",
        scale_factor=222,
        bins=50
    ):
        os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)
        
        datos = pd.read_csv(
            path_data_file, sep=r'\s+', header=0, engine='python'
        )
        
        if datos.shape[1] != 2:
            print("Error: El archivo no tiene exactamente 2 columnas.")
            return
        
        datos.columns = ['Reference Energy (eV)', 'Inference Energy (eV)']
        
        datos['Reference Energy (eV)'] /= scale_factor
        datos['Inference Energy (eV)'] /= scale_factor
        
        xy_min = min(datos['Reference Energy (eV)'].min(), datos['Inference Energy (eV)'].min())
        xy_max = max(datos['Reference Energy (eV)'].max(), datos['Inference Energy (eV)'].max())
        
        hist, xedges, yedges = np.histogram2d(
            datos['Reference Energy (eV)'], datos['Inference Energy (eV)'],
            bins=bins, range=[[xy_min, xy_max], [xy_min, xy_max]], density=True
        )
        
        non_zero_indices = hist != 0
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos[non_zero_indices]
        ypos = ypos[non_zero_indices]
        dz = hist[non_zero_indices]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_trisurf(xpos.ravel(), ypos.ravel(), dz.ravel(), cmap='Blues')
        
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        
        ax.set_xlabel('DFT Energy (eV)')
        ax.set_ylabel('TorchMD-Net Energy (eV)')
        ax.set_zlabel('Density')
        ax.set_title('Reference Energy vs Inferred Energy (Density Surface)')
        
        plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

    def grafica_ref_vs_inf_energy_hist(
        path_data_file="analysis/refence_energy_vs_inference_energy.data",
        output_path_tif="analysis/ref_vs_inf_energy_hist.tiff",
        scale_factor=222,
        bins=50
    ):
        os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)
        
        datos = pd.read_csv(
            path_data_file, sep=r'\s+', header=0, engine='python'
        )
        
        if datos.shape[1] != 2:
            print("Error: El archivo no tiene exactamente 2 columnas.")
            return
        
        datos.columns = ['Reference Energy (eV)', 'Inference Energy (eV)']
        
        datos['Reference Energy (eV)'] /= scale_factor
        datos['Inference Energy (eV)'] /= scale_factor
        
        plt.figure(figsize=(8, 6))
        plt.hist(datos['Reference Energy (eV)'], bins=bins, edgecolor='red', alpha=0.5, label='Reference', align='mid', density=True, color='red')
        plt.hist(datos['Inference Energy (eV)'], bins=bins, edgecolor='green', alpha=0.5, label='Inference', align='mid', density=True, color='green')
        
        plt.xlabel('Energy (eV)')
        plt.ylabel('Density')
        plt.title('Reference and Inference Energy Distribution')
        plt.legend()
        
        plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

    def grafica_ref_vs_inf_energy_abs_dif(
        path_data_file="analysis/refence_energy_vs_inference_energy.data",
        output_path_tif="analysis/ref_vs_inf_energy_abs_dif.tiff",
        scale_factor=222,
        bins=50
    ):
        os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)
        
        datos = pd.read_csv(
            path_data_file, sep=r'\s+', header=0, engine='python'
        )
        
        if datos.shape[1] != 2:
            print("Error: El archivo no tiene exactamente 2 columnas.")
            return
        
        datos.columns = ['Reference Energy (eV)', 'Inference Energy (eV)']
        
        datos['Reference Energy (eV)'] /= scale_factor
        datos['Inference Energy (eV)'] /= scale_factor
        
        abs_diff = np.abs(datos['Reference Energy (eV)'] - datos['Inference Energy (eV)'])
        
        plt.figure(figsize=(8, 6))
        plt.hist(abs_diff, bins=bins, edgecolor='black', density=True)
        plt.xlabel('Difference between the reference energy and the energy inferred (eV)')
        plt.ylabel('Density')
        plt.title('Absolute Energy Difference Distribution')
        plt.grid(True)
        
        plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

    def graficas_ref_vs_inf_forces_angles_abs_dif(
        path_data_folder="analysis",
        output_folder="analysis",
        bins=200,
        xlim=(0,45),
        ylim=(0, 5)
    ):

        os.makedirs(output_folder, exist_ok=True)
        
        archivos = glob.glob(os.path.join(path_data_folder, "*.data"))

        for path_data_file in archivos:
            nombre_archivo = os.path.basename(path_data_file)
            elemento = None

            # Intentar extraer el nombre del elemento a partir del archivo
            for num, (nombre, simbolo) in ElementoQuimico.tabla_elementos.items():
                if f"_{simbolo}" in nombre_archivo or f"_{nombre}" in nombre_archivo:
                    elemento = ElementoQuimico(num)
                    break
            
            if not elemento:
                continue  # Si no se reconoce el elemento, omitir el archivo

            output_path_tif = os.path.join(output_folder, f"ref_vs_inf_forces_angles_abs_dif_{elemento.nombre_str}.tiff")
            os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)

            datos = pd.read_csv(path_data_file, header=None, skiprows=1, delim_whitespace=True)
            hist, bin_edges = np.histogram(datos.values.flatten(), bins=bins, density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.figure(figsize=(8, 6))
            plt.scatter(bin_centres, hist * 100)

            plt.xlabel('Angle between reference and inference forces (degrees)')
            plt.ylabel('Frequency of occurrence (%)')
            plt.title(f'Angles Distribution ({elemento.nombre_str})')

            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)

            plt.grid(True)
            plt.tight_layout()

            plt.savefig(output_path_tif, dpi=600, bbox_inches="tight", format="tiff")
            plt.close()

    def graficas_ref_vs_inf_forces_modules(
        path_data_folder="analysis",
        output_folder="analysis",
        xlim=(0, 12),
        ylim=(0, 12)
    ):

        os.makedirs(output_folder, exist_ok=True)
        
        archivos = glob.glob(os.path.join(path_data_folder, "forces_modules_*.data"))

        for path_data_file in archivos:
            nombre_archivo = os.path.basename(path_data_file)
            elemento = None

            # Extraer el sufijo del archivo (_H, _O, _Al) para identificar el elemento
            for num, (nombre, simbolo) in ElementoQuimico.tabla_elementos.items():
                if nombre_archivo.endswith(f"_{simbolo}.data"):
                    elemento = ElementoQuimico(num)
                    break
            
            if not elemento:
                continue  # Si no se reconoce el elemento, omitir el archivo

            output_path_tif = os.path.join(output_folder, f"ref_vs_inf_forces_modules_{elemento.nombre_str}.tiff")
            os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)

            datos = pd.read_csv(path_data_file, sep='\s+', header=0, engine='python')
            datos.columns = ['Reference modules (eV/$\AA$)', 'Inference modules (eV/$\AA$)']

            plt.figure(figsize=(8, 6))
            plt.scatter(datos.iloc[:, 0], datos.iloc[:, 1], s=25)

            plt.xlabel('Reference modules (eV/$\AA$)')
            plt.ylabel('Inference modules (eV/$\AA$)')
            plt.title(f'Reference modules vs Inference modules ({elemento.nombre_str})')

            plt.xlim(xlim)
            plt.ylim(ylim)

            plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
            plt.close()

    def graficas_ref_vs_inf_forces_modules_hist_2d(
        path_data_folder="analysis",
        output_folder="analysis",
        bins=50,
        xlim=(0, 12),
        ylim=(0, 12)
    ):

        os.makedirs(output_folder, exist_ok=True)
        
        archivos = glob.glob(os.path.join(path_data_folder, "forces_modules_*.data"))

        for path_data_file in archivos:
            nombre_archivo = os.path.basename(path_data_file)
            elemento = None

            # Extraer el sufijo del archivo (_H, _O, _Al) para identificar el elemento
            for num, (nombre, simbolo) in ElementoQuimico.tabla_elementos.items():
                if nombre_archivo.endswith(f"_{simbolo}.data"):
                    elemento = ElementoQuimico(num)
                    break
            
            if not elemento:
                continue  # Si no se reconoce el elemento, omitir el archivo

            output_path_tif = os.path.join(output_folder, f"ref_vs_inf_forces_modules_{elemento.nombre_str}_hist_2d.tiff")
            os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)

            datos = pd.read_csv(path_data_file, sep='\s+', header=0, engine='python')

            if datos.shape[1] != 2:
                print(f"Error: El archivo {nombre_archivo} no tiene exactamente 2 columnas.")
                continue
            
            datos.columns = ['Reference modules (eV/$\AA$)', 'Inference modules (eV/$\AA$)']

            hist, xedges, yedges = np.histogram2d(
                datos.iloc[:, 0], datos.iloc[:, 1], bins=bins, density=True
            )
            
            hist_filtered = np.where(hist != 0, hist, np.nan)

            plt.figure(figsize=(8, 6))
            plt.imshow(hist_filtered.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    cmap='Blues', origin='lower', aspect='auto')
            plt.colorbar(label='Density')

            plt.xlabel('Reference modules (eV/$\AA$)')
            plt.ylabel('Inference modules (eV/$\AA$)')
            plt.title(f'Reference modules vs Inference modules ({elemento.nombre_str})')

            plt.xlim(xlim)
            plt.ylim(ylim)

            plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
            plt.close()

    def graficas_ref_vs_inf_forces_modules_hist_3d(
        path_data_folder="analysis",
        output_folder="analysis",
        bins=50,
        xlim=(0, 12),
        ylim=(0, 12)
    ):

        os.makedirs(output_folder, exist_ok=True)
        
        archivos = glob.glob(os.path.join(path_data_folder, "forces_modules_*.data"))

        for path_data_file in archivos:
            nombre_archivo = os.path.basename(path_data_file)
            elemento = None

            # Extraer el sufijo del archivo (_H, _O, _Al) para identificar el elemento
            for num, (nombre, simbolo) in ElementoQuimico.tabla_elementos.items():
                if nombre_archivo.endswith(f"_{simbolo}.data"):
                    elemento = ElementoQuimico(num)
                    break
            
            if not elemento:
                continue  # Si no se reconoce el elemento, omitir el archivo

            output_path_tif = os.path.join(output_folder, f"ref_vs_inf_forces_modules_{elemento.nombre_str}_hist_3d.tiff")
            os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)

            datos = pd.read_csv(path_data_file, sep='\s+', header=0, engine='python')

            if datos.shape[1] != 2:
                print(f"Error: El archivo {nombre_archivo} no tiene exactamente 2 columnas.")
                continue
            
            datos.columns = ['Reference modules (eV/$\AA$)', 'Inference modules (eV/$\AA$)']

            hist, xedges, yedges = np.histogram2d(
                datos.iloc[:, 0], datos.iloc[:, 1], bins=bins, density=True
            )

            xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            dz = hist.ravel()

            non_zero_indices = dz != 0
            xpos = xpos[non_zero_indices]
            ypos = ypos[non_zero_indices]
            dz = dz[non_zero_indices]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_trisurf(xpos, ypos, dz, cmap='Blues')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_xlabel('Reference modules (eV/$\AA$)')
            ax.set_ylabel('Inference modules (eV/$\AA$)')
            ax.set_zlabel('Density')
            ax.set_title(f'Reference modules vs Inference modules ({elemento.nombre_str}) (Surface)')

            plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
            plt.close()

    def graficas_ref_vs_inf_forces_modules_hist(
        path_data_folder="analysis",
        output_folder="analysis",
        bins=100,
        xlim=None,
        ylim=None
    ):

        os.makedirs(output_folder, exist_ok=True)
        
        archivos = glob.glob(os.path.join(path_data_folder, "forces_modules_*.data"))

        for path_data_file in archivos:
            nombre_archivo = os.path.basename(path_data_file)
            elemento = None

            # Extraer el sufijo del archivo (_H, _O, _Al) para identificar el elemento
            for num, (nombre, simbolo) in ElementoQuimico.tabla_elementos.items():
                if nombre_archivo.endswith(f"_{simbolo}.data"):
                    elemento = ElementoQuimico(num)
                    break
            
            if not elemento:
                continue  # Si no se reconoce el elemento, omitir el archivo

            output_path_tif = os.path.join(output_folder, f"ref_vs_inf_forces_modules_{elemento.nombre_str}_hist.tiff")
            os.makedirs(os.path.dirname(output_path_tif), exist_ok=True)

            datos = pd.read_csv(path_data_file, sep='\s+', header=0, engine='python')

            if datos.shape[1] != 2:
                print(f"Error: El archivo {nombre_archivo} no tiene exactamente dos columnas.")
                continue
            
            datos.columns = ['Reference modules (eV/$\AA$)', 'Inference modules (eV/$\AA$)']

            plt.figure(figsize=(8, 6))
            plt.hist(datos['Reference modules (eV/$\AA$)'], bins=bins, edgecolor='red', alpha=0.5, 
                    label='Reference', align='mid', color='red', density=True)
            plt.hist(datos['Inference modules (eV/$\AA$)'], bins=bins, edgecolor='green', alpha=0.5, 
                    label='Inference', align='mid', color='green', density=True)

            plt.xlabel('Modules (eV/$\AA$)')
            plt.ylabel('Density')
            plt.title(f'Distribution of the module of reference and inferred forces ({elemento.nombre_str})')

            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)

            plt.legend()

            plt.savefig(output_path_tif, dpi=600, bbox_inches='tight', format='tiff')
            plt.close()

class Publicacion:
    """
    Clase para gestionar gráficas con formato de publicación.
    """

    def __init__(self, output_dir="analysis/publicacion"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generar_grafica(self):
        """
        Método para generar y guardar una gráfica con formato de publicación.
        (Aquí se insertarán las funciones de graficado)
        """
        pass  # Reemplazar con la lógica de graficado
