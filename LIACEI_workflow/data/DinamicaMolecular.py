import h5py
import numpy as np
import os

class DinamicaMolecular:
    """
    Representa una simulación de dinámica molecular con almacenamiento HDF5.
    """
    def __init__(self):
        self.frames = {}  # Diccionario que almacena los frames

    def agregar_frame(self, frame, numero_frame=None):
        """
        Agrega un frame a la simulación, asignando un número de frame si no se proporciona.

        :param frame: Objeto de la clase Frame.
        :param numero_frame: (Opcional) Número asignado al frame. Si no se proporciona,
                             se asignará el número más bajo disponible.
        """
        if not isinstance(frame, Frame):
            raise TypeError("El objeto agregado debe ser una instancia de la clase Frame.")

        # Determinar el número de frame automáticamente si no se proporciona
        if numero_frame is None:
            numero_frame = 0 if not self.frames else max(self.frames) + 1

        # Validar si el número de frame ya existe
        if numero_frame in self.frames:
            raise ValueError(f"Ya existe un frame con el número {numero_frame}.")

        # Agregar el frame al diccionario y actualizar la lista de números de frame
        self.frames[numero_frame] = frame

        # Asignar el número de frame al objeto Frame
        frame.numero_frame = numero_frame 
    
    def cargar_desde_data(input_file):
        """
        Procesa un archivo de entrada de dinámica molecular y almacena los datos en una base de datos.

        Parámetros:
        input_file (str): Ruta al archivo de entrada a procesar.

        Retorna:
        DinamicaMolecular: Instancia de la base de datos con los frames procesados.
        """
        HARTREE_TO_EV=27.211386245981 #Original: 27.21138602
        HA_BO_TO_EV_ANGS=51.4220675112 #Original: 51.422067137
        BO_TO_ANGS=0.529177210544
        
        # Crear la base de datos de dinámica molecular
        base_de_datos = DinamicaMolecular()

        # Leer el archivo línea por línea
        with open(input_file, "r") as archivo:
            for linea in archivo:
                # Convertir línea de texto a una lista Python
                linea = linea.split()

                # Detectar el inicio de un frame
                if linea[0] == "begin":
                    frame_actual = Frame()  # Crear un nuevo frame vacío

                # Procesar información del átomo
                elif linea[0] == "atom":
                    # Convertir valores de posición y fuerza a float
                    x, y, z = map(float, linea[1:4])
                    fx, fy, fz = map(float, linea[7:10])

                    # Crear instancias de los elementos relacionados con el átomo
                    elemento = ElementoQuimico(linea[4])  # Crear el objeto ElementoQuimico
                    posicion = Posicion(x, y, z) * BO_TO_ANGS
                    fuerza = Fuerza(fx, fy, fz) * HA_BO_TO_EV_ANGS

                    # Agregar el átomo al frame actual
                    frame_actual.agregar_atomo(elemento, posicion, fuerza)

                # Procesar información de la energía
                elif linea[0] == "energy":
                    frame_actual.energia = float(linea[1]) * HARTREE_TO_EV

                # Finalizar el frame actual
                elif linea[0] == "end":
                    # Agregar el frame a la base de datos
                    base_de_datos.agregar_frame(frame_actual)

        return base_de_datos  # Devuelve la base de datos con los frames cargados

    def guardar_a_hdf5(dinamica_molecular, archivo):
        """
        Guarda una instancia de DinamicaMolecular en un archivo HDF5.

        :param dinamica_molecular: Objeto DinamicaMolecular a guardar.
        :param archivo: Nombre del archivo HDF5.
        """
        with h5py.File(archivo, 'w') as f:
            for numero_frame, frame in dinamica_molecular.frames.items():
                grupo_frame = f.create_group(f"frame_{numero_frame}")
                grupo_frame.attrs['energia'] = frame.energia
                grupo_frame.create_dataset('posiciones', data=frame.posiciones, compression="gzip")
                grupo_frame.create_dataset('fuerzas', data=frame.fuerzas, compression="gzip")
                grupo_frame.create_dataset('elementos', data=frame.elementos, compression="gzip")

    def guardar_a_xyz(dinamica_molecular,archivo):
        """
        Guarda una instancia de DinamicaMolecular en un archivo XYZ.

        :param dinamica_molecular: Objeto DinamicaMolecular a guardar.
        :param archivo: Nombre del archivo XYZ.
        """
        with open(archivo, "a") as file:
            for frame in dinamica_molecular.frames.values():
                muestra=frame.generar_xyz(dinamica_molecular)
                file.write(muestra)

    def cargar_desde_xyz(archivo_xyz):
        """
        Carga una instancia de DinamicaMolecular desde un archivo XYZ.

        :param archivo_xyz: Ruta del archivo XYZ.
        :return: Objeto DinamicaMolecular reconstruido.
        """
        dinamica_molecular = DinamicaMolecular()
        
        with open(archivo_xyz, "r") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            # Leer el número de átomos
            numero_atomos = int(lines[i].strip())
            i += 1

            # Leer el encabezado con la energía
            encabezado = lines[i].strip()
            if "MD_Step" in encabezado:
                energia = float(encabezado.split("=")[1].strip())
            else:
                raise ValueError("Formato de encabezado no reconocido en el archivo XYZ.")
            i += 1

            # Crear un nuevo frame
            frame = Frame(energia=energia)

            # Leer los datos de los átomos
            for _ in range(numero_atomos):
                partes = lines[i].split()
                elemento = int(partes[0])
                posicion = Vector(float(partes[1]), float(partes[2]), float(partes[3]))
                fuerza = Vector(float(partes[4]), float(partes[5]), float(partes[6]))

                # Agregar el átomo al frame
                frame.agregar_atomo(ElementoQuimico(elemento), posicion, fuerza)
                i += 1

            # Agregar el frame a la DinamicaMolecular
            dinamica_molecular.agregar_frame(frame)

        return dinamica_molecular


    def cargar_desde_hdf5(archivo):
        """
        Carga una instancia de DinamicaMolecular desde un archivo HDF5.
        """
        dinamica_molecular = DinamicaMolecular()

        with h5py.File(archivo, 'r') as f:
            for nombre_frame in f.keys():
                grupo_frame = f[nombre_frame]
                energia = grupo_frame.attrs['energia']
                posiciones = grupo_frame['posiciones'][:]
                fuerzas = grupo_frame['fuerzas'][:]
                elementos = grupo_frame['elementos'][:]

                # Reconstruir el frame
                frame = Frame(energia=energia)
                frame.posiciones = posiciones
                frame.fuerzas = fuerzas
                frame.elementos = elementos.tolist()

                # Extraer el número de frame
                numero_frame = int(nombre_frame.split('_')[1])
                dinamica_molecular.agregar_frame(frame, numero_frame)

        # Ordenar el diccionario por sus claves
        dinamica_molecular.frames = dict(sorted(dinamica_molecular.frames.items()))

        return dinamica_molecular
    

    
    def subconjunto(dinamica_molecular, indices):
        """
        Obtiene un subconjunto de una instancia de DinamicaMolecular basado en los índices proporcionados.

        :param dinamica_molecular: Objeto DinamicaMolecular del cual se extraerá el subconjunto.
        :param indices: Lista o array de índices de los frames a incluir en el subconjunto.
        :return: Una nueva instancia de DinamicaMolecular que contiene únicamente los frames especificados por los índices.
        """
        # Crear una nueva instancia de DinamicaMolecular para el subconjunto
        dinamica_subconjunto = DinamicaMolecular()

        # Iterar sobre los índices y agregar los frames correspondientes al subconjunto
        for indice in indices:
            if indice in dinamica_molecular.frames:
                frame = dinamica_molecular.frames[indice]
                dinamica_subconjunto.agregar_frame(frame, numero_frame=indice)
            else:
                print(f"Advertencia: El índice {indice} no existe en la dinámica molecular original.")

        return dinamica_subconjunto
            
    @property
    def numero_frames(self):
        """
        Devuelve el número de frames en la Dinamica Molecular.
        """
        return len(self.frames)
    
class Frame:
    """
    Representa un Frame optimizado para NumPy.
    """
    def __init__(self, energia=0.0):
        self.energia = energia
        self.posiciones = np.empty((0, 3), dtype=np.float32)  # Almacena posiciones como matriz
        self.fuerzas = np.empty((0, 3), dtype=np.float32)  # Almacena fuerzas como matriz
        self.elementos = []  # Lista de números atómicos (más eficiente que objetos completos)
        self.numero_frame: int = None

    def agregar_atomo(self, elemento, posicion, fuerza):
        """
        Agrega un átomo al frame, almacenando sus propiedades en estructuras optimizadas.
        """
        if not isinstance(elemento, ElementoQuimico):
            raise TypeError("Elemento debe ser una instancia de la clase ElementoQuimico.")
        if not isinstance(posicion, Vector):
            raise TypeError("Posición debe ser una instancia de la clase Vector.")
        if not isinstance(fuerza, Vector):
            raise TypeError("Fuerza debe ser una instancia de la clase Vector.")

        # Agregar propiedades a las estructuras del frame
        self.elementos.append(elemento.numero)  # Guardar el número atómico
        self.posiciones = np.vstack([self.posiciones, posicion.componentes])
        self.fuerzas = np.vstack([self.fuerzas, fuerza.componentes])

    def obtener_numero_frame(self, dinamica_molecular):
        """
        Devuelve el número de frame asociado a este frame en una Dinámica Molecular.

        :param dinamica_molecular: Objeto DinamicaMolecular que contiene los frames.
        :return: Número del frame asociado a este frame, o None si no se encuentra.
        """
        for clave, valor in dinamica_molecular.frames.items():
            if valor == self:
                return clave
        return None  # Devuelve None si no se encuentra el valor

    def imprimir_xyz(self, dinamica_molecular=None):
        """
        Imprime la posición, el elemento, la fuerza de cada átomo y la energía del frame.

        :param dinamica_molecular: Objeto DinamicaMolecular que contiene los frames.
        """
        # Determinar el número de frame
        if dinamica_molecular is None:
            numero_frame = "None"
        else:
            numero_frame = self.obtener_numero_frame(dinamica_molecular)

        # Imprimir información del frame
        print(f"Número de átomos: {self.numero_atomos}")
        print(f"MD_Step: {numero_frame} Total_energy = {self.energia:.10f}")
        for i, elemento in enumerate(self.elementos):
            posicion = self.posiciones[i]
            fuerza = self.fuerzas[i]
            print("{:>2}{:>20.10f}{:>20.10f}{:>20.10f}{:>20.10f}{:>20.10f}{:>20.10f}".format(
                elemento,
                posicion[0], posicion[1], posicion[2],
                fuerza[0], fuerza[1], fuerza[2]
            ))

    def generar_xyz(self, dinamica_molecular=None):
        """
        Genera un string con la posición, el elemento, la fuerza de cada átomo y la energía del frame
        en formato XYZ.

        :param dinamica_molecular: Objeto DinamicaMolecular que contiene los frames.
        :return: String con la representación en formato XYZ.
        """
        # Determinar el número de frame
        if dinamica_molecular is None:
            numero_frame = "None"
        else:
            numero_frame = self.obtener_numero_frame(dinamica_molecular)

        # Crear el string para el frame
        resultado = []
        resultado.append(str(self.numero_atomos))
        resultado.append(f"MD_Step: {numero_frame} Total_energy = {self.energia:.10f}")
        for i, elemento in enumerate(self.elementos):
            posicion = self.posiciones[i]
            fuerza = self.fuerzas[i]
            resultado.append("{:>2}{:>20.10f}{:>20.10f}{:>20.10f}{:>20.10f}{:>20.10f}{:>20.10f}".format(
                elemento,
                posicion[0], posicion[1], posicion[2],
                fuerza[0], fuerza[1], fuerza[2]
            ))
        resultado.append("")

        # Unir el resultado en un único string con saltos de línea
        return "\n".join(resultado)


    @property
    def atomos(self):
        """
        Devuelve una lista de diccionarios que representan los átomos del frame.
        Cada diccionario incluye el número atómico, posición y fuerza.
        """
        return [
            {
                "numero_atomico": numero_atomico,
                "posicion": self.posiciones[i],
                "fuerza": self.fuerzas[i],
            }
            for i, numero_atomico in enumerate(self.elementos)
        ]
    
    @property
    def numero_atomos(self):
        """
        Devuelve el número de átomos en el frame.
        """
        return len(self.elementos)
    
    
class ElementoQuimico:
    """
    Representa un elemento químico con su número atómico, nombre y símbolo.
    """
    tabla_elementos = {
        1: ("Hydrogen", "H"),
        8: ("Oxygen", "O"),
        13: ("Aluminium", "Al"),
    }

    energias_atomicas = {
        1: -0.500272784191 * 27.21138602,   # Energía atómica del Hidrógeno
        8: -74.9555225243 * 27.21138602,   # Energía atómica del Oxígeno
        13: -242.365764213 * 27.21138602,  # Energía atómica del Aluminio
    }

    def __init__(self, representacion):
        """
        Inicializa un objeto de ElementoQuimico basado en número atómico, nombre o símbolo.

        :param representacion: Puede ser un número atómico (int), nombre (str) o símbolo (str).
        """
        if isinstance(representacion, int):
            self.numero = representacion
            self.nombre_str, self.simbolo_str = ElementoQuimico.tabla_elementos[representacion]
        elif isinstance(representacion, str):
            for num, (nombre, simbolo) in ElementoQuimico.tabla_elementos.items():
                if representacion.lower() in [nombre.lower(), simbolo.lower()]:
                    self.numero = num
                    self.nombre_str, self.simbolo_str = nombre, simbolo
                    break
            else:
                raise ValueError(f"Representación '{representacion}' no reconocida.")

    def energia_atomica(self):
        """
        Devuelve la energía atómica del elemento.

        :return: Energía atómica en unidades correspondientes, o None si no está definida.
        """
        return ElementoQuimico.energias_atomicas.get(self.numero, None)

    def __repr__(self):
        return (f"ElementoQuimico(numero={self.numero}, "
                f"nombre='{self.nombre_str}', simbolo='{self.simbolo_str}')")

class Vector:
    """
    Representa un vector genérico en 3D.
    """
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.componentes = np.array([x, y, z], dtype=np.float32)

    def __mul__(self, escalar):
        if not isinstance(escalar, (int, float)):
            raise TypeError("La multiplicación debe ser con un escalar.")
        return Vector(*(self.componentes * escalar))

    def __truediv__(self, escalar):
        if not isinstance(escalar, (int, float)):
            raise TypeError("La división debe ser con un escalar.")
        return Vector(*(self.componentes / escalar))
    
    def magnitud(self):
        """
        Calcula la magnitud (norma) del vector.
        
        Returns:
            float: Magnitud del vector.
        """
        from numpy.linalg import norm
        return norm(self.componentes)

    def angulo(v1, v2):
        """
        Calcula el ángulo en grados entre dos vectores.

        Args:
            v1 (array-like): Primer vector.
            v2 (array-like): Segundo vector.

        Returns:
            float: Ángulo en grados entre los dos vectores.
        """
        from math import acos, degrees
        from numpy.linalg import norm
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)

        producto_punto = np.dot(v1, v2)
        norma_v1 = norm(v1)
        norma_v2 = norm(v2)
        
        # Evitar divisiones por cero
        if norma_v1 == 0 or norma_v2 == 0:
            raise ValueError("Uno de los vectores es nulo y no se puede calcular el ángulo.")

        theta_radianes = acos(producto_punto / (norma_v1 * norma_v2))
        return degrees(theta_radianes)  # Convertir de radianes a grados



class Posicion(Vector):
    """Representa una posición en 3D."""


class Fuerza(Vector):
    """Representa una fuerza en 3D."""

class Utils:

    def comparar_frames(referencias, inferencias):
        """
        Compara los elementos de los frames en dos estructuras de datos.

        Args:
            referencias: Objeto que contiene frames con elementos de referencia.
            inferencias: Objeto que contiene frames con elementos inferidos.

        Raises:
            ValueError: Si las estructuras tienen diferente número de frames o si hay diferencias en los elementos.
        """
        # Verificar que ambos tienen la misma cantidad de frames
        if len(referencias.frames) != len(inferencias.frames):
            raise ValueError("Error: Los arreglos tienen diferente número de frames.")

        # Recorrer los frames comparando los arrays directamente
        for index_frame in referencias.frames.keys():
            elementos_referencias = referencias.frames[index_frame].elementos
            elementos_inferencias = inferencias.frames[index_frame].elementos

            # Comparar los arrays directamente (manteniendo el orden)
            if elementos_referencias != elementos_inferencias:
                raise ValueError(f"Error: Diferencia detectada en el frame {index_frame}.")

    def obtener_elementos_unicos(referencias, inferencias):
        """
        Obtiene una lista de elementos únicos a partir de los frames en referencias e inferencias.

        Args:
            referencias: Objeto que contiene frames con elementos de referencia.
            inferencias: Objeto que contiene frames con elementos inferidos.

        Returns:
            list: Lista de elementos únicos combinados de ambas estructuras.
        """
        # Crear conjuntos vacíos para almacenar elementos únicos
        elementos_referencias = set()
        elementos_inferencias = set()

        # Recorrer todos los frames en referencias e inferencias y extraer elementos únicos
        for frame in referencias.frames.values():
            elementos_referencias.update(frame.elementos)

        for frame in inferencias.frames.values():
            elementos_inferencias.update(frame.elementos)

        # Unir los conjuntos y devolver la lista de elementos únicos
        return list(elementos_referencias | elementos_inferencias)

    def calcular_angulos_y_guardar(referencias, inferencias, output_dir="analysis"):
        """
        Calcula los ángulos entre fuerzas en referencias e inferencias y los guarda en archivos por tipo de elemento.

        Args:
            referencias: Objeto que contiene frames con datos de referencia.
            inferencias: Objeto que contiene frames con datos inferidos.
            output_dir (str, opcional): Directorio donde se guardarán los archivos de salida. Por defecto "analysis".

        Returns:
            None. Escribe los resultados en archivos de texto por elemento.
        """
        import os

        # Obtener elementos únicos de ambas estructuras
        elementos_unicos = Utils.obtener_elementos_unicos(referencias, inferencias)

        # Asegurar que el directorio de salida existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear los archivos con el encabezado antes de agregar datos
        for elemento in elementos_unicos:
            with open(f"{output_dir}/forces_angles_{ElementoQuimico(elemento).simbolo_str}.data", "w") as archivo:
                archivo.write("theta(grados)\n")

        # Diccionario para almacenar datos antes de escribir en archivos
        datos_por_elemento = {}

        # Iterar sobre cada frame en inferencias
        for key, frame_inferencia in inferencias.frames.items():
            frame_referencia = referencias.frames[key]  # Acceder al frame de referencia

            # Iterar sobre los átomos del frame
            for index, atomo in enumerate(frame_inferencia.atomos):
                elemento = ElementoQuimico(atomo['numero_atomico']).simbolo_str

                f_inferencia = atomo['fuerza']
                f_referencia = frame_referencia.atomos[index]['fuerza']
                angulo = Vector.angulo(f_inferencia, f_referencia)

                # Almacenar en un diccionario por tipo de elemento
                if elemento not in datos_por_elemento:
                    datos_por_elemento[elemento] = []
                datos_por_elemento[elemento].append(f"{angulo}\n")

        # Escribir todos los datos en archivos con un buffer de 1MB para evitar bloqueos de I/O
        for elemento, datos in datos_por_elemento.items():
            with open(f"{output_dir}/forces_angles_{elemento}.data", "a", buffering=1024*1024) as archivo:
                archivo.writelines(datos)  # Escribe toda la lista de una sola vez

    import os

    def calcular_modulos_fuerza_y_guardar(referencias, inferencias, output_dir="analysis"):
        """
        Calcula la magnitud de las fuerzas en referencias e inferencias y las guarda en archivos por tipo de elemento.

        Args:
            referencias: Objeto que contiene frames con datos de referencia.
            inferencias: Objeto que contiene frames con datos inferidos.
            output_dir (str, opcional): Directorio donde se guardarán los archivos de salida. Por defecto "analysis".

        Returns:
            None. Escribe los resultados en archivos de texto por elemento.
        """
        # Obtener elementos únicos de ambas estructuras
        elementos_unicos = Utils.obtener_elementos_unicos(referencias, inferencias)

        # Asegurar que el directorio de salida existe
        os.makedirs(output_dir, exist_ok=True)

        # Crear los archivos con el encabezado antes de agregar datos
        for elemento in elementos_unicos:
            simbolo_elemento = ElementoQuimico(elemento).simbolo_str
            with open(f"{output_dir}/forces_modules_{simbolo_elemento}.data", "w") as archivo:
                archivo.write("f_reference(eV/A)\t f_inference(eV/A)\n")

        # Diccionario para almacenar datos antes de escribir en archivos
        datos_por_elemento = {}

        # Iterar sobre cada frame en inferencias
        for key, frame_inferencia in inferencias.frames.items():
            frame_referencia = referencias.frames[key]  # Acceder al frame de referencia

            # Iterar sobre los átomos del frame
            for index, atomo in enumerate(frame_inferencia.atomos):
                simbolo_elemento = ElementoQuimico(atomo['numero_atomico']).simbolo_str

                # Extraer fuerzas de inferencia y referencia
                fx, fy, fz = atomo['fuerza']
                f_inferencia_magnitud = Fuerza.magnitud(Fuerza(fx, fy, fz))

                fx, fy, fz = frame_referencia.atomos[index]['fuerza']
                f_referencia_magnitud = Fuerza.magnitud(Fuerza(fx, fy, fz))

                # Almacenar en un diccionario por tipo de elemento
                if simbolo_elemento not in datos_por_elemento:
                    datos_por_elemento[simbolo_elemento] = []
                datos_por_elemento[simbolo_elemento].append(f"{f_referencia_magnitud}\t{f_inferencia_magnitud}\n")

        # Escribir todos los datos en archivos de manera eficiente
        for elemento, datos in datos_por_elemento.items():
            with open(f"{output_dir}/forces_modules_{elemento}.data", "a", buffering=1024*1024) as archivo:
                archivo.writelines(datos)  # Escribe toda la lista en una sola operación

    def guardar_energia_referencia_vs_inferencia(referencias, inferencias, output_path="analysis/refence_energy_vs_inference_energy.data"):
        """
        Guarda la comparación de energía entre referencias e inferencias en un archivo.

        Args:
            referencias: Objeto que contiene frames con energía de referencia.
            inferencias: Objeto que contiene frames con energía inferida.
            output_path (str, opcional): Ruta del archivo donde se guardarán los datos. Por defecto "analysis/refence_energy_vs_inference_energy.data".

        Returns:
            None. Escribe los resultados en un archivo de texto.
        """
        # Asegurar que el directorio de salida existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Lista para acumular los datos antes de escribir en el archivo
        datos = ["energia_referencia\tenergia_inferencia\n"]  # Agregar encabezado

        # Iterar sobre cada frame en inferencias
        for key, frame_inferencia in inferencias.frames.items():
            frame_referencia = referencias.frames[key]  # Acceder al frame de referencia

            energia_referencia = frame_referencia.energia
            energia_inferencia = frame_inferencia.energia

            # Almacenar los datos en la lista
            datos.append(f"{energia_referencia}\t{energia_inferencia}\n")

        # Escribir todos los datos en el archivo de manera eficiente
        with open(output_path, "w", buffering=1024*1024) as archivo:
            archivo.writelines(datos)  # Escribe toda la lista en una sola operación

    def guardar_diferencia_energia(referencias, inferencias, output_path="analysis/diference_refence_energy_and_inference_energy.data"):
        """
        Calcula y guarda la diferencia de energía entre referencias e inferencias en un archivo.

        Args:
            referencias: Objeto que contiene frames con energía de referencia.
            inferencias: Objeto que contiene frames con energía inferida.
            output_path (str, opcional): Ruta del archivo donde se guardarán los datos. 
                                        Por defecto "analysis/diference_refence_energy_and_inference_energy.data".

        Returns:
            None. Escribe los resultados en un archivo de texto.
        """
        # Asegurar que el directorio de salida existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Lista para acumular los datos antes de escribir en el archivo
        datos = ["reference_energy(eV)\t inference_energy(eV)\t difference(eV)\n"]  # Agregar encabezado

        # Iterar sobre cada frame en inferencias
        for key, frame_inferencia in inferencias.frames.items():
            frame_referencia = referencias.frames[key]  # Acceder al frame de referencia

            energia_referencia = frame_referencia.energia
            energia_inferencia = frame_inferencia.energia
            diferencia = np.abs(energia_referencia - energia_inferencia)

            # Almacenar los datos en la lista
            datos.append(f"{energia_referencia}\t{energia_inferencia}\t{diferencia}\n")

        # Escribir todos los datos en el archivo de manera eficiente
        with open(output_path, "w", buffering=1024*1024) as archivo:
            archivo.writelines(datos)  # Escribe toda la lista en una sola operación
