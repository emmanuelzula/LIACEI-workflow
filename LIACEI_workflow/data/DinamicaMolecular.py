import h5py
import numpy as np

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
        1: ("Hidrógeno", "H"),
        8: ("Oxígeno", "O"),
        13: ("Aluminio", "Al"),
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

