class Genericas:
    """
    Clase para gestionar gráficas rápidas de dinámica molecular.
    """

    def __init__(self, output_dir="analysis/genericas"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generar_grafica(self):
        """
        Método para generar y guardar una gráfica rápida.
        (Aquí se insertarán las funciones de graficado)
        """
        pass  # Reemplazar con la lógica de graficado


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
