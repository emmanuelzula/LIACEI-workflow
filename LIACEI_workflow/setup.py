from setuptools import setup, find_packages

setup(
    name="LIACEI_workflow",
    version="0.1",
    packages=find_packages(),  # Detectará automáticamente los módulos
    package_dir={"": "."},  # Indica que los paquetes están en la misma carpeta
    install_requires=[],  # Agrega dependencias aquí si es necesario
)