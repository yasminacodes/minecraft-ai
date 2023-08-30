# Minecraft AI

## Cómo empezar

### Disclaimers
Antes de empezar a trastear con este proyecto, ten en cuenta un par de cosas:  
* Únicamente se ha testeado este código en Mac, por lo que puede que no funcione en otros sistemas operativos como Windows o Linux. Siéntete libre de adaptarlo si lo necesitas.
* Este código se ha hecho por diversión/aprendizaje y puede que haya cosas que no funcionen como se espera. Si encuentras un error, puedes abrir un issue aquí en GitHub o enviarme un mensaje a mis RRSS (@yasminacodes) para que lo revise.  
* Para poder hacer que la IA juegue y se entrene, necesitas tener alguna versión de Minecraft instalada en tu dispositivo.  
* Actualmente, el código no está preparado para utilizar un modelo pre-entrenado, solamente para entrenar nuevos modelos.  
* El programa espera que mientras que la IA esté jugando no se toque el ratón ni el teclado, ni se cierre o minimice la ventana del Minecraft, por lo que no podrás utilizar tu pc mientras estés ejecutando este código (puedes pararlo en cualquier momento).

### Librerías
Este proyecto utiliza Python 3 (testeado con la 3.9.6) utiliza las siguientes librerías:  
* opencv-python  
* numpy  
* tensorflow  
* keyboard  
* pyautogui  
* random2  
* pytesseract  
* imutils  
* Pillow  
* tk  
* pyobjc-framework-Quartz  
* pyobjc-framework-Cocoa  

Para instalarlas, es recomendable que inicies un entorno virtual de python. Puedes hacerlo ejecutando el siguiente código en un terminal:  
```
pip3 install virtualenv # para instalar el paquete  
cd carpeta/del/proyecto/  # para ir a la carpeta del proyecto
python3 -m venv nombre_del_entorno # para crear el entorno virtual  
source nombre_del_entorno/bin/activate #para activar el entorno virtual  
```

Para instalar las librerías, utiliza el archivo requirements.txt: `pip3 install -r requirements.txt`

### Programa
Una vez que tengas todas las librerías instaladas y con el entorno virtual lanzado (en caso de que lo hayas creado), simplemente ejecuta el archivo main escribiendo en un terminal:
```
cd carpeta/del/proyecto/
sudo python3 main.py # El código necesita permisos para temas de accesibilidad, como capturar la pantalla o utilizar el teclado/ratón
```

Tras lanzar el programa, tardará unos momentos en configurar los parámetros iniciales del modelo y, después, emitirá un mensaje indicando que se debe seleccionar la ventana que vamos a capturar, en este caso la del Minecraft con la partida ya abierta (hacemos click encima de la ventana con el ratón para seleccionarla) y pulsamos la tecla c en nuestro teclado para que el programa empiece a capturar frames.  

A partir de aquí, no debemos cerrar ni minimizar la ventana del minecraft ni tocar el teclado/ratón hasta que el programa termine. Si queremos cerrar el programa de forma anticipada, podemos hacerlo seleccionando el terminal en el que se haya lanzado y pulsando CTRL+C.

## Un ko-fi? :)
Espero que el proyecto te resulte interesante! Si es así y te gustaría colaborar para que pueda seguir compartiendo proyectos como este, puedes regalarme un cafecito: ko-fi.com/yasminacodes