# Trabalho de Oficina de Integração
Código do trabalho de Oficina de Integração do curso de Engenharia de Computação 7º período, UTFPR - PB.

O código deste repositório foi o algoritmo construído para treinar uma rede neural que identificaria alguns comandos de voz.

Primeiramente foi realizado uma captura dos áudios de teste do banco de dados de áudios da Google, amostrados em 16 kHz. Cada um desses áudios foi posteriormente processados pela Transformada de Fourier janelada e acentuados pela função log.

Uma vez definida bem as características, cada áudio processado serviu de entrada para a rede neural MultiLayer Perceptron.

Treinada a rede, copiou-se os pesos e os hiper-parâmetros para uma RaspBerry Pi 3B. Lá um código foi escrito para, cada vez que se apertasse um botão, o Rasp capturasse um áudio de 1s, processasse da mesma forma dos áudios do banco de dados e realizasse a classificação.

Classificado o áudio, eles acenderiam alguns LEDs correspondentes a cada comando de voz.
