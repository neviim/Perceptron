import random

''' Classe RedeNeural - perceptron
    ------------------------------

            * Variaveis recebida por parametro:

                saidas_esperadas       - Matrix
                entradas               - Lista
                limiar = -1            - Inteiro
                epocas = 1000          - Inteiro
                aprendizado = 0.1      - Flout

----------------- '''
class RedeNeural:

    def __init__(self, entradas, saidas_esperadas, aprendizado=0.1, limiar=-1, epocas=10000):

        # variaveis de uso publico desta class
        self.pesos = []                             # vetor dos pesos
        self.epocas = epocas                        # número de épocas
        self.limiar = limiar                        # limiar
        self.entradas = entradas                    # todas as entradas
        self.aprendizado = aprendizado              # taxa de aprendizado (entre 0 e 1)
        self.num_entradas = len(entradas)           # numero de elementos de entradas
        self.num_elemento = len(entradas[0])        # numero de elementos de cada entrada
        self.saidas_esperadas = saidas_esperadas    # saídas respectivas de cada entrada

    # metodo para teinar a rede
    def treinar_a_rede(self):

        # adiciona -1 para cada entrada
        for amostra in self.entradas:
            amostra.insert(0, -1)

        # inicia o vetor de pesos com valores aleatórios pequenos *
        for i in range(self.num_elemento):
            self.pesos.append(random.random())

        # insere o limiar no vetor de pesos
        self.pesos.insert(0, self.limiar)

        # inicia o contador de épocas
        num_epocas = 0

        # loop, até se atingir o fim do trenamento
        while True:

            # inicialmente erro inexiste
            erro = False

            # para todas as entradas de treinamento
            for i in range(self.num_entradas):

                pa = 0 # potencial de ativação

                # pa = somatoria(i=1 até ne) (wi * xi) - theta
                for item in range(self.num_elemento + 1):
                    pa += self.pesos[item] * self.entradas[i][item]

                # obtém a saída da rede utilizando a função de ativação
                y = self.degrau_bipolar(pa)

                # verifica se a saída da rede é diferente da saída desejada
                if y != self.saidas_esperadas[i]:

                    # calcula o erro: subtração entre a saída desejada e a saída da rede
                    erro_aux = self.saidas_esperadas[i] - y

                    # faz o ajuste dos pesos para cada elemento da amostra
                    for j in range(self.num_elemento + 1):
                        self.pesos[j] = (self.pesos[j] + (self.aprendizado * erro_aux * self.entradas[i][j]))

                    # True = ainda existe erro
                    erro = True

            # incrementa o número de épocas
            num_epocas += 1

            # critério de parada é pelo número de épocas ou se não existir erro
            if num_epocas > self.epocas or not erro:
                break # sai do while

    # metodo a formula para a função degrau
    def degrau(self, pa):
        return 1 if pa >= 0 else 0

    # metodo a formula para a função degrau pipolar (sinal)
    def degrau_bipolar(self, pa):
        return 1 if pa >= 0 else -1

    # metodo para testar a rede
    def teste_da_rede(self, entradas, estado1, estado2):

        # insere o (-1), peso inicial de ajuste.
        entradas.insert(0, -1)

        # usa o vetor de pesos ajustado durante o treinamento da rede
        pa = 0
        for item in range(self.num_elemento + 1):
            pa += self.pesos[item] * entradas[item]

        # calcula a saída da rede
        y = self.degrau_bipolar(pa)
        #print pa

        # verifica a qual estado pertence a resposta
        if y == -1:
            print('A lampada esta %s' % estado1)
        else:
            print('A lampada esta %s' % estado2)
