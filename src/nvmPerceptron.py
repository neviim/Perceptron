# encoding: utf-8
#
# Nome..: Neviim Jads
# Versão: 0.1
#
# Depos de estar estudando os fundamentos de redeneural a mais de dois anos, creio que esta na hora
# de escrever minha primeira implementação do zero de uma rede Perceptron, vamos ver o que sai...
#
# Descritivo, do como se dara a evolução da escrida deste codigo, teoria e pratica aplicada.
#
# Uma rede perseptron é a base mais simples de se começãr a entender a mecanica de funcionamento de
# uma rede neural artificial, como este codigo é a base para ver o funcionamento na pratica de uma
# base teorica que no inicio sempre é muito complicado se entender, conforme for evoluindo no descritivo
# teorico do como fui entendendo cada atapa do processo vou colocar as referencias teoricas que me foram
# esclarecendo alguns pontos de duvidas do entendimento, ficando assim uma fonde de referencia para pesquisar
# caso algo do que esto fazendo venha não ser compreendido ou não faça sentido do porque esta sendo utilizado.
#
# https://github.com/neviim/Perceptron.git
#
# Rede Neural Artificial
#
#   referencias:
#       http://www.teses.usp.br/teses/disponiveis/18/18153/tde-29042009-102601/publico/Fabriciu.pdf

'''
    Matemática da coisa:


        pa = somatoria(i=1 até ne) (wi * xi) - theta


        (pa)    -> potencial de ativação {diferença do valor entre o combinador linear somatória, e limiar de ativação.
        (ne)    -> numero de entradas.
        (xi)    -> numero das entradas da rede.
        (wi)    -> peso sináptico associado a i-ésima entrada.
        (theta) -> limiar de ativação com representação de theta sendo o valor de (-1)


        - Com o resultado de (pa), passa por parametro para a função de ativação (f) obtendo assim o resultado de (y)

                y = f(pa)

        (f)     -> proposito desta função é limitar a saída do neuronio dentro de um internalo de valores aceitaveis.
        (y)     -> resultado final gerado do neurônio mediante ao conjunto de sinais de entrada determinado.


        - Estas são as definições internas do precessamento interno gerado por um Perceptron.


    Definindo estrutura funcional interna:

        - De acordo com p proble a ser mapeado pode-se definir o numero de sinais de entrada a uma rede perceptron.

        - Sinais de entrada sera representado por (x1, x2, ...), inicialmente composta somente por um neurônio e
          uma saida resultante em (y).

        - Cada entradas (x1, x2, ...) representa o comportamento do processo a ser mapeado, individualmente cada
          entrada sera ponderada polos pesos sinápticos representados por (w1, w2, ...) com o proposito de obter
          um valor de relevancia a importancia de cada um representa.


    Funções de ativação:

        - referencias:
            http://www.telecom.uff.br/~jmarcos/disciplinas/RedesNeurais/Aula02.pdf
            http://www.carolina.unir.br/downloads/1721_aula9_perceptron.pdf
            http://www.maxwell.vrac.puc-rio.br/3509/3509_3.PDF


        - Ha varias funções de ativação que pode ser utilizadas, mais para cada caso no geral precisa ser efetuado
          testes e verificação por resultado obtido, e verificar qual o melhor se adequara aos propósitos do
          resultado que se esta buscando alcançar.

        - As funções que mais tenho encontradas nas literaturas que vejo, são as:

            degrau
            degrau bipolar

            praticamento funções bem simples de ser aplicada.

                degrau, tambem conhecida como função de Theshold:

                    f(pa) = 1, se pa >= 0
                    f(pa) = 0, se pa <  0

                degrau pipolar, tambem conhecida por função sinal:

                    f(pa) =  1, se pa >= 0
                    f(pa) = -1, se pa <  0

        - Já as entradas (xi) podem assumir qualquer valor numéricos

        - Normalmente se itiliza um perceptron para classificação de padrões, isso porque a saida só podera assumir
          dois valores, tipo, 0 ou 1, verdadeito ou falso, exemplo sobre o proposito pratico para o qual no momento
          esto fazendo isso, quero obter uma classificação sobre uma base de dados dada como aprendisado para a rede
          neural um resultado de que:

                Em um periodo predeterminado a definir (uma hora) uma lampada fica por mais tempo ligada ou desligada?
                Em que horas do dia normalmente a temperatura utrapassa determinado graus acema da valor X estipulado?

                Exemplos basicos para se ter uma referencia aplicada do contesto abordado aqui, podendo ser
                representado por momento A e momento B.

                    Fazendo um paralelo:

                        A função degrau ficaria:

                                quando a saida for  0 estaria no momento A
                                quanto a saida for  1 estaria no momento B

                        A função degrau bipolar:

                                quanto a saida for -1 estaria no momento A
                                quando a saida for  1 estaria no momento B

                        estas duas funçoes geram resultantes que podem ser atribuidos as estas caracteristicas
                        representativas destas necessidades esperadas.


                    A formula para a função grau pipolar (sinal) é considerado somente duas entradas, sendo elas.


                        y =  1, se w1 * x1 + w2 * x2 - theta >= 0
                        y = -1, se w1 * x1 + w2 * x2 - theta <  0

                        neste caso o percepton se comporta como um classificador, por esta definição obteremos quando
                        a saida for:

                            -1 que esta no momento A
                             1 que esta no momento B


    Aprendizado, para este caso utilizaremos a regra de HEBB.

        - referencias encontradas em:
            https://pt.wikipedia.org/wiki/Teoria_hebbiana
            http://www2.ica.ele.puc-rio.br/Downloads/30/ICA-Aula-2b-Perceptron.pdf
            https://pt.wikipedia.org/wiki/Sinapse


        - Se a saida gerada for igual a desejada os pesos sinápticos e limiares receberão o incremento, sendo
          proporcionalmete aos valores de entrada, contrario a isso sera decrementados.
          Este processo sera repetido em loop de forma sequencial para as entradas até se obter a saida do perceptron
          iqual a saida desejada de cada amostra.

          O limiar é uma variavem que é sjustada para auxiliar o treinamento do perceptron, ela pode ser implementada
          dentro do vetor de pesos sínapticos.


                        w = (formula para gerar o vetor com o limiar e os pesos sinápticos)

                             Verificar como é montada esta formula!!!!

                             ... terminar de escrever esta etapa.


'''

# variaveis de identidade
__autor__ = "Neviim Jads"
__versao__ = 0.1

# defini os import
import random
import copy


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
        print pa

        # verifica a qual estado pertence a resposta
        if y == -1:
            print('A lampada sera %s' % estado1)
        else:
            print('A lampada sera %s' % estado2)


# inicio de processamento
if __name__ == '__main__':

    # saida esperada para cada item da entrada
    saidas_esperadas = [1, -1, -1, 1]

    # referencia de lista de entradas para a rede perceptron que serão
    # utilizadas nos testes de validação da aprendizagem.

    # entrada que sera aprendica.
    entradas =            [[0.2315, 1,  0.0],    # hora: 23:15:10, dia: 1, status: acender
                           [0.0610, 2, -0.1],    # hora: 06:10:20, dia: 2, status: apagar
                           [0.1220, 3, -0.1],    # hora: 12:20:30, dia: 3, status: apagar
                           [0.1830, 4,  0.0]]    # hora: 18:30:40, dia: 4, status: acender

    # entradas a serem testadas na rede pos aprendizato, simula leitura do estado atual da lampada
    entradas_diferentes = [[0.2315, 1, -0.1],    #
                           [0.0610, 2,  0.0],    #
                           [0.1220, 3,  0.0],    #
                           [0.1830, 4, -0.1]]    #

    # copia identica as entradas que serão aprendidas pela rede.
    entradas_identicas = copy.deepcopy(entradas)

    # cria uma rede perceptron
    rede_perceptron = RedeNeural(entradas=entradas, saidas_esperadas=saidas_esperadas, aprendizado=0.1, epocas=20000)

    # realiza o treinamento da redeNeural
    rede_perceptron.treinar_a_rede()


    # === Testes...


    # testa de validação na rede pos ser treinada com os mesmos itens de entrada do trenamento
    print "#"
    print "Testa mesmas estradas do trenamento."
    for testa_item in entradas_identicas:
        print(testa_item)
        rede_perceptron.teste_da_rede(testa_item, 'apagada.', 'acesa.')
        print


    print ""
    print "#"
    print "Testa outras entradas diferente do trenamento."
    for testa_item in entradas_diferentes:
        print(testa_item)
        rede_perceptron.teste_da_rede(testa_item, 'apagada.', 'acesa.')
        print