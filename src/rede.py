# encoding: utf-8
#
# Nome..: Neviim Jads
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

# variaveis de identidade
__autor__ = "Neviim Jads"
__versao__ = 0.2

# defini os import
import copy
from perceptron import RedeNeural

# inicio de processamento
if __name__ == '__main__':

    # referencia de lista de entradas para a rede perceptron que serão
    # utilizadas nos testes de validação da aprendizagem.

    # entrada que sera aprendica.
    entradas =            [[0.2315, 1,  0.0],    # hora: 23:15:10, dia: 1, status: acender
                           [0.0610, 2, -0.1],    # hora: 06:10:20, dia: 2, status: apagar
                           [0.1220, 3, -0.1],    # hora: 12:20:30, dia: 3, status: apagar
                           [0.1830, 4,  0.0]]    # hora: 18:30:40, dia: 4, status: acender

    # saida esperada para cada item da entrada
    saidas_esperadas = [1, -1, -1, 1]

    # copia identica as entradas que serão aprendidas pela rede.
    entradas_identicas = copy.deepcopy(entradas)

    # cria uma rede perceptron
    rede_perceptron = RedeNeural(entradas=entradas, saidas_esperadas=saidas_esperadas, aprendizado=0.1, epocas=20000)

    # realiza o treinamento da redeNeural
    rede_perceptron.treinar_a_rede()


    # === Testes as resposta da rede já tendo passado pela aprendizagem ...


    # testa de validação na rede pos ser treinada com os mesmos itens de entrada do trenamento
    print ('#')
    print ('Testando a redeneural, usando as mesmas estradas antes do trenamento.')
    for testa_item in entradas_identicas:
        print(testa_item)
        rede_perceptron.teste_da_rede(testa_item, 'apagada.', 'acesa.')
        print ()
