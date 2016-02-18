# Perceptron

    Protótipo de um perceptron escrito do zero.

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
                EM que horas do dia normalmente a temperatura utrapassa determinado graus acema da valor X estipulado?

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
          Este processo sera repetido em loop de forma sequencial para as amostras até se obter a saida do perceptron
          iqual a saida desejada de cada amostra.

          O limiar é uma variavem que é sjustada para auxiliar o treinamento do perceptron, ela pode ser implementada
          dentro do vetor de pesos sínapticos.


                        w = (formula para gerar o vetor com o limiar e os pesos sinápticos)

                             Verificar como é montada esta formula!!!!


