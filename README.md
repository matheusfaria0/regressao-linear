### Observações:
Implementaremos o algorítmo utilizando a linguagem Python. Por ser uma linguagem de alto nível, sacrificaremos um pouco de eficiência, mas tornaremos nosso código muito mais legível para quem não tem tanta experiência com programação. Caso estivessemos desenolvendo um algorítimo de visão computacional, seria inviável utilizar Python, já que eficiência importa muito nessas aplicações, como em carros autônomos, uma diferença de um segundo entre aplicações pode salvar uma vida no trânsito.

# Regressão linear.
Você certamente já usou algo bem similar a regressão linear em seu dia-dia sem saber, a famosa "regra de três". Nela, tentamos estimar valores que seguem uma proporção definida, por exemplo, suponha que uma xícara custa 10 reais e você decide dar três delas de presente para seus familiares. Você pode facilmente estimar que gastará 30 reais, pois segue uma proporção muito simples.
Na regressão linear, faremos algo similar, buscaremos por uma linha que melhor se encaixa em nossos dados, faremos previsões futuras com base nela.

![alt text](https://github.com/matheusfaria0/regressao-linear/blob/master/download.png "Exemplo de regressão linear.")

Para nós, é extremamente simples traçar uma linha entre pontos, já que temos nossa visão que nos permite fazer isso, mas pense que precisamos implementar isso em um computador, então precisaremos encontrar meios matemáticos de descrever essa reta. Faremos o seguinte:

### Equação de reta:
Toda reta pode ser expressada pela seguinte equação: **y=ax+b**.
 
 Com isso, podemos deixar nossa meta mais clara, precisamos encontrar o melhor valor de "a" e "b" para que nosso erro seja o menor possível (veremos como calcular o erro logo em seguida).
 Para isso, vamos iniciar os coeficientes com valores aleatórios e vamos melhorando eles aos poucos utilizando o método do gradiente de descida (falaremos mais dele mais para frente). Implementaremos o peso aleatório da seguinte forma:

```python
import random   # o pacote random é necessário para gerar números aleatórios.

def randomInitializer():
  '''
  Parâmetros:
    - nenhum parâmetro é necessário.
  
  Retorna:
    - uma lista com valores aleatórios de A e B.
  '''
  
  a = random.random()
  b = random.random()
  return (a,b)
```

### Funções de custo:
Imagine que você tenha começado a cozinhar há pouco tempo e não tem certeza se sua comida está boa ou não, então você decide dar para alguma outra pessoa provar e te dar uma nota. Você saberá se está cozinhando certo ou não dependendo da nota da pessoa. É exatamente isso que as funções de custo fazem, elas dizem o quão bom nosso modelo preditivo é. Existem muitas funções de custo, aqui usaremos a de erro quadrádico médio. Ela é definida pela seguinte fórmula:

![alt text](https://github.com/matheusfaria0/regressao-linear/blob/master/e258221518869aa1c6561bb75b99476c4734108e.svg)

* O primeiro passo é encontrar o quadrado da diferença entre "y" (valor real) e "^y" (valor previsto). Caso nosso modelo seja perfeito, os dois termos serão iguais, logo o MSE será zero. Portanto, quanto mais próximo de zero o MSE, melhor nosso modelo.

*_Ex: Acima previmos que três xícaras custam 30 reais, e é exatamente o que os dados dizem também, logo nossa MSE será zero. Mas imagine que tivessemos previsto que as três xícaras custam 500 reais. Nesse cenário nossa MSE seria extremamente alta._*

* Depois de encontrar o quadrado da diferença de todos os termos, iremos soma-los e dividir pelo número de termos, encontrando a média. Implementaremos da seguninte forma: 

```python
def erroQuadraticoMedio(Y, Y_true):
  '''
  Parâmetros:
    - Y: Lista com  os valores-alvo estimados.
    - Y_true: Lista com  os valores-alvo verdadeiros (corretos).
  
  Retorna:
    - o erro quadratico médio de seu modelo atual.
  '''
  somaDosErros = 0
  for valor in range(len(Y)):
    diferencaEntrePrevisoes = (Y_true[valor] - Y[valor])
    somaDosErros += (diferencaEntrePrevisoes**2)
  media = somaDosErros / len(Y)
  
  return media
```

### Gradiente de descida:
Agora que sabemos se nosso modelo está bom ou ruim, como podemos melhora-lo? Usaremos o método do gradiente de descida, são eles que fazem as inteligências artificiais aprenderem. Imagine que você é um alpinista e está em uma montanha muito alta e deseja chegar ao chão. Para isso você deve olhar ao seu redor e dar pequenos passos na direção que te levar na menor altura. O gradiente de descida faz exatamente isso, onde a altura seria o erro de nosso modelo. Ele é definido pela seguinte fórmula:

![alt text](https://github.com/matheusfaria0/regressao-linear/blob/master/sgd.png)

Em suma, a parte da derivada serve simplesmente para indicar se você deve diminuir ou aumentar o valor dos parâmetros, o gráfico abaixo é de um coeficiente em função do erro, podemos perceber que se ele tivesse movido para a esquerda, seu erro apenas aumentaria, e esse não é nosso objetivo.
O "J" da equação representa a função de custo, ou seja, o erro quadrático médio. Iremos multiplica-lo pela derivada e pelo **learning reate**. Teremos como produto o novo coeficiente, precisaremos atualiza-lo, **mas você precisa atualiza-los simultaneamente**, abaixo falarei mais sobre.

![alt text](https://github.com/matheusfaria0/regressao-linear/blob/master/gradiente.png)

* **_O que é learning rate?_** O learning rate é uma valor extremamente pequeno (normalmente algo em torno de 0.001) que é usado para não modificar drasticamente nosso modelo.

* **_ATUALIZAR PESOS SIMULANEAMENTE:_** Sempre devemos atualizar os pesos simulaneamente, ou seja, imagine que nossa equação de reta atual seja "2x+5=y", notamos que ela não se encaixa bem em nossos dados e passamos ela pelo gradiente de descida, e encontramos que o novo "a" deve ser 2,3. Não devemos atualiza-lo por agora, precisamos calcular o novo "b", e depois atualiza-los simulataneamente.
