# Modelos

Documentação dos modelos servidos pela API. O catálogo vive em
[`app/registry.py`](app/registry.py) e é exposto em `GET /models`.

---

## Por que vários modelos e não um

Um único classificador "universal" parece econômico e é a decisão errada, por
motivos concretos.

**O espaço de rótulos não é o mesmo.** Podridão de colmo, dano de broca e
sintomas radiculares são **fisicamente invisíveis** numa foto aérea de dossel.
Ferrugem do cafeeiro não existe numa foto de milho. Um softmax universal seria
obrigado a distribuir probabilidade entre classes que não podem estar ali —
não existe resposta certa, só ruído confiante.

**A pergunta não é a mesma.** O drone responde *onde* há algo errado. O rover
responde *o que* é. Métricas diferentes: o drone otimiza **recall** (perder um
foco custa caro, um falso positivo custa uma visita do rover); o rover otimiza
**precisão** (é ele que gera a recomendação de manejo).

**A escala de decisão não é a mesma.** O drone decide sobre um talhão, o rover
sobre uma planta.

O que une drone e rover é o pipeline:

```
DRONE  varre a lavoura, barato, frequente
  │    modelo: aerial-canopy  →  mapa de risco por tile
  │    alta sensibilidade, baixa especificidade
  ▼
TASKING  envia o rover só às linhas suspeitas
  │
  ▼
ROVER  lento, caro, alta especificidade
       modelo: rover-understory  →  diagnóstico + patologia baixa
```

O ativo da empresa é essa fusão, não um classificador melhor. E fusão exige que
cada modelo saiba o que **não** consegue responder — exatamente o que um softmax
único destrói.

---

## Catálogo

| Modelo | Cultura | Plataforma | Classes | Estado | Papel |
| --- | --- | --- | --- | --- | --- |
| [`banana-foliar`](#banana-foliar) | Banana | rover / qualquer | 5 | **Treinado** · campo 0,829 · teste 0,977 | Folhas baixas no rover |
| `banana-foliar-cc` | Banana | rover / qualquer | 5 | **Treinado** · campo 0,592 · teste 0,985 | Versão comercial (só CC BY) |
| [`citrus-foliar`](#citrus-foliar) | Citros | rover / qualquer | 12 | **Treinado** · campo 0,969 · teste 0,884 | Copa baixa no rover |
| `citrus-foliar-cc` | Citros | rover / qualquer | 12 | **Treinado** · campo 0,695 · teste 0,874 | Versão comercial (só CC BY) |
| [`leaf-foliar`](#leaf-foliar) | Milho | qualquer | 4 | **Treinado** · teste 0,9555 | Piloto comercial |
| [`leaf-foliar-b`](#leaf-foliar-b) | Milho | qualquer | 4 | **Treinado** · teste 0,9601 | Segunda opinião da dupla validação |
| [`coffee-foliar`](#coffee-foliar) | Café | qualquer | 5 | **Treinado** · teste 0,9499 | Diferencial estratégico |
| [`rover-understory`](#rover-understory) | Milho | rover | 8 | **Em quarentena** · precisa de coleta | Produto principal |
| [`aerial-canopy`](#aerial-canopy) | Milho | drone | 2 | Precisa de coleta | Triagem |

Acurácia sobre conjunto de teste isolado, com TTA. Para banana e citros, o
número a citar é o **campo** — recortes na escala do rover de fotos que nenhum
treino viu ([teste de campo](#teste-de-campo-rover)). Cada número vem com a sua
ressalva na seção do modelo — leia antes de citar.

```bash
curl http://localhost:8000/models        # catálogo + quais estão carregados
```

---

<a id="leaf-foliar"></a>
## 1. `leaf-foliar` — diagnóstico foliar de milho em close

> **É o modelo que sustenta o piloto desta semana.** Dataset público maduro,
> treina em minutos, é o único que roda sem coleta própria.

### O que responde

"Esta folha, ocupando o quadro, tem qual doença foliar?"

Não é um modelo de plataforma — é um modelo de **enquadramento**. Serve ao drone
em voo baixo e ao rover no baixo dossel, desde que a folha preencha a imagem.

### Ficha técnica

| Item | Valor |
| --- | --- |
| Chave na API | `leaf-foliar` |
| Plataforma | `any` |
| Entrada | 224×224 RGB, escala `[-1, 1]` |
| Arquitetura | MobileNetV2 (ImageNet) + GAP + Dense 128 + Dropout 0.3 + softmax |
| Saída | softmax de 4 classes |
| Pesos | `models/corn_disease_model.keras` |
| Rótulos | `models/corn_disease_model.labels.json` |

### Classes

| Classe | Patologia | Sinal visual |
| --- | --- | --- |
| `Blight` | Helmintosporiose / *Exserohilum turcicum* | Lesões alongadas, cinza-esverdeadas, formato de charuto |
| `Common_Rust` | Ferrugem comum / *Puccinia sorghi* | Pústulas marrom-canela, salientes, nas duas faces |
| `Gray_Leaf_Spot` | Cercosporiose / *Cercospora zeae-maydis* | Lesões retangulares limitadas pelas nervuras |
| `Healthy` | — | Sem lesão |

### Datasets

Em ordem de prioridade — ver o [catálogo de datasets](#datasets) para detalhes.

1. `corn-maize-leaf-disease` (Kaggle) — **base principal**, já usa exatamente estes 4 nomes.
2. `plantvillage-corn` — engorda o volume, mas é laboratório (fundo uniforme).
3. `plantdoc-corn` — poucas imagens, porém **em campo**: é o que dá robustez real.
4. `cd-and-s` — campo, com severidade.
5. `plantseg-corn` — recorte de milho do PlantSeg (ver abaixo). Em campo, com máscara de lesão.

### Como treinar

```bash
# 1. baixe e extraia os datasets em qualquer pasta
# 2. consolide no layout canônico
#
#    Aponte para a SUBPASTA certa, nunca para a raiz: o PlantVillage publica
#    color/, grayscale/ e segmented/ — a mesma folha três vezes — e o
#    prepare_dataset varre a árvore inteira. Apontar para a raiz triplica cada
#    imagem e o split de validação passa a avaliar o que já está no treino.
python -m training.prepare_dataset --model leaf-foliar \
    --source "C:/datasets/corn-or-maize-leaf-disease" \
    --source "C:/datasets/PlantVillage/color" \
    --dry-run                      # confira o plano antes

python -m training.prepare_dataset --model leaf-foliar \
    --source "C:/datasets/corn-or-maize-leaf-disease" \
    --source "C:/datasets/PlantVillage/color"

# 3. remova as repetidas — sem isto a métrica final é ficção
python -m training.dedupe_dataset --model leaf-foliar --dry-run
python -m training.dedupe_dataset --model leaf-foliar

# 4. treine e meça
python -m training.train --model leaf-foliar --tta
python -m training.evaluate --model leaf-foliar --tta
```

### Como usar

```bash
curl -F "file=@folha.jpg" "http://localhost:8000/predict?model=leaf-foliar"
```

É o modelo padrão, então `?model=` pode ser omitido.

### A métrica que importa

**Não é a `val_accuracy`.** Sobre dataset público ela vai passar de 0,95 e isso
não diz nada sobre campo. O número que vale é a acurácia sobre um conjunto de
teste **capturado pela sua própria plataforma**, separado antes de qualquer
treino. Sem ele, você não tem número para prometer a cliente.

Medido, e não suposto: a primeira versão treinada só no
`corn-maize-leaf-disease` (4.566 imagens) marcou **0,9157** de `val_accuracy`.
O mesmo modelo, avaliado no conjunto de teste montado depois com dado de campo
(`maize-in-field` e `plantseg-corn`), tirou **0,5499** — e acertou apenas 27%
das folhas saudáveis, chamando 412 delas de ferrugem comum. A queda de 37
pontos não foi regressão: foi a diferença entre a régua de laboratório e a régua
de campo, aparecendo pela primeira vez.

Com o dataset consolidado e deduplicado (19.787 imagens), o mesmo treino sobe
para **0,9555** de acurácia e **0,9380** de macro-F1 no conjunto de teste
isolado — as duas medidas no mesmo conjunto, então os números se comparam. O
ganho real, portanto, não foi de 0,9157 para 0,9555: foi de **0,55 para 0,96 na
condição que importa**, a de campo.

**Prefira `macro-F1` a acurácia.** A acurácia deixa a classe abundante decidir o
número; o macro-F1 dá o mesmo peso à doença rara, que costuma ser a que importa
detectar cedo. Rode `python -m training.evaluate --model leaf-foliar --tta` para
o relatório por classe.

<a id="teto-do-dataset"></a>
### O teto: por que 100% não é a meta

Três medidas independentes, feitas neste dataset, apontam para o mesmo lugar:

1. **Rótulos que se contradizem.** A deduplicação por assinatura perceptual
   achou 88 grupos onde a *mesma imagem* aparece como `Blight` num dataset e
   `Gray_Leaf_Spot` noutro — 1.164 arquivos. Nenhum modelo pode acertar os dois
   rótulos. Isso é erro de Bayes, e é o teto real.
2. **A resolução acabou.** 93,9% das imagens têm o menor lado entre 224 e 319
   pixels. Treinar com entrada maior que 224 só reamostra o que já existe.
3. **A capacidade parou de pagar.** Trocar a MobileNetV2 (2,3M parâmetros) pela
   EfficientNetV2-S (21,6M) rendeu +0,65 ponto de acurácia e **piorou** o
   macro-F1. O TTA, que rende +1,2 ponto na MobileNetV2, rende +0,1 na
   EfficientNetV2-S — sinal de que o modelo já extraiu o que o dado tinha.

A conclusão prática: acurácia acima de ~0,95 sobre estes datasets públicos
indicaria vazamento entre treino e validação, não qualidade. O caminho para
subir de verdade é **dado de campo próprio e melhor rotulado**, não modelo maior.

### Resultado medido — 08/09/2026

Conjunto de teste isolado (15% da base, sorteado por permutação semeada sobre a
lista de arquivos, nunca usado para escolher época), sobre 19.787 imagens
consolidadas e deduplicadas:

| | Acurácia | macro-F1 | Pior classe |
| --- | ---: | ---: | --- |
| `leaf-foliar` | 0,9498 | 0,9313 | Gray_Leaf_Spot (recall 0,855) |
| `leaf-foliar` + TTA | **0,9555** | **0,9380** | Gray_Leaf_Spot (recall 0,851) |

O erro se concentra entre `Blight` e `Gray_Leaf_Spot` — o mesmo par dos 88
grupos de rótulo contraditório. O modelo erra onde os próprios datasets
discordam, o que é o comportamento esperado de um modelo que generaliza em vez
de decorar.

**Vazamento residual, medido:** nenhuma imagem do teste tem vizinho no treino a
distância de Hamming ≤ 2 (a deduplicação removeu até aí), mas 6,5% têm vizinho a
≤ 4. São folhas muito parecidas — provavelmente do mesmo pé. Isso torna o 0,9498
otimista em talvez 1 a 3 pontos. Para o número mais defensável possível,
rededuplique a `--max-distance 4` antes de retreinar.

### Limitações

- Exige a folha preenchendo o quadro. Em imagem nadir a média/grande altitude
  ele não funciona (ver [zoom e detecção por lesão](#zoom-e-deteccao)).
- Softmax de 4 classes: uma folha com duas doenças ao mesmo tempo é forçada a uma só.
- Sem classe "outro": foto de solo ou de céu sai como doença. Use `MIN_CONFIDENCE`
  (ver [abstenção](#abstenção)) e o portão de enquadramento
  ([README](README.md#enquadramento)).
- **388 máscaras de segmentação do PlantSeg estão no dataset como fotos**
  (`Common_Rust` 163, `Blight` 127, `Gray_Leaf_Spot` 98; 55 no teste de 15%).
  Medido em 26/09/2026: o modelo acerta 80% delas — aprendeu a diagnosticar
  imagem preta. Sem elas, a acurácia no mesmo teste sobe de 95,49% para 95,78%
  com TTA; o número publicado não estava inflado. Saem no próximo retreino
  (ver [README](README.md#imagens-quase-pretas-no-dataset)).

---

<a id="leaf-foliar-b"></a>
## 1b. `leaf-foliar-b` — a segunda opinião

### O que responde

O mesmo que o `leaf-foliar`: as mesmas quatro classes, sobre o mesmo dataset
consolidado. Muda só a base convolucional — EfficientNetV2B0 no lugar da
MobileNetV2 —, e é só isso que muda, de propósito.

Existe por causa da dupla validação da plataforma (`VALTEXA-SAAS`): um
diagnóstico só chega a `AI_VALIDATED` quando duas análises **independentes**
concordam. Com um modelo só, tudo vai para a fila do agrônomo.

### Ficha técnica

| | |
| --- | --- |
| Base | EfficientNetV2B0 (ImageNet), com `include_preprocessing=False` |
| Treino | cabeça com a base congelada; depois as 40 camadas finais liberadas |
| Entrada | 224 × 224, escala `[-1, 1]` — o mesmo pré-processamento do `leaf-foliar` |
| Pesos | `models/leaf_foliar_b.keras` (~40 MB) |
| Dataset | o do `leaf-foliar` (`dataset_key`), sem cópia |

### Como treinar

```bash
python -m training.train --model leaf-foliar-b --tta
python -m training.fit_novelty --model leaf-foliar-b
```

A arquitetura vem do catálogo; `--arch` só serve para experimentar.

### Resultado medido — 11/09/2026

Teste isolado de 10% (1.978 imagens). Pela semente e pelo corte, ele está
contido no teste de 15% do `leaf-foliar`: nenhum dos dois modelos viu estas
imagens.

| | Acurácia | macro-F1 | Pior classe |
| --- | ---: | ---: | --- |
| `leaf-foliar-b` | 0,9535 | 0,9319 | Gray_Leaf_Spot (recall 0,893) |
| `leaf-foliar-b` + TTA | **0,9601** | **0,9421** | Gray_Leaf_Spot (recall 0,893) |

### Independência em relação ao `leaf-foliar` — medida em 11/09/2026

Arquitetura diferente não prova independência: os dois aprendem dos mesmos
rótulos, inclusive dos contraditórios. A medição roda os dois modelos no mesmo
teste isolado e, à parte, um par de 3 classes treinado **sem** `Gray_Leaf_Spot`
diante de 400 imagens dela.

| Par de produção (1.978 imagens) | |
| --- | ---: |
| acurácia `leaf-foliar` / `leaf-foliar-b` | 95,4% / 95,3% |
| média das probabilidades dos dois | 96,1% |
| discordam | 5,2% |
| acurácia quando concordam | **98,1%** |
| erros do `leaf-foliar` que a discordância pega | **60%** |
| erros em que os dois erram igual | 36 de 90 |

| Doença retida (400 imagens de `Gray_Leaf_Spot`) | |
| --- | ---: |
| discordância entre os dois | 32,0% |
| confiança baixa (< 0,7) em algum dos dois | 26,5% |
| novidade do primário | 3,5% |
| **os três juntos** | **45,0%** |

Diante do que nunca viram, os dois erram diferente: o primário dividiu a
cercosporiose entre ferrugem (186) e helmintosporiose (174); o segundo puxou
para helmintosporiose (267). Por isso a discordância é o maior sinal isolado —
na medição de 08/09, o modelo sozinho interceptava 19%.

**Conclusão: parcialmente independente.** Vale como segunda opinião — quando
concordam, o erro cai de 4,6% para 1,9%, e a discordância é sinal real —, mas 40%
dos erros são compartilhados e 55% de uma doença nunca vista ainda passam
concordantes. A plataforma só conta o par como independente quando isso é
declarado (`SECONDARY_MODEL_INDEPENDENT`), e a recomendação é repetir esta
medição com imagens da própria frota antes.

A média das probabilidades (96,1%) é melhor que os dois sozinhos, mas não é o
uso aqui: um *ensemble* devolve uma resposta só; o par existe para mostrar quando
as duas discordam.

---

<a id="coffee-foliar"></a>
## 2. `coffee-foliar` — diagnóstico foliar de café

> **É o segundo modelo treinável hoje**, e o que carrega o nome da empresa. Sul
> de Minas é uma das principais regiões cafeeiras do país — é aqui que o
> diferencial real do produto nasce.

### O que responde

"Esta folha de café, ocupando o quadro, está sadia? Se não, é ferrugem,
cercosporiose, Phoma ou dano de bicho-mineiro?"

Mesma lógica de enquadramento do `leaf-foliar`: serve a qualquer plataforma
capaz de chegar perto da folha — drone em voo baixo, rover, ou até celular na
mão do agrônomo.

### Ficha técnica

| Item | Valor |
| --- | --- |
| Chave na API | `coffee-foliar` |
| Plataforma | `any` |
| Entrada | 224×224 RGB, escala `[-1, 1]` (mesmo pré-processamento do `leaf-foliar`) |
| Arquitetura | Idêntica ao `leaf-foliar`, cabeça de 5 saídas |
| Pesos | `models/coffee_foliar.keras` |
| Rótulos | `models/coffee_foliar.labels.json` |

### Classes

| Classe | Patologia/praga | Sinal visual |
| --- | --- | --- |
| `Healthy` | — | Sem lesão |
| `Rust` | Ferrugem / *Hemileia vastatrix* | Pústulas alaranjadas na face inferior da folha |
| `Cercospora_Leaf_Spot` | Cercosporiose / *Cercospora coffeicola* ("olho pardo") | Manchas circulares de centro claro e halo amarelado |
| `Phoma_Leaf_Spot` | Mancha de Phoma / *Phoma costarricensis* | Lesões escuras, geralmente na borda ou ápice da folha, associadas a frio/geada |
| `Leaf_Miner_Damage` | Bicho-mineiro / *Leucoptera coffeella* | Não é doença, é dano de inseto — galerias e manchas secas irregulares. Incluído porque o sintoma visual é do mesmo tipo (mancha foliar) e o dataset de origem já o trata como classe |

### Datasets

Em ordem de prioridade — ver o [catálogo de datasets](#datasets-cafe) para detalhes.

1. `jmuben-coffee` (JMuBEN/JMuBEN2) — **volume**: ~58.500 imagens já rotuladas
   nestas 5 classes exatas.
2. `rocole` — pequeno, mas em campo real, com variação de luz/fundo genuína.
3. `plantseg-coffee` — recorte de café do PlantSeg, in-the-wild, com máscara de lesão.

### Como treinar

```bash
python -m training.prepare_dataset --model coffee-foliar \
    --source "C:/datasets/JMuBEN" --source "C:/datasets/JMuBEN2" \
    --source "C:/datasets/RoCoLe" \
    --dry-run

python -m training.prepare_dataset --model coffee-foliar \
    --source "C:/datasets/JMuBEN" --source "C:/datasets/JMuBEN2" \
    --source "C:/datasets/RoCoLe"

# Obrigatório aqui: o JMuBEN do Kaggle é 96% duplicata (ver acima).
python -m training.dedupe_dataset --model coffee-foliar

python -m training.train --model coffee-foliar --tta
python -m training.evaluate --model coffee-foliar --tta
```

Os nomes de pasta do JMuBEN não foram inspecionados por nós — o mapeamento em
`SOURCE_CLASS_ALIASES` é *best-effort* nos nomes mais comuns (`Healthy`, `Rust`,
`Cercospora`, `Phoma`, `Miner`). Rode sempre com `--dry-run` primeiro: qualquer
pasta não reconhecida aparece na lista de "ignoradas" em vez de ser encaixada
errada.

### Como usar

```bash
curl -F "file=@folha_cafe.jpg" "http://localhost:8000/predict?model=coffee-foliar"
```

### A métrica que importa

Igual ao `leaf-foliar`: **acurácia sobre um conjunto de teste próprio**, não
`val_accuracy` do JMuBEN. Motivo específico deste modelo: o JMuBEN foi capturado
numa única fazenda em Kirinyaga, Quênia, com câmera digital comum, em imagens de
**128×128 pixels** — resolução baixa e origem geográfica única. Ele ensina o
modelo a reconhecer o *padrão* da lesão, não necessariamente a variação real do
Sul de Minas (outra cultivar, outra luz, outro solo, outra fase do ano).

<a id="jmuben-duplicado"></a>
**O JMuBEN distribuído no Kaggle é 96% duplicata.** Medido em 08/09/2026: das
59.942 imagens, apenas **2.570 são distintas**. Não são parecidas — a diferença
de pixel entre as cópias é **exatamente zero**, e as três imagens mais repetidas
aparecem cerca de 1.800 vezes cada. A varredura de limiar confirma que não é
artefato do método: com distância de Hamming 0, 2 e 4 a taxa fica em 4,3%, 4,0%
e 3,9% de sobreviventes.

A consequência é grave para quem usa esse dataset sem checar: com divisão
aleatória entre treino e validação, quase toda imagem de teste tem cópia
idêntica no treino. Um modelo treinado assim relata **0,9962** de `val_accuracy`
— e o número é ficção, porque o modelo está sendo consultado sobre o que
decorou. Rode `python -m training.dedupe_dataset --model coffee-foliar` antes de
qualquer treino.

### Resultado medido — 08/09/2026

Sobre as **3.462 imagens** que sobraram após a deduplicação, com conjunto de
teste isolado de 15%:

| | Acurácia | macro-F1 | Pior classe |
| --- | ---: | ---: | --- |
| `coffee-foliar` | 0,9441 | 0,9586 | Rust (recall 0,901) |
| `coffee-foliar` + TTA | **0,9499** | **0,9623** | Rust (recall 0,905) |

Número honesto, mas leia junto com a limitação de origem: a maior parte do que
sobrou ainda vem da fazenda única no Quênia, a 128×128. **Espere queda em lavoura
brasileira** — é exatamente o que a coleta no Sul de Minas existe para corrigir.

<a id="segunda-leitura-do-cafe"></a>
### Segunda leitura: a cabeça dNATY com o BRACOL — 01/10/2026

A queda em foto de outra fonte foi medida (6,5%, ver as limitações) e tinha causa
concreta: as **únicas** folhas inteiras sobre fundo branco do dataset eram 73 folhas de
phoma do BRACOL, e o modelo aprendeu "folha inteira em fundo branco = phoma" — o
bicho-mineiro externo virava phoma em 181 de 257 fotos.

A correção não retreina a CNN. Uma rede pequena lê o embedding SigLIP que o portão já
calcula (ver a [segunda leitura](README.md#segunda-leitura-cabeças-sobre-o-embedding-do-portão)),
com a arquitetura buscada pelo dNATY (768 → 1176 → 64 → 64 → 256 → 5) e treinada com o
`coffee-foliar` **mais o BRACOL** (CC BY 4.0, Espírito Santo: as cinco classes em folha
inteira sobre fundo branco), sem as 220 folhas do BRACOL parecidas demais com o dataset
do café. A resposta do `coffee-foliar` passa a ser **só a cabeça**: misturada com a CNN,
o acerto externo caía para 21%, porque a CNN continua achando que folha inteira é phoma.

| Medido pela API | Só a CNN | Com a cabeça |
| --- | ---: | ---: |
| teste isolado (494) | 0,939 (0,950 com TTA) | **0,964** |
| teste do BRACOL (163, nunca treinado) | 0,313 | **0,914** |
| **fotos externas** (542, Carneiro et al. 2021, só medida) | **6,5%** | **80,1%** |
| — bicho-mineiro / ferrugem | 0% / 12% | 77,4% / 82,5% |

As fotos externas são de outra fazenda e de outros autores (nenhum par com o BRACOL
acima de 0,98 de cosseno; a mais parecida difere 26 níveis de cinza em média) e não
escolheram nada: a cabeça saiu pela validação (café + BRACOL), com a regra fixada antes.
Na comparação, a sonda linear teria dado 90% no externo, mas perdia na validação e nos
dois testes isolados; trocar olhando o externo o transformaria em ajuste.

```bash
python -m training.import_bracol --zip ~/valtexa-raw/bracol/yy2k5y8mxg-v1.zip \
    --output data/coffee-bracol
python -m training.fit_heads --model coffee-foliar --extra data/coffee-bracol \
    --mix cnn=0,head=1 --holdout data/coffee-bracol-holdout
python -m training.fit_conformal --model coffee-foliar --field data/coffee-bracol-holdout
```

**O conjunto conformal precisou de campo.** Calibrado só na validação do café, ele
cobria 89% no teste isolado, mas só 72% no BRACOL e 48% nas fotos externas — a garantia
não valia no estilo "folha inteira em fundo branco". Alargado com as folhas do BRACOL que
a cabeça não treinou (metade calibra, metade mede, como no campo da banana e do citros),
passou a cobrir 94% no isolado, **92% no BRACOL e 71% nas fotos externas**. As externas
continuam abaixo dos 90% prometidos: outra fazenda, outro celular, e a garantia só vale
para o que se parece com a calibração.

O que a cabeça não muda: a novidade continua no embedding da CNN, o mapa de oclusão
explica a CNN, e o TTA deixa de mexer na resposta (a CNN entra com peso zero).

### Por que este é o modelo estratégico

Cruzando com a leitura de mercado: café puxa turismo/renda para boa parte do Sul
de Minas, a ferrugem é a doença mais economicamente relevante da cultura no
Brasil, e não existe concorrente óbvio já servindo diagnóstico por imagem
calibrado para lavoura brasileira. Um modelo público (JMuBEN + RoCoLe) tira o
produto do zero esta semana; **10-20 mil imagens próprias, capturadas em
lavouras do Sul de Minas com drone/rover**, é o que transforma isso de "um YOLO
treinado em dataset público" em vantagem real e defensável.

### Limitações

- **Foto de outra fonte: de 6,5% para 80% com a [segunda leitura](#segunda-leitura-do-cafe)
  (01/10/2026).** O que segue abaixo é a medição da CNN sozinha, que motivou a correção.
  Continua sem medida a folha **na planta** (o rover): as fotos externas e o BRACOL são
  folha destacada sobre fundo branco, e o único campo do teste isolado é o RoCoLe.
  Medido em 30/09/2026 em 542 fotos de
  folha inteira, nítida, sobre fundo claro (`alvarole/coffee-leaves-disease`,
  bicho-mineiro e ferrugem; nenhuma parecida com o treino pela mesma checagem do
  `dedupe_dataset`): **6,5% de acerto**, contra 95% no teste isolado. O bicho-mineiro
  vira phoma (181 de 257 fotos) e a ferrugem, cercospora ou phoma. O teste isolado
  mistura duas fontes (as 754 folhas sadias são fotos de campo de 1920×1080; as
  outras três doenças são lesões em 128×128) e o modelo aprendeu o jeito de
  fotografar de cada uma, não a doença: em campo, a ferrugem acerta 76% (65 de 86) e
  em laboratório 99% (155 de 157). Dos 507 erros, 85% disparam alguma salvaguarda;
  os outros 74 saem sem nenhum alarme. Esse conjunto externo é "copyright-authors"
  (serve para medir, não para treinar), e o "Ethiopian Coffee Leaf Disease" é 96% cópia
  do JMuBEN (1.153 de 1.200 fotos), então não vale nem como teste. O dado licenciado
  que faltava era o BRACOL (CC BY 4.0), já baixado e mal catalogado como "só detecção":
  a pasta `leaf` dele tem rótulo por folha, e é com ele que a segunda leitura corrige.
  Um experimento sem retreino ajuda pouco: recortar a caixa da folha numa grade 4×4 e
  agregar os recortes (média dos logs) sobe o acerto externo de 6,7% para 37,4% (a
  grade foi escolhida numa metade das fotos e medida na outra), mas só a ferrugem
  melhora (de ~12% para 71%); o bicho-mineiro segue em **0%**. Não virou recurso.
- As mesmas do `leaf-foliar`: exige folha preenchendo o quadro, softmax de
  classe única, sem categoria "outro".
- `Leaf_Miner_Damage` mistura semanticamente doença e dano de inseto na mesma
  cabeça de classificação — funciona porque o dataset de origem já assim o
  trata, mas vale revisar se um modelo de pragas separado (ver `IP102` no
  catálogo) compensar mais à frente.
- Nenhum dos três datasets públicos foi capturado no Brasil. Trate a
  `val_accuracy` deles como piso, não como meta.
- **O modelo aprendeu com fotos ruins.** Medido em 26/09/2026 pelo
  `fit_quality`, sobre as 3.462 imagens deduplicadas: na entrada do modelo, o
  dataset de treino já é tão pouco nítido e tão mal exposto (o JMuBEN é 128×128,
  ampliado para 224) que um desfoque de 2 px aplicado de propósito só sai do que
  ele conhece em 18% das fotos — no milho, em 98% —, e um escurecimento a 30% da
  luz, em 19%. A verificação de qualidade não protege este modelo contra foto
  tremida ou escura, porque para ele isso é normal. Só a coleta própria, em
  resolução real, muda isso. Ver [qualidade da captura](README.md#qualidade-da-captura).
- **Sem textura, ele responde `Rust`.** Medido em 26/09/2026 pelo
  `audit_attention`: trocada a foto pela cor média dela, o modelo mantém `Rust`
  com confiança ≥ 0,9 em 92% das folhas com ferrugem — e uma varredura de 936
  cores chapadas deu `Rust` confiante nas 936. É o mesmo tipo de atalho que pôs o
  `rover-understory` em quarentena, e explica o caso do drone (cafezal visto de
  cima vira `Rust` em 94% dos quadros). A causa está nos dados: **150 das 1.540
  imagens de `Rust` são máscaras de segmentação do PlantSeg**, fundo preto com
  polígonos cinza. A correção é reimportar sem as máscaras e retreinar; o
  candidato sem máscaras não chegou a ser treinado (prioridade mudou para citros e
  banana em 27/09/2026).

---

<a id="rover-understory"></a>
## 3. `rover-understory` — diagnóstico sob o dossel

> **É o produto.** O drone vê tarde; o rover vê cedo e vê o que o drone nunca vê.

### O que responde

"Esta planta, vista de lado no baixo dossel, está sadia? Se não, o que tem —
inclusive no colmo e na base?"

### As duas razões de existir

**Detecção precoce das mesmas doenças foliares.** Cercosporiose e ferrugem
polissora **começam nas folhas baixas e sobem**. Quando o drone as enxerga no
topo do dossel, a infecção já avançou. O rover pega as mesmas classes semanas
antes — essa antecipação é o valor comercial mais defensável do produto.

**Patologias exclusivas do baixo.** Podridão de colmo, dano de broca e sintomas
de base não existem na imagem aérea, em nenhuma altitude.

### Ficha técnica

| Item | Valor |
| --- | --- |
| Chave na API | `rover-understory` |
| Plataforma | `ground` |
| Entrada | 224×224 RGB, escala `[-1, 1]` |
| Arquitetura | Igual ao `leaf-foliar` (mesmo backbone, cabeça de 8 saídas) |
| Pesos | `models/rover_understory.keras` |
| Rótulos | `models/rover_understory.labels.json` |

### Classes

As quatro primeiras têm dataset público. As quatro últimas **exigem coleta própria**.

| Classe | O que é | Dataset público? |
| --- | --- | --- |
| `Healthy` | Planta sem sintoma | ✅ |
| `Blight` | Helmintosporiose | ✅ |
| `Common_Rust` | Ferrugem comum | ✅ |
| `Gray_Leaf_Spot` | Cercosporiose — **começa embaixo** | ✅ |
| `Stalk_Rot` | Podridão de colmo (Fusarium, antracnose, Diplodia) | ❌ |
| `Stalk_Borer_Damage` | Broca / lagarta no colmo | ❌ |
| `Nutrient_Deficiency` | Deficiência de N (o "V" amarelo clássico) | ❌ |
| `Corn_Stunt` | Enfezamento — cigarrinha (*Dalbulus maidis*) | ❌ |

Sobre agrupar as podridões de colmo numa classe só: separar Fusarium de
antracnose de Diplodia por foto é pouco confiável mesmo para agrônomo — a
confirmação é laboratorial. Uma classe única é honesta e acionável (o manejo é
parecido). Divida depois, se os dados justificarem.

`Corn_Stunt` está na lista por relevância de mercado: no Brasil, o complexo de
enfezamentos transmitido pela cigarrinha virou um problema maior que várias das
doenças foliares clássicas.

### Aviso — os pesos de 08/09/2026 estão em quarentena

Uma primeira tentativa de treinar este modelo com o `corn-stalks`
(kaggle:ndisan/corn-stalks) marcou **1,0000 de acurácia** no conjunto de teste,
e já na primeira época com a base congelada. Um modelo não aprende a
diagnosticar podridão de colmo numa época: aquilo era sinal de que as classes
eram separáveis por algo que não é a doença.

A verificação foi direta. Uma regressão logística treinada com **seis números
por imagem** — média e desvio de R, G e B, sem nenhuma textura, forma ou lesão —
também acerta **100%**. As imagens de colmo saudável e de colmo podre foram
capturadas em condições diferentes, e o modelo aprendeu a reconhecer a *sessão
de fotografia*.

O mesmo dataset já havia mostrado 457 imagens (15%) presentes simultaneamente em
`SEHAT` e nas pastas de podridão, removidas pela deduplicação.

Os pesos estão em `models/_quarentena/` e **não devem ser servidos** — em campo
o modelo erraria na primeira imagem, porque a pista que ele usa não existe fora
daquele dataset. A lição para a coleta própria: **fotografe saudável e doente na
mesma sessão, mesma luz, mesmo fundo**, intercalando as classes. Se um lote
inteiro de uma classe sair de uma condição e o outro lote de outra, o dataset
nasce inutilizável — e o número bonito esconde isso.

### Estratégia de treino em duas etapas

Você **não precisa** ter as 8 classes para começar.

**Etapa A — foliares, agora.** Treine com as 4 classes públicas e valide se o
modelo funciona em ângulo lateral, sob sombra de dossel. Já é um produto.

```bash
python -m training.prepare_dataset --model rover-understory \
    --source "C:/datasets/PlantDoc" --source "C:/datasets/CD&S"
python -m training.train --model rover-understory
```

O `prepare_dataset` avisa quais classes ficaram vazias e o treino segue com as
que existem — o `.labels.json` registra a ordem real, e a API se adapta.

**Etapa B — patologias baixas, após a coleta.** Ver o
[protocolo de coleta](#coleta). Quando cada classe nova tiver volume, retreine
com as 8.

### Como usar

```bash
curl -F "file=@colmo.jpg" "http://localhost:8000/predict?model=rover-understory"
curl -F "file=@colmo.jpg" "http://localhost:8000/predict?platform=ground"
```

### A métrica que importa

**Precisão por classe, não acurácia global.** Este modelo gera recomendação de
manejo: um falso "podridão de colmo" pode motivar dessecação antecipada de um
talhão sadio. Acompanhe a matriz de confusão classe a classe e trate
`Healthy → doença` como o erro caro.

### Desafios específicos da captura terrestre

Antecipe estes ao montar o dataset — todos degradam o modelo se só aparecerem em produção:

- **Luz baixa e irregular** sob o dossel, com manchas de sol.
- **Oclusão** — folhas na frente do colmo alvo.
- **Motion blur** do terreno irregular.
- **Poeira** na lente.
- **Fundo poluído** — solo, palhada e outras plantas no quadro.

### Limitações

- Sintoma radicular só aparece se a câmera enxergar a base. Defina a altura da
  câmera com isso em mente.
- Classificação de imagem inteira: se o quadro pega três plantas, o rótulo é
  ambíguo. Detecção (YOLO) é o passo natural depois.
- 224 px pode ser pouco para textura fina de colmo. Subir para 320 é um teste
  válido — a API lê o tamanho de entrada direto do modelo e se adapta sozinha.

---

<a id="aerial-canopy"></a>
## 4. `aerial-canopy` — triagem de dossel

> Modelo de **triagem**, não de diagnóstico. Existe para direcionar o rover.

### O que responde

"Este pedaço de dossel merece uma visita?"

### Por que é binário

Em altitude de varredura, uma lesão de cercosporiose ocupa poucos pixels e o
quadro cobre dezenas de plantas. Pedir "qual das 4 doenças" nessa resolução é
pedir algo que a informação não sustenta. Binário com **recall alto** é honesto
e suficiente para o papel dele no pipeline.

### Ficha técnica

| Item | Valor |
| --- | --- |
| Chave na API | `aerial-canopy` |
| Plataforma | `aerial` |
| Entrada | 224×224 RGB — **tiles**, não a foto inteira |
| Saída | `Healthy_Canopy` / `Suspect` |
| Pesos | `models/aerial_canopy.keras` |

### O passo de tiling

Uma foto de drone tem milhares de pixels de lado. Enviá-la inteira ao modelo
dilui qualquer lesão. O fluxo correto:

1. recortar a imagem em tiles (ex.: 512×512 com 10-20% de sobreposição);
2. classificar cada tile — cabem no `/predict/batch`;
3. reprojetar o resultado no GPS para virar mapa de risco.

O recorte fica no cliente ou num serviço de pré-processamento; a API classifica
os tiles como imagens comuns.

### Datasets

`disease-nlb-drone` é a peça-chave: é o subconjunto **só-UAV** do dataset de
NLB em milho da Cornell (Wiesner-Hanks et al.), hospedado separadamente na OPIA
— 7.669 imagens com 42.117 lesões anotadas, capturadas com DJI Matrice 600.
Ver [zoom e detecção por lesão](#zoom-e-deteccao) para o dado real de altitude e
velocidade de voo usado nessa coleta — é a melhor referência publicada que
existe para calibrar o próprio voo.

### A métrica que importa

**Recall**, com folga. Um falso positivo custa uma passagem do rover. Um falso
negativo é um foco que não foi visto. Escolha o limiar pela curva
precisão/recall, não pelo padrão de 0,5.

---

<a id="teste-de-campo-rover"></a>
## O teste de campo na escala do rover

Vale para os dois modelos abaixo, e é o número a citar deles.

O teste isolado sai **das mesmas fontes do treino** — mesma câmera, mesmo fundo,
mesma luz — e por isso mede pouco. O primeiro `banana-foliar` marcou **0,9846**
nele; nas fotos de campo do PlantSeg, que nenhum treino viu, **0,4281**. Metade
das folhas com Sigatoka saiu como Fusariose, e a Cordana quase nunca foi
reconhecida (2 de 55).

Olhando os erros, o problema não era só o modelo: as fotos do PlantSeg são
bananal inteiro, planta murcha de corpo inteiro, uma mão segurando a folha — e o
rover **nunca classifica a planta inteira**. Ele fatia o quadro em recortes de
150 mm (ver [README](README.md#rover-do-quadro-à-zona-da-planta)), cada um um
pedaço de limbo. Medir o modelo em foto de bananal é medir outra tarefa.

O [`lesion_tiles`](training/lesion_tiles.py) resolve isso com as **máscaras de
lesão** que o PlantSeg traz: cada foto vira uma grade de recortes quadrados, em
duas escalas, e fica só o recorte com lesão de verdade dentro (≥ 5% da área) —
o que o rover veria se passasse por aquela planta. Os recortes respeitam o split
oficial do PlantSeg, então nenhuma lesão aparece no treino e no teste:

```bash
python -m training.lesion_tiles --source ~/valtexa-raw/plantseg/plantseg \
    --output ~/valtexa-raw/_plantseg-recortes
python -m training.evaluate --model banana-foliar --tta \
    --external ~/valtexa-raw/_plantseg-recortes/test/banana
```

| Cultura | Classe | Treino (fotos → recortes) | Teste (fotos → recortes) |
| --- | --- | ---: | ---: |
| banana | Sigatoka | 131 → 777 | 32 → 191 |
| banana | Fusarium_Wilt | 42 → 233 | 12 → 70 |
| banana | Cordana | 41 → 238 | 8 → 48 |
| citros | Canker | 284 → 1.489 | 72 → 390 |
| citros | Greening | 97 → 537 | 23 → 121 |

**O tamanho real da amostra é o número de fotos**, não o de recortes: recortes
da mesma foto são a mesma lesão vista de novo. Oito fotos de Cordana no teste
não sustentam casa decimal — os números abaixo trazem as duas contagens.

Conferido antes de medir: nenhuma foto do PlantSeg de banana ou citros está nos
dados de treino (dHash + miniatura, mesma regra do `dedupe_dataset`).

Nem todo recorte com lesão é folha. Na Fusariose, parte é o **corte
transversal** do pseudocaule (o sintoma interno, que o rover não vê); no cancro,
metade é **fruto**. Em produção quem separa isso é o [portão de
enquadramento](README.md#enquadramento) — por isso o teste é relatado duas
vezes: com todos os recortes, e só com os que passam o portão, que é o caminho
real de um recorte no rover.

---

<a id="banana-foliar"></a>
## 5. `banana-foliar` — folhas baixas da bananeira

> **A zona de folhas baixas do rover na bananeira.** As folhas velhas, de baixo,
> são onde a Sigatoka é mais severa e onde o mal-do-Panamá amarela primeiro.

### O que responde

"Este pedaço de folha de bananeira, ocupando o recorte, está sadio? Se não, é
Sigatoka, Fusariose (mal-do-Panamá), Cordana ou Pestalotiopsis?"

### Ficha técnica

| Item | Valor |
| --- | --- |
| Chave na API | `banana-foliar` |
| Plataforma | `any` — no rover, pela rota `/predict/rover-frame` |
| Entrada | 224×224 RGB, `[-1, 1]` |
| Arquitetura | MobileNetV2 + cabeça de 5 saídas (a do catálogo) |
| Pesos | `models/banana_foliar.keras` |

### Classes

| Classe | Patologia | Sinal visual |
| --- | --- | --- |
| `Healthy` | — | Limbo verde, sem lesão |
| `Sigatoka` | Sigatoka-negra e amarela / *Pseudocercospora fijiensis*, *P. musae* | Estrias que escurecem e coalescem em áreas necróticas |
| `Fusarium_Wilt` | Mal-do-Panamá / *Fusarium oxysporum* f. sp. *cubense* | Amarelecimento a partir da margem das folhas velhas |
| `Cordana` | Mancha de Cordana / *Neocordana musae* | Lesões ovais grandes, centro pardo e halo amarelo |
| `Pestalotiopsis` | *Pestalotiopsis* spp. | Manchas irregulares pardas na margem |

Sigatoka-negra e amarela ficam juntas: nos datasets públicos os rótulos não as
separam com segurança.

### Datasets

| Dataset | Origem | Licença | Uso |
| --- | --- | --- | --- |
| Banana leaf and stem (Tanzânia) | Harvard Dataverse `doi:10.7910/DVN/LQUWXW` | CC BY 4.0 | Sigatoka, Fusariose, sadia — em campo |
| BananaLSD | Kaggle `shifatearman/bananalsd` (Bangladesh) | CC BY 4.0 | Cordana, Pestalotiopsis, Sigatoka, sadia |
| Banana Disease Recognition | Kaggle `sujaykapadnis/banana-disease-recognition-dataset` | CC BY 4.0 | Sigatoka, Fusariose, sadia |
| PlantSeg (banana) | Zenodo, split oficial | CC BY-NC 4.0 | recortes do treino **entram no treino**; recortes do teste = teste de campo |

As fotos de "caule" do dataset da Tanzânia são o corte transversal interno e não
entram. Deduplicado com o `dedupe_dataset` (versão com confirmação por miniatura,
ver [citros](#citrus-foliar)): **17.191 imagens**. A versão anterior, só com
dHash, descartava 1.508 fotos como "a mesma imagem em classes diferentes"; com a
confirmação sobra um único conflito real (2 arquivos).

### Como treinar

```bash
python -m training.prepare_dataset --model banana-foliar \
    --source ~/valtexa-raw/banana-tz \
    --source ~/valtexa-raw/bananalsd/BananaLSD/OriginalSet \
    --source "~/valtexa-raw/banana-recognition/Banana Disease Recognition Dataset/Original Images/Original Images"
python -m training.dedupe_dataset --model banana-foliar
# recortes de campo na escala do rover (split oficial do PlantSeg)
python -m training.lesion_tiles --source ~/valtexa-raw/plantseg/plantseg \
    --output ~/valtexa-raw/_plantseg-recortes
# data/banana-foliar-campo = data/banana-foliar + _plantseg-recortes/train/banana
python -m training.train --model banana-foliar --tta --test-split 0.15 \
    --data-dir data/banana-foliar-campo
python -m training.evaluate --model banana-foliar --tta \
    --external ~/valtexa-raw/_plantseg-recortes/test/banana
python -m training.fit_prior --model banana-foliar --data-dir data/banana-foliar-campo
```

### Resultado medido — 27/09/2026

| | Teste isolado | Campo — recortes | Campo — fotos (voto) |
| --- | ---: | ---: | ---: |
| 1º treino, sem recortes de campo (hoje `banana-foliar-cc`) | 0,985 | 0,592 | 0,692 (36/52) |
| **`banana-foliar`**, com os recortes do PlantSeg-treino | 0,977 | **0,829** | **0,846 (44/52)** |

Por classe, no campo (recortes): Fusariose 0,929, Sigatoka 0,848, Cordana
0,625 (era 0,083). O erro que sobra é Cordana ↔ Sigatoka nos dois sentidos (16 e
19 recortes). No teste isolado — que agora inclui recortes do PlantSeg — a pior
classe é Cordana (recall 0,922, n = 51).

Salvaguardas calibradas: novidade, conformal, qualidade e prevalência de treino.
Auditoria de atenção: **nenhuma classe responde pela cor** (0% com o quadro
chapado); Pestalotiopsis, porém, tem 80% das imagens "sem região" — cobrir o
trecho mais decisivo quase não muda a confiança, sinal de que o modelo lê a foto
inteira nessa classe. ONNX: 9,7 MB, mesma classe que o Keras em 100% de 256
imagens reais (diferença máxima 2,9e-6), 8,9 ms por imagem em CPU.

### Limitações

- **Pseudocaule não é lido.** Não há dataset público com o sintoma externo do
  pseudocaule na escala do rover. O portão manda esses recortes para
  `out_of_scope`, e a fila de revisão do rover é o que vai formar esse dataset.
- **Cordana e Pestalotiopsis vêm de uma fonte só** (BananaLSD, Bangladesh), com
  ~160 fotos cada. Os recortes do PlantSeg somaram 238 de Cordana (41 fotos) e levaram o recall de campo de 8% a 62,5% — mas são 8 fotos de teste, e Cordana e Sigatoka continuam se confundindo. Pestalotiopsis não tem teste de campo.
- **A versão comercial (`banana-foliar-cc`) tem um atalho de cor.** Na auditoria
  de atenção, um quadro chapado da cor média da foto sai `Healthy` com confiança
  ≥ 0,9 em 100% das folhas sadias, e Pestalotiopsis não depende de região
  nenhuma (cobrir o trecho decisivo não muda nada). Qualquer superfície verde
  uniforme vira "sadia" com certeza. A versão de campo não tem o defeito (0%) —
  os recortes do PlantSeg também corrigiram isso.
- Moko (*Ralstonia*), presente no Brasil, e *bunchy top* não estão no catálogo:
  caem numa das cinco classes, e só a novidade e o conjunto conformal avisam.
- Nenhum dos datasets é brasileiro.

---

<a id="citrus-foliar"></a>
## 6. `citrus-foliar` — folhas de citros (laranja e limão)

> **A copa baixa do citros no rover.** Laranja e limão juntos: as doenças
> foliares que os datasets públicos cobrem são as mesmas nos dois.

### O que responde

"Esta folha de citros, ocupando o recorte, está sadia? Se não, qual das onze
doenças e pragas?"

### Ficha técnica

| Item | Valor |
| --- | --- |
| Chave na API | `citrus-foliar` |
| Plataforma | `any` — no rover, pela rota `/predict/rover-frame` |
| Entrada | 224×224 RGB, `[-1, 1]` |
| Arquitetura | MobileNetV2 + cabeça de 12 saídas |
| Pesos | `models/citrus_foliar.keras` |

### Classes

`Algal_Leaf_Spot` (mancha de alga, *Cephaleuros*), `Anthracnose`
(antracnose), `Black_Spot` (pinta-preta, *Phyllosticta citricarpa*), `Canker`
(cancro cítrico, *Xanthomonas citri*), `Curl_Leaf` (enrolamento), `Greening`
(HLB, *Candidatus* Liberibacter), `Healthy`, `Leafminer` (larva-minadora,
*Phyllocnistis citrella*), `Melanose` (melanose, *Diaporthe citri*), `Mites`
(ácaros), `Scab` (verrugose, *Elsinoë fawcettii*), `Sooty_Mold` (fumagina).

### Datasets

| Dataset | Origem | Licença |
| --- | --- | --- |
| Large-Scale Lemon Leaf Disease and Pest | Mendeley `8d9fv6kpt3` v1 (Bangladesh) | CC BY 4.0 |
| BDLemonLeaf — só `Raw Image` | Mendeley `643f5bbc2t` v6 | CC BY 4.0 |
| Sweet orange leaf — só `Original Image` | Mendeley `f7cr74mwpj` v1 | CC BY 4.0 |
| PlantSeg (citros) | Zenodo, split oficial | CC BY-NC 4.0 |

Do BDLemonLeaf fica de fora a pasta `Annotated Image` (tem as caixas
desenhadas na foto); do de laranja, a `Converted Image` (a mesma foto de novo).
O `citrus-jsilva` do Kaggle é só fruto e não entra.

### Dois problemas dos dados, e o que foi feito

**A folha está sobre papel.** Os três datasets fotografam a folha destacada,
sobre papel liso; o rover a vê na árvore. A primeira tentativa foi a **troca de
fundo** ([`background`](training/background.py)): separar a folha do papel (cor
da borda + Otsu; a folha foi separada em 92% das fotos reais) e colá-la, a cada
época, sobre recortes de pomar filmado do chão (184 quadros de vídeos públicos
do YouTube). **Medida, ela piorou o modelo** — ver a tabela abaixo — e ficou de
fora. O que resolveu foram os recortes de campo do PlantSeg. O código continua
disponível (`--backgrounds`), com a máscara da folha calculada uma vez e guardada
no cache do `tf.data` (o passo de treino caiu de 394 para 77 ms).

**O dHash sozinho apagava folhas diferentes.** Toda folha destacada sobre papel
tem o mesmo gradiente — claro, escuro, claro. Com a distância 4, a união
transitiva juntou as 12 classes num grupo só e ia apagar 68% do dataset. O
`dedupe_dataset` agora só une um par que o dHash indicou se a **miniatura
colorida da folha** (a caixa do que não é papel, sem o papel) também bater.
Medido em 6.000 pares indicados: entre classes, **nenhum** fica até 10 (mediana
39); dentro da classe há um vão limpo entre a mesma folha fotografada em
sequência (9%) e folhas diferentes. Resultado: de 18.839 imagens, **11.764**
distintas — os datasets fotografam a mesma folha até 14 vezes seguidas — e 30
conflitos reais de rótulo (65 arquivos, a maioria `Algal_Leaf_Spot` e
`Black_Spot` com a mesma foto).

| Classe | Imagens | Classe | Imagens |
| --- | ---: | --- | ---: |
| Greening | 2.170 | Leafminer | 775 |
| Healthy | 1.724 | Anthracnose | 765 |
| Canker | 1.684 | Mites | 669 |
| Sooty_Mold | 1.589 | Black_Spot | 380 |
| Curl_Leaf | 1.356 | Algal_Leaf_Spot | 322 |
| | | Melanose | 188 |
| | | Scab | 142 |

### Como treinar

```bash
python -m training.prepare_dataset --model citrus-foliar \
    --source ~/valtexa-raw/lemon-large/8d9fv6kpt3-v1 \
    --source ".../BDLemonLeaf.../Root/Raw Image" \
    --source ".../Sweetorange/Original Image"
python -m training.dedupe_dataset --model citrus-foliar
python -m training.lesion_tiles --source ~/valtexa-raw/plantseg/plantseg \
    --output ~/valtexa-raw/_plantseg-recortes
# data/citrus-foliar-campo = data/citrus-foliar + _plantseg-recortes/train/citrus
python -m training.train --model citrus-foliar --tta --test-split 0.15 \
    --data-dir data/citrus-foliar-campo
python -m training.evaluate --model citrus-foliar --tta \
    --external ~/valtexa-raw/_plantseg-recortes/test/citrus
python -m training.fit_prior --model citrus-foliar --data-dir data/citrus-foliar-campo
```

### Resultado medido — 27/09/2026

Quatro treinos, para separar o efeito de cada decisão — o teste de campo só
tem cancro e greening (95 fotos):

| Treino | Teste isolado | Campo — recortes | Campo — fotos (voto) |
| --- | ---: | ---: | ---: |
| só papel (hoje `citrus-foliar-cc`) | 0,874 | 0,695 | 0,726 (69/95) |
| papel + troca de fundo | 0,847 | 0,382 | 0,421 (40/95) |
| papel + troca de fundo + recortes de campo | 0,848 | 0,959 | 0,979 (93/95) |
| **`citrus-foliar`**: papel + recortes de campo | **0,884** | **0,969** | — |

No campo, o instalado acerta 96,4% do cancro e 98,3% do greening. No teste
isolado, as piores classes são Leafminer e Melanose.

Salvaguardas calibradas: novidade, conformal, qualidade e prevalência de treino.
Auditoria de atenção: nenhuma das 12 classes responde pela cor (0%). Cancro (16%
"sem região") e verrugose (0%) são lidos pela lesão; fumagina (68%) e ácaros
(84%) pela folha inteira, como o sintoma; **Melanose (92%) é suspeita** — classe
pequena, pode estar reconhecendo a fonte. ONNX: mesma classe que o Keras em 100%
de 256 imagens (diferença máxima 4,6e-6), 10,7 ms por imagem em CPU.

### Orientação da folha — medido em 01/10/2026

O citros acerta 88,8% com a folha em pé e **82,3% de cabeça para baixo** (espelho
vertical ou giro de 180°, 1.200 imagens do teste isolado); a banana, o milho e o café
variam menos de 1,2 ponto. A causa estava certa: o treino só espelha na horizontal.
Retreinado com o espelho nos dois eixos, a diferença some (87,3 a 87,7%), **mas sem
ganho líquido**: com TTA o acerto é o mesmo (88,67%), o isolado sem TTA perde 1,8 ponto,
o campo com TTA fica em 0,967 contra 0,969 (93 contra 95 das 95 fotos), e o comercial
ficou **pior** no campo (0,634 contra 0,693; 66 contra 70 das 95 fotos). Como original,
espelho horizontal, espelho vertical e giro de 180° formam um grupo fechado, **com TTA a
resposta já é idêntica para a foto em pé e de cabeça para baixo**: com `?tta=true` o
problema some por construção para flips e giro de 180°, sem retreinar. No rover, porém,
o TTA vem **desligado** (`ROVER_TTA=false`), porque melhora o recorte mas piora a
detecção por planta (29 para 24 de 33 plantas de citros no talhão de ponta a ponta;
ver o README), então a sensibilidade à orientação continua valendo ali até a fusão ser
recalibrada para saídas com TTA. Os pesos instalados ficaram os de 27/09/2026. Ficam de
fora os giros de 90° (foto de celular em retrato, que a API não corrige pelo EXIF) e os
arbitrários, que o TTA não cobre.

**Resolvido sem TTA, pela segunda leitura (01/10/2026).** Uma rede pequena sobre o
embedding SigLIP que o portão já calcula para toda foto, com a arquitetura buscada pelo
dNATY e treinada com as quatro orientações (ver [segunda leitura](README.md#segunda-leitura-cabeças-sobre-o-embedding-do-portão)),
lê a folha do mesmo jeito nas quatro. Misturada à CNN (média geométrica, 0,25 CNN + 0,75
cabeça, peso escolhido na validação com as quatro orientações), **sem TTA** — que é como o
rover roda —, no teste isolado inteiro:

| | Em pé | Espelho H | Espelho V | Giro 180° | Campo, 511 recortes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `citrus-foliar`, só a CNN | 0,884 | 0,880 | 0,819 | 0,821 | 0,951 |
| `citrus-foliar` + cabeça | **0,931** | **0,936** | **0,937** | **0,933** | **1,000** |
| `citrus-foliar-cc`, só a CNN | 0,870 | 0,867 | 0,806 | 0,819 | 0,665 |
| `citrus-foliar-cc` + cabeça | **0,918** | **0,922** | **0,926** | **0,921** | **0,795** |

Com TTA, a API dá 0,931 no isolado e 0,996 no campo (`citrus-foliar`; antes 0,879 e
0,969) e 0,919 / 0,798 no comercial (antes 0,871 / 0,693). A cabeça do comercial só viu
dado CC BY (folhas sobre papel); a do `citrus-foliar` viu também os recortes de treino do
PlantSeg, como a CNN dele. O teste de campo continua com só duas classes (cancro e
greening). A detecção por planta no talhão de ponta a ponta está na
[segunda leitura](README.md#segunda-leitura-cabeças-sobre-o-embedding-do-portão).

### Limitações

- **O teste de campo só tem duas das doze classes** (cancro e greening, as que o
  PlantSeg traz). As outras dez só foram medidas sobre papel.
- **Nenhum dataset é brasileiro**, e as doenças que mais importam no citros
  paulista — CVC, leprose, gomose, pinta-preta em fruto — ou não estão no
  catálogo ou só aparecem em folha. Caem na fila de revisão, se a novidade ou o
  conjunto conformal perceberem.
- Scab (142) e Melanose (188) são classes finas.
- **Melanose e Leafminer** são as classes mais fracas, e a auditoria sugere que a
  Melanose é lida pela foto, não pela lesão.
- Os recortes de campo são do PlantSeg (CC BY-NC 4.0). A versão para uso
  comercial é o `citrus-foliar-cc`, que erra muito mais no campo — a saída é a
  coleta própria do rover. Ela não tem atalho de cor (0% com o quadro chapado,
  apesar do papel), mas lê Mancha de Alga (84% "sem região") e Pinta-Preta (88%)
  pela folha inteira.

---

<a id="fusao-por-planta"></a>
## Fusão por planta e incidência do talhão — calibradas em 02/10/2026

O rover não diagnostica foto: soma a evidência dos recortes de uma planta, de várias
câmeras, e diz `sadia`, `suspeita` ou `doente` ([`plant_fusion`](app/plant_fusion.py)).
O artefato é `models/<pesos>.fusion.json`, gravado pelo
[`fit_fusion`](training/fit_fusion.py): a razão de verossimilhança de um recorte por
faixa de probabilidade de cada doença, os limiares dos status pelo alarme falso em
plantas sadias (5% para `suspeita` ou `doente`, 2% para `doente`) com piso no que o
status quer dizer (`suspeita` só com razão de verossimilhança calibrada ≥ 1, `doente`
só com posterior ≥ 50% no prior de 5%), a escala de Platt da posterior e a
sensibilidade e o alarme falso por planta. Vale para os pesos com que
foi medido e para recortes sem TTA: pesos novos pedem `fit_fusion` de novo, como o
`fit_conformal`.

### Resultado medido — validação cruzada aninhada em 10 partes

Planta = três fotos, uma por altura de câmera do mastro, pelo caminho do
`/predict/rover-frame`. Tudo fora da amostra.

| | Doentes detectadas | Doença certa | Sadias com alarme | Sadias como `doente` |
| --- | ---: | ---: | ---: | ---: |
| `banana-foliar`, sem calibração | 66/125 (53%) | 20 | 13/283 (4,6%) | 0 |
| **`banana-foliar`, calibrada** | **115/125 (92%)** | 61 | 15/283 (5,3%) | 0 |
| `citrus-foliar`, sem calibração | 27/33 | 27 | 3/83 (3,6%) | 0 |
| **`citrus-foliar`, calibrada** | **33/33** | 32 | 2/83 (2,4%) | 0 |

**O piso dos limiares.** A primeira versão instalada (02/10/2026, de manhã) só tinha a
meta de alarme falso. Nos citros, as sadias de papel têm evidência muito negativa, e a
meta de 5% pôs o limiar de `suspeita` em −4,6 na escala calibrada: aceitava planta com
evidência **contra** a doença. Na demonstração do SaaS, as três laranjeiras sadias
saíram `suspeita` com posterior de 0,1% a 1,7%. Com o piso, os citros passaram de 5/83
para 2/83 sadias com alarme sem perder nenhuma doente, e na banana nenhuma sadia sai
mais como `doente` (eram 8); as doentes como `doente` caíram de 95 para 30 — as outras
85 continuam sinalizadas, como `suspeita`. A `FusionCalibration` agora recusa limiar de
`suspeita` negativo.

| Banana, por doença | Calibrada | Fonte | Calibrada |
| --- | ---: | --- | ---: |
| Cordana | 18/20 | doentes do acervo (campo) | 98/107 |
| Fusariose | 41/44 | doentes do PlantSeg | 17/18 |
| Pestalotiopsis | 10/10 | sadias do acervo | 15/283 |
| Sigatoka | 46/51 | | |

Pela API, no talhão de ponta a ponta (as mesmas fotos montadas em outras plantas): banana
17/18 doentes e 3/18 sadias como `suspeita` (antes 9/18 e 1/18); citros 33/33 e 0/33
(antes 27/33 e 0/33); nenhuma sadia como `doente`. Em 2.000 trios sadios sorteados das
mesmas fotos, o alarme é 5,9% na banana (0,3% como `doente`) e 0,2% nos citros. Na
demonstração do SaaS (`simulator.pomar` com a IA de verdade), a situação da planta saiu
certa em 18 de 18 no bananal e 15 de 15 no pomar.

Parâmetros escolhidos: banana `tile_correlation` 1,0, corte da LLR 1,0, Platt
−0,10 + 2,65 L; citros 1,0, 2,0 e −0,93 + 6,43 L. Com correlação 1,0, os recortes de
um quadro valem uma vista só (a média deles): tratá-los como mais independentes não
melhorou a detecção em nenhuma das calibrações.

**Por que a razão de verossimilhança é um histograma.** A primeira tentativa estimou
sensibilidade e alarme falso por recorte pela média das probabilidades. Na banana
funcionou (9 → 14 de 18 doentes do PlantSeg), mas nos citros piorou o Greening de 7
para 4 de 8: o modelo põe, em média, 48% de probabilidade de Greening em recorte de
planta sem Greening, e pela média a evidência dele quase some. Só a cauda perto de 1
separa as plantas, e o histograma por faixa vê a cauda.

**Por que as doentes de papel ficaram fora dos citros.** O acervo do citros é de folha
destacada sobre papel; com as 315 plantas doentes dele na calibração, a detecção nas
fotos de campo do PlantSeg caiu de 33 para 25 de 33. A calibração sem elas, medida sem
reajuste nessas 315, ainda acha Canker em 33/40 e Greening em 29/40 — reage à doença, e
não à diferença entre papel e campo. As outras nove doenças do citros acham pouco no
papel (Leafminer 0/40, Mites 0/33, Curl Leaf 1/40) e herdam a tabela combinada.

### Incidência do talhão

`GET /fields/{talhão}/incidence`: a fração de **plantas** doentes, corrigida por
Rogan-Gladen com as taxas acima, com intervalo de 95% (Beta de Jeffreys nas três
proporções, com o tamanho efetivo das taxas). Medido em 8 divisões aleatórias em
metades: calibra numa, simula talhões com plantas da outra.

| Talhão de 1.000 plantas | 0% | 5% | 10% | 20% | 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| banana, erro médio (pontos) | 1,4 | 3,1 | 2,9 | 2,6 | 2,7 |
| banana, cobertura do IC95 | 95% | 83% | 85% | 94% | 97% |
| citros, erro médio (pontos) | 0,0 | 1,3 | 1,2 | 1,1 | 0,7 |
| citros, cobertura do IC95 | 100% | 100% | 100% | 100% | 100% |

Com 100 plantas, a cobertura da banana sobe para 96–100% (a contagem do talhão pesa
mais no intervalo). A fração aparente, sem correção, erra 3 a 6 pontos na banana. Os
citros saem quase perfeitos porque as sadias de papel quase nunca dão alarme (2/83) —
num pomar de verdade isso não está medido. A estimativa antiga, a EM sobre os **recortes**,
dava 60% de doença num talhão de bananeiras todas sadias e quase 100% num de citros:
ela mede a proporção de recortes, não de plantas.

### Limitações

- **Fotos agrupadas não são o rover.** As plantas da calibração são três fotos de
  folha; no pomar, o recorte vê folhas, galhos e fundo da mesma árvore. A validação de
  campo (README, [calibração da fusão](README.md#calibração-da-fusão)) mede isto com o
  registro do rover e o rótulo do agrônomo por fileira.
- **O alarme falso dos citros numa árvore de verdade não foi medido.** As 83 sadias são
  folhas sobre papel. Antes do piso nos limiares, com 83 sadias o próprio limiar era
  ruidoso (o alarme real variou de 0 a 17% entre metades sorteadas); com o piso, ficou
  entre 0 e 2,4% — no papel. A taxa que vale é a que a validação de campo medir.
- **Acertar a doença é o ponto fraco da banana**: 61 das 115 plantas detectadas saem
  com a doença certa. Sigatoka, Cordana e Fusariose se confundem por planta; a
  prescrição por doença herda esse erro.
- O PlantSeg é CC BY-NC 4.0, como nos modelos.

---

<a id="severidade"></a>
## Edge AI do Rover — limiares medidos em 07/10/2026

### Café no Rover — 07/10/2026

O café é a cultura do produto, e é a que mais muda no Rover:

* **Caminho com cabeça.** No café a CNN sozinha acerta 6,5% num acervo externo; quem
  diagnostica é a cabeça (`coffee_foliar.head.npz`) sobre o embedding da torre de visão do
  SigLIP 2. O Rover passou a rodar a torre em ONNX (`framing_vision.onnx`, 355 MB, levada
  como `<sha>.encoder.onnx` e conferida pelo SHA-256 do perfil) e a cabeça copiada byte a
  byte do `ia/` (`app/inference/heads.py`). Paridade Rover × IA em 24 folhas do BRACOL
  separado: diferença máxima 1,6e-5, mesma classe em 100%. Custo: 131 ms por recorte em CPU
  de PC (~9,8 s por quadro de 70 recortes); **na Jetson, a medir** (a torre vai para a GPU).
* **Fusão e zonas.** Café na fusão por planta (pé a 0,7 m, ruas de 3,5 m; o espaçamento do
  cadastro do talhão substitui o padrão) e os terços do cafeeiro como zonas.
* **Calibração por mosaico.** As fotos de café são folha em close: cortadas como 1 m de
  planta, cada recorte virava um pedaço ampliado de folha, e o Rover pegava 90 de 203
  plantas doentes (ferrugem 13 de 90 na fusão). A 270 mm o recorte de 150 mm mostra ~uma
  folha — o que a foto mostra. `fit_edge mosaico` monta o quadro do Rover com 70 folhas
  fotografadas (cada uma classificada uma vez pela cabeça), com a fração de folhas doentes
  da planta (5, 10 e 20%); as fotos são repartidas em 5 grupos sem repetição, e cada parte
  da validação cruzada é um grupo (fonte `dobra:N`).

Fora da amostra, com fotos que o ajuste não viu (teste do `coffee-foliar` + BRACOL separado,
883 fotos; acerto da cabeça por folha: ferrugem 95,3%, phoma 96,8%, bicho-mineiro 94,6%,
sadia 95,7%, cercosporiose 83,3%):

| | Folhas doentes 5% | 10% | 20% | Quadros sadios evento |
| --- | ---: | ---: | ---: | ---: |
| Rover, regra por evidência (orçamento de 5% **em cada** grupo de fotos) | 160/240 | 240/240 | 240/240 | 103/900 (11,4%) |
| Fusão da borda (`coffee_foliar.edge-fusion.json`), plantas | — | — | — | 12/300 sadias com alarme (4%); 631/720 doentes |

Com o orçamento só na média dos grupos, os quadros sadios fora da amostra iam a 20,6%: as
sadias vêm de 164 fotos, e as poucas folhas que a cabeça erra se repetem em todos os quadros
do grupo. A regra de anomalia fica **desligada** no café: medida em foto limpa, marcava quase
todo quadro sadio de cafezal montado como ANOMALIA.

Cadeia completa com a IA no ar (8 pés montados com folhas do BRACOL separado, 25% de folhas
doentes nos doentes): ferrugem e bicho-mineiro → `doente` com a doença certa (posterior 1,00);
trechos sadios → 6 `sadia`, 1 `suspeita` (0,28); prescrição de ferrugem: tratar 7 de 11
trechos, 89% do ganho de pulverizar tudo por 64% do custo. Mas 8 dos 15 quadros sadios
desse cafezal montado viraram evento (sobem em original): o alarme falso por quadro do café
**não é conhecido** — entre 11% e 50% nas duas montagens — e só a coleta de campo diz.

**Limite.** Nada aqui é o Rover no cafezal. A folha em close não é o recorte de folhagem
embaraçada, com galho, fruto e sombra. A primeira coleta no Sul de Minas deve rodar com o
perfil provisório (o custo do alarme falso é só subir mais original — o que a coleta quer),
o agrônomo rotulando os pés de uma fileira, e recalibrar com `fit_edge fit --labels` e
`fit_fusion fit --labels` sobre o registro do Rover na IA (`FIELDS_TILE_LOG=true`). O
mesmo dia vira treino: [`training/import_campo.py`](training/import_campo.py) corta os
originais exportados pela plataforma (`scripts/exporta_originais.py` do SaaS) nos recortes
do registro — recortes de pés rotulados sadios entram como `Healthy`; os de pés doentes vão
para `_revisar` com um manifesto, porque nem todo recorte de um pé doente mostra lesão.

### Banana e citros

No Rover a IA roda sem internet, só o ONNX (sem TTA, sem o portão de enquadramento e
sem as cabeças, que só existem na nuvem), e decide quadro a quadro o que sobe em
qualidade original primeiro. O artefato é `models/<pesos>.edge.json` (com o
`.categories.json`), gravado pelo [`fit_edge`](training/fit_edge.py) e copiado para o
Rover junto com o modelo (`valtexa-rover preparar-modelo`). Mesmas plantas da fusão
(três fotos por planta, uma por andar de câmera, recortes de 150 mm a 270 mm),
validação cruzada em 5 partes por planta, orçamento de 5% de quadros sadios virando
evento. Tudo fora da amostra.

| | Plantas doentes com ≥ 1 quadro evento | Quadros doentes evento | Quadros sadios evento |
| --- | ---: | ---: | ---: |
| banana, contagem de recortes (canto da grade: 3 recortes ≥ 0,99) | 125/125 | 298/373 | **260/779 (33%)** |
| **banana, evidência** | **99/125 (79%)** | 168/373 | **41/779 (5,3%)** |
| citros, contagem de recortes (3 recortes ≥ 0,99) | 33/33 | 98/98 | **165/249 (66%)** |
| **citros, evidência** | **33/33** | 84/98 | **15/249 (6,0%)** |

**Por que evidência e não contagem.** O modelo põe probabilidade de doença ≥ 0,99 em
pedaço de folha sadia com frequência: nenhum par (limiar, número de recortes) da grade
cabe no orçamento. A regra por evidência é a da fusão por planta num quadro só: cada
recorte vira a log-razão de verossimilhança da faixa da sua probabilidade (faixas mais
finas perto de 1: 0,98, 0,99, 0,995, 0,999), limitada; o escore é a maior, entre as
doenças, da média nos recortes com o desconto de correlação; evento acima do quantil
95% dos quadros sadios. Tabelas, corte e limiar são refeitos em cada parte de treino.
Escolhidos: banana corte 2,0, correlação 1,0, evento > 0,518; citros 2,0, 1,0, > 0,343.

**Fusão dos recortes do Rover** (`<pesos>.edge-fusion.json`, `fit_edge fusao`): a nuvem
funde por planta as probabilidades que o Rover manda (`POST /fields/{id}/frames`). Fora da
amostra (10 partes): banana 113/125 doentes com 15/283 sadias em alarme; citros 32/33 com
4/83 (contra 115/125 e 33/33 da calibração da nuvem, que tem o portão).

Tempo por quadro de 12 MP (70 recortes) em CPU de PC, sem GPU: 2,8 s na banana, 1,9 s
nos citros — ~1,9 s da rede e ~0,85 s do pré-processamento idêntico ao do treino. Na
Jetson a rede vai para a GPU (TensorRT); o pré-processamento no ARM tende a ser o gargalo.
**Não medido na Jetson.**

Limites: fotos de folha agrupadas, não o Rover no pomar; nos citros as sadias são folha
sobre papel e só 83 plantas (o alarme falso real pode variar bem mais que o medido). A
validação de campo deve remedir com o registro do Rover.

## Severidade — área lesionada no quadro

### O que responde

Quanto do quadro é lesão, em `[0, 1]`. Os classificadores dizem **qual** doença;
a severidade diz **quanto**. Sai no mesmo `/predict` dos modelos de folha de
milho (`leaf-foliar` e `leaf-foliar-b`), no campo `severity`, calculada na mesma
requisição.

É área **no quadro**, não na folha: aproxima a severidade da folha quando ela
preenche a imagem, que é a premissa dos modelos foliares. Numa foto com solo ou
céu, o número sai menor do que a severidade real. Medir sobre a área da folha
exigiria segmentar a folha também, e não há máscara pública de folha de milho
fotografada em campo.

### Ficha técnica

| | |
| --- | --- |
| Arquitetura | U-Net com encoder MobileNetV2 (ImageNet), saída sigmoide por pixel |
| Entrada | 224 × 224, escala `[-1, 1]` — o mesmo pré-processamento dos classificadores |
| Perda | entropia cruzada binária + Dice |
| Limiar | pixel com probabilidade acima de 0,5 conta como lesão |
| Pesos | `models/leaf_severity.keras`, com `leaf_severity.metrics.json` ao lado |

### Dados

- **PlantSeg, milho** — máscaras de lesão de ferrugem, cercosporiose e
  helmintosporiose, com o `train`/`test` do próprio dataset. `corn_smut` fica de
  fora: carvão ataca a espiga, e segmentá-lo ensinaria o modelo a marcar espiga
  como lesão foliar.
- **Folhas saudáveis do `leaf-foliar`, com máscara vazia** — sem elas o modelo só
  vê imagens com lesão e aprende a sempre achar alguma.

| | Treino | Teste |
| --- | ---: | ---: |
| PlantSeg, lesão foliar de milho | 332 | 82 |
| saudáveis, máscara vazia | 320 | 80 |

15% do treino vira validação, que escolhe a época; o teste não decide nada.

### Como treinar

```bash
python -m training.train_severity --plantseg ~/valtexa-raw/plantseg/plantseg
```

### Resultado medido — 11/09/2026

No teste do PlantSeg e nas saudáveis separadas, nenhum dos dois usado para
escolher época. "Erro" é a diferença absoluta entre a fração estimada e a da
máscara, em pontos percentuais do quadro.

| | n | Dice | Erro médio | Erro p90 | Correlação | Real → estimada |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **com lesão** | 82 | 0,68 | **7,6** | 22,7 | **0,89** | 28,7% → 25,4% |
| ferrugem | 35 | 0,80 | 7,1 | 22,6 | 0,90 | 39,3% → 37,1% |
| cercosporiose | 21 | 0,65 | 11,8 | 33,2 | 0,85 | 30,7% → 24,0% |
| helmintosporiose | 26 | 0,54 | 4,8 | 10,1 | 0,79 | 12,6% → 10,6% |

| Saudáveis | |
| --- | ---: |
| n | 80 |
| lesão falsa média | 0,01% |
| com mais de 5% de lesão falsa | 0 |

Leitura: o número acompanha a severidade real (correlação 0,89) e tende a
subestimá-la um pouco, mais na cercosporiose — cujas manchas pequenas e
retangulares se perdem na redução para 224 px. Um erro de 7,6 pontos serve para
separar "pouca", "moderada" e "muita" lesão; não serve para uma escala de 10
graus. A lesão falsa quase nula nas saudáveis é otimista: elas vêm da mesma
origem das saudáveis do treino, e uma folha sadia fotografada em campo, com
reflexo e sujeira, ainda não foi medida.

### Limitações da severidade

- Área no quadro, não na folha (acima).
- Poucas centenas de máscaras de treino: a máscara pública é o gargalo, não a
  arquitetura.
- Não é a escala diagramática de um fitopatologista, e ainda não foi comparada
  com uma. Por isso a plataforma mostra o número ao agrônomo e **não** o usa na
  regra de decisão.
- Um canal só: numa folha com duas doenças, a área é a soma das duas.

---

<a id="zoom-e-deteccao"></a>
## Zoom digital vs. óptico, e detecção por lesão

Pergunta recorrente: "não dá pra voar mais alto e dar zoom depois?" A resposta
depende de qual zoom.

### Zoom digital (recortar e ampliar depois) — não resolve

Se a foto é tirada larga e você recorta + amplia no software depois, nenhuma
informação nova é criada — só se interpola os pixels que já existiam. Se a
lesão ocupava 3 pixels na foto original, ela continua sendo 3 pixels de
informação real depois de "ampliada" para 300, só que borrados. Isso não muda a
altitude mínima de voo.

### Zoom óptico (lente que magnifica de verdade) — resolve, mas muda o modelo

Câmeras de inspeção industrial (linha de transmissão, gasoduto) fazem
exatamente isso: voam a distância segura e usam zoom óptico real para trazer o
detalhe. Se o drone carregar uma câmera com zoom óptico de verdade — não "zoom
digital" do aplicativo —, dá para voar mais alto/seguro e ainda resolver a
lesão.

O que o zoom óptico não resolve sozinho é o enquadramento: com campo de visão
estreito, mirar a folha inteira em voo vira impraticável. O que ele de fato
habilita é trocar de arquitetura — de **classificar a folha inteira** para
**detectar a lesão** dentro de uma imagem mais ampla (YOLO/detecção de objetos
em vez do softmax do `leaf-foliar`). Isso tolera muito melhor variação de
distância e ângulo, porque só precisa da lesão resolvida, não da folha
enquadrada com precisão.

### Feasibilidade por tamanho de lesão

| Doença | Tamanho típico da lesão | Zoom/distância necessária |
| --- | --- | --- |
| `Blight` (NLB) | Grande, formato de charuto, vários cm | **Mais fácil** — GSD de poucos mm/pixel já resolve |
| `Gray_Leaf_Spot` | Retangular, 1-3 cm | Intermediário |
| `Common_Rust` | Pústulas de 2-5 mm | Difícil — precisa de GSD sub-milimétrico |

### O dado real que existe: como o único dataset de drone+doença em milho foi capturado

O `disease-nlb-drone` (mesma origem do `nlb-cornell`) não é hipotético — os
parâmetros de voo usados para capturá-lo estão publicados: **DJI Matrice 600,
altitude de 6 m, velocidade de 1 m/s, visão nadir, um disparo a cada 2
segundos**. Com esses parâmetros, especialistas conseguiram anotar lesões de
NLB nas imagens depois de capturadas — ou seja, o sensor tinha resolução
suficiente para o *zoom digital posterior* funcionar, porque a captura já era
de alta resolução o bastante desde o início. Isso é a validação prática de que
"sensor de alta resolução + recorte digital depois" funciona **quando a doença
já é grande o bastante** (NLB) — não é mágica de zoom, é sensor bom e distância
calibrada para o tamanho da lesão-alvo.

Isso também refina uma resposta anterior desta documentação: 6 m de altitude
(bem mais que a estimativa inicial de "quase encostando na folha") já é
suficiente **para NLB especificamente**, com câmera de boa resolução e voo
lento/nadir. Para ferrugem (pústulas de poucos mm) essa distância não deve
funcionar sem zoom óptico real.

### O mesmo dado, agora para café: existe pesquisa publicada de verdade

Achado que muda a resposta anterior sobre café: **já existe estudo publicado de
detecção de ferrugem do cafeeiro por drone RGB**, não só o leaf-foliar de perto.
[Robust Coffee Rust Detection Using UAV-Based Aerial RGB Imagery](https://doi.org/10.3390/agriengineering5030088)
(*AgriEngineering*, 2023) voou **a ~2,8 m de altura, seguindo a direção das
linhas de plantio**, com câmera RGB comum, e conseguiu **93,5% de detecção** nas
imagens de drone (97% no RoCoLe, que é de perto). Uma segunda linha de pesquisa
— [vegetation indices + decision tree](https://dl.acm.org/doi/abs/10.1016/j.compag.2021.106476) —
usa câmera **multiespectral** (Mapir Survey3W) e índices como NDVI/GRVI/GNDVI em
vez de classificar a imagem RGB direto, chegando a F1 0,90.

**Isso é briefing acionável para o piloto desta semana com café**: ~2,8 m de
altura, voando ao longo da rua de plantio (não nadir puro por cima da copa,
que veria só o topo), é o dado publicado mais próximo do seu caso de uso.

O que não achamos: as imagens de drone dessas pesquisas **não parecem ter sido
liberadas como dataset público separado** — o link de dados que os autores citam
volta para o RoCoLe (que já temos). Trate os números acima como validação de
viabilidade e parâmetro de voo, não como fonte de treino adicional.

### Datasets que já vêm com anotação de lesão (bounding box), não só rótulo de imagem

- **PlantDoc** — ~8.800 anotações de bounding box, feito para detecção.
- **`disease-nlb-drone` / `nlb-cornell`** — anotação por lesão, em campo, com
  subconjunto de drone isolado.
- **PlantSeg** — máscara de segmentação por lesão (mais fina que bounding box;
  dá para converter máscara → caixa).

### Recomendação prática

Não é trabalho para esta semana — exige nova arquitetura (YOLO/Ultralytics),
novo formato de anotação e novo schema de resposta (caixas, não só uma classe
por imagem). Fica como item de roadmap real, com o caminho de dados já mapeado
acima, e não como pré-requisito do piloto comercial.

---

<a id="datasets"></a>
## Catálogo de datasets — milho e aéreo/terrestre

> Números confirmados por busca nas publicações originais em setembro de 2026;
> confira a versão exata na fonte ao baixar, porque várias dessas bases têm
> mais de uma revisão em circulação.

### Diretamente úteis — milho

| Chave | Dataset | Escala | Condição | Serve para |
| --- | --- | --- | --- | --- |
| `corn-maize-leaf-disease` | [Corn or Maize Leaf Disease](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) (Kaggle) | ~4.200 imgs, 4 classes | Misto | **Base do `leaf-foliar`.** Já usa os nomes canônicos |
| `plantvillage-corn` | [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) — recorte de milho | ~3.850 imgs (de ~54k totais) | Laboratório | Volume. **Cuidado**: fundo uniforme |
| `plantdoc-corn` | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) — classes de milho | ~2.600 imgs totais, ~8.800 caixas | **Campo** | Robustez. Tem versões YOLO |
| `cd-and-s` | CD&S — Corn Disease & Severity (Purdue, *Data in Brief* 2021) | Campo, com máscaras | **Campo** | Severidade e segmentação |
| `plantseg-corn` | Recorte de milho do [PlantSeg](https://zenodo.org/records/13762907) | Parte dos ~11.400 imgs doentes do PlantSeg | **In-the-wild** | Alternativa mais realista ao PlantVillage |

**A armadilha do PlantVillage.** É o dataset mais usado da área e o mais
enganoso: fotos de folha destacada, fundo uniforme, luz controlada. Modelos
treinados só nele chegam a >99% de acurácia e desabam em campo — o modelo
aprende o fundo, não a lesão. Use para volume, **sempre misturado** com PlantDoc,
CD&S ou PlantSeg, e nunca valide só nele.

### PlantSeg — o substituto de fundo real do PlantVillage

[PlantSeg](https://zenodo.org/records/13762907) ([artigo, *Scientific Data*
2025](https://www.nature.com/articles/s41597-025-06513-4)) é mais recente e
resolve exatamente o problema acima: **mais de 11.400 imagens doentes** (e cerca
de 8.000 saudáveis) cobrindo **115 combinações planta-doença em 34 espécies**,
majoritariamente capturadas **in-the-wild** — fundo complexo, luz variável,
oclusão — em vez de laboratório. Cada imagem tem **máscara de segmentação** da
área doente (PNG), não só um rótulo de classe. Licença CC BY-NC 4.0, splits
70/10/20 prontos.

Entre as 34 espécies estão milho e **café** — daí ele aparecer tanto em
`leaf-foliar`/`plantseg-corn` quanto em `coffee-foliar`/`plantseg-coffee`. Não é
um dataset de drone (as imagens não têm metadado de captura aérea), então
complementa mas não substitui os datasets aéreos abaixo.

### Aéreos / drone

| Chave | Dataset | Observação |
| --- | --- | --- |
| `disease-nlb-drone` | [DiseaseNLBdrone (OPIA)](https://ngdc.cncb.ac.cn/opia/dataset/datasets?dataId=38) | Subconjunto **só-UAV** do dataset abaixo — 7.669 imgs, 42.117 lesões, DJI Matrice 600. **Baixe este, não os dois** |
| `nlb-cornell` | [Maize NLB](https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-018-3548-6) (Wiesner-Hanks et al., *BMC Res Notes* 2018) | Coleção completa: 18.222 imgs / 105.705 lesões, câmera de mão + boom + UAV. `disease-nlb-drone` é a fatia UAV desta mesma coleção |
| `weed2c` | [WEED2C](https://github.com/EvertonTetila/WEED2C-Dataset) — soja + 2 daninhas | UAV Phantom 4 Advanced, **10 m de altitude**, 20 MP, lavouras brasileiras (Caarapó/Dourados-MS), 1.140 imgs, estádios V3-V5. Tem [versão YOLOv11 pronta no Roboflow](https://universe.roboflow.com/tetila/weed2c-dataset). **Melhor referência de captura por drone em lavoura brasileira** que existe |
| `droneweed` | [DRONEWEED](https://doi.org/10.20350/DIGITALCSIC/16559) — milho + tomate, 7 espécies de daninha | UAV, câmera Sony ILCE-6300L, **11 m de altitude**. **67.558 imagens** (31.002 em estádio inicial V4 + 36.556 em V7). Espanha — espécies de daninha não são 100% as mesmas do Brasil, mas várias são cosmopolitas (*Chenopodium album*, *Convolvulus arvensis*, *Sorghum halepense*). É de longe o **maior dataset público de daninha por drone em milho** |
| `agriculture-vision` | [Agriculture-Vision](https://github.com/SHI-Labs/Agriculture-Vision) | 94.986 tiles aéreos de 3.432 lavouras dos EUA (majoritariamente **milho e soja**), 2017-2019, **avião de asa fixa** (não multirotor pequeno), RGB+NIR, até 10 cm/pixel. Anotado por **anomalia** (falha de plantio, deficiência nutricional, dano de tempestade, `weed cluster` etc.), não por doença específica — bom para treinar a *forma* de "algo está errado" no dossel |
| `pdt-uav` | PDT — Pests and Diseases Tree | UAV real, mas **copa de árvore, não milho**. Vale pela metodologia de tiling e altitude, não pelos pesos |
| `cotton-weed` | [Cotton Weed Detection](https://huggingface.co/datasets/Project-AgML/cotton_weed_detection) | UAV, algodão + daninha, ~50.800 bounding boxes em 262 imgs. Não é a cultura certa, mas é referência de detecção por drone pronta pra YOLO |

**Sobre daninhas especificamente**: `weed2c` (soja, Brasil) + `droneweed` (milho,
maior volume) juntos cobrem as duas culturas de linha do catálogo com dataset de
drone de verdade — é o par de referência pra quando o item "modelo de daninhas"
do roadmap sair do papel.

**Sobre `disease-nlb-drone` vs `nlb-cornell`**: pelas contagens publicadas
(7.669/18.222 ≈ 42% das imagens, 42.117/105.705 ≈ 40% das lesões), tudo indica
que é o mesmo estudo, com a OPIA hospedando só a fatia capturada por drone.
Confirme antes de baixar os dois — provavelmente é redundante.

### Terrestres / robô de campo

Não há dataset público de milho sob dossel. Estes servem como **referência
metodológica** para montar o seu:

| Dataset | Por que olhar |
| --- | --- |
| Sugar Beets 2016 (Bonn) | Imagens de robô agrícola em campo — protocolo de captura e metadados |
| PhenoBench | Segmentação semântica em campo real, bem documentado |
| CropAndWeed | Grande volume em condição real de campo |
| DeepWeeds (17.509 imgs, 8 espécies) | Daninhas. Fonte de negativos para o modelo de daninhas do roadmap |
| Field-PDD | Imagens em condição natural (soja, trigo), fundo complexo — pequeno, auxiliar |
| IP102 | ~75.000 imgs, 102 espécies de praga, ~19.000 com bounding box. Pré-treino para um futuro modelo de pragas |

### Ferramentas

| Recurso | Para quê |
| --- | --- |
| [AgML](https://github.com/Project-AgML/AgML) | Biblioteca Python que agrega dezenas de datasets agrícolas com loader unificado. **Economiza dias** de wrangling |
| [digital-agriculture-datasets](https://github.com/) | Lista curada pela comunidade, separada por tarefa |
| Embrapa Digipathos / PDDB | Base brasileira, campo real, várias culturas. Relevante para o mercado BR |

---

<a id="datasets-cafe"></a>
## Catálogo de datasets — café

| Chave | Dataset | Escala | Condição | Observação |
| --- | --- | --- | --- | --- |
| `jmuben-coffee` | [JMuBEN/JMuBEN2](https://www.sciencedirect.com/science/article/pii/S2352340921004261) | 58.549 imgs — Healthy 18.961, Miner 17.000, Rust 8.336, Cercospora 7.681, Phoma 6.571 | Fazenda única (Kirinyaga, Quênia), câmera digital | **Base do `coffee-foliar`.** Resolução baixa (128×128) — trate como piso, não meta |
| `rocole` | [RoCoLe](https://data.mendeley.com/datasets/c5yvn32dzg/2) | 1.560 imgs | **Campo real**, smartphone a 200-300 mm, robusta (não arábica) | Pequeno, mas com variação genuína de luz/fundo/umidade — anotação de severidade |
| `plantseg-coffee` | Recorte de café do [PlantSeg](https://zenodo.org/records/13762907) | Parte do PlantSeg (34 espécies) | **In-the-wild** | Café listado entre as "culturas de lucro" do PlantSeg — melhor opção de fundo realista |
| `bracol` | [BRACOL](https://data.mendeley.com/datasets/yy2k5y8mxg/1) — Brazilian Arabica Coffee Leaf, **CC BY 4.0** | 1.747 folhas inteiras (1.401 recuperáveis, ver abaixo); rótulo por folha no `leaf/dataset.csv` | Folha destacada sobre **fundo branco**, vários celulares, Espírito Santo (BR) | **Usado pela cabeça do `coffee-foliar` desde 01/10/2026** (`training.import_bracol` -> `data/coffee-bracol`). 73 das folhas de phoma dele já estavam no `coffee-foliar` |

**Sobre café brasileiro**: `bracol` fecha a lacuna que antes ficava em aberto —
já existe dataset público **do Brasil** (Espírito Santo, mesma faixa de altitude
e clima montanhoso Arabica que o Sul de Minas). A pasta `leaf` traz um rótulo por
folha (estresse predominante: sadia, bicho-mineiro, ferrugem, phoma, cercospora; o
código 5, sem predominante, fica de fora), e é isso que a cabeça usa. O zip hospedado
no Mendeley vem **truncado** (medido em 01/10/2026: termina no meio de uma imagem,
sem diretório central); o `import_bracol --zip` lê as entradas completas pelos
cabeçalhos locais e recupera 1.401 das 1.747 folhas. A CNN do `coffee-foliar` não
foi retreinada com ele: quem aprende o BRACOL é a [segunda
leitura](#segunda-leitura-do-cafe). A coleta própria no Sul de Minas continua sendo o
que vira produto defensável — o BRACOL é folha destacada sobre fundo branco, não a
folha na planta que o rover vê.

---

## Prioridade de culturas

Ordem sugerida para investimento de coleta própria, cruzando a maturidade do
dataset público com a relevância para a região de atuação:

| Cultura | Dataset público | Coleta própria já vale a pena? |
| --- | --- | --- |
| **Milho** | Maduro (`leaf-foliar` treinável hoje) | Sim — para `rover-understory` e `aerial-canopy` |
| **Café** | Utilizável, mas de origem única/baixa resolução (`coffee-foliar` treinável hoje) | **Sim, é a maior prioridade** — nenhum público é brasileiro |
| **Soja** | Bom para daninha (`weed2c`), fraco para doença foliar específica | Vale começar por daninha, doença depois |
| **Trigo** | Só via PlantSeg + Agriculture-Vision, nenhum específico | Aguardar — dataset público ainda imaturo |

Recomendação: **não tente treinar as 34 espécies do PlantSeg de uma vez.** Café
+ milho já cobrem a demanda regional concreta; adicione soja/trigo quando o
pipeline de coleta e o `prepare_dataset` já estiverem rodando de forma repetível
para as duas primeiras culturas.

---

<a id="coleta"></a>
## Protocolo de coleta própria

Sem isto, `Stalk_Rot`, `Stalk_Borer_Damage`, `Nutrient_Deficiency` e `Corn_Stunt`
não existem — e o `coffee-foliar` continua limitado ao que o Quênia e o Equador
(RoCoLe é do Equador) ensinaram a ele. O protocolo abaixo vale para qualquer
cultura/classe nova, não só para o rover.

### Volume

Comece mirando **300–500 imagens por classe** para um primeiro modelo utilizável
com transfer learning. Abaixo de ~150 por classe o resultado não se sustenta.

Balanceie: classes novas tendem a ficar sub-representadas frente às que já têm
dataset público. Se a diferença passar de ~3×, use `class_weight` no treino.

### Metadados — grave desde a primeira imagem

É o que torna a fusão drone↔rover possível depois, e o que separa "dataset
genérico" de "dataset da lavoura que vocês atendem". Um JSON ao lado de cada
imagem (ou no EXIF):

```json
{
  "plot_id": "T-14",
  "lat": -22.4128, "lon": -45.4494,
  "timestamp": "2026-09-05T14:32:10-03:00",
  "platform": "ground",
  "crop": "coffee",
  "camera_height_cm": 45,
  "growth_stage": "florada",
  "row": 23,
  "light": "sombra_dossel"
}
```

Sem `plot_id` e GPS, drone e rover nunca conversam — e a camada de fusão é o
diferencial do produto.

### Diversidade obrigatória

Varie deliberadamente, senão o modelo aprende o atalho errado:

- **estádios fenológicos** (para café: florada, chumbinho, granação, maturação);
- **horários** — manhã, meio-dia, fim de tarde;
- **céu** — limpo e nublado;
- **cultivares e talhões** diferentes;
- **níveis de severidade**, não só o caso didático.

### Rotulagem

Rótulo confirmado por agrônomo. Para podridão de colmo, o padrão-ouro é abrir o
colmo — fotografe **antes** de abrir e rotule com o que a abertura revelou.

Registre também **quem** rotulou e o nível de certeza. Um rótulo duvidoso vale
menos que nenhum.

### Conjunto de teste

Separe **antes** de treinar, por talhão inteiro (nunca por imagem aleatória:
fotos da mesma planta nos dois lados do split inflam a métrica). Esse conjunto
nunca entra no treino. É o único número que você pode mostrar a cliente.

---

<a id="novidade"></a>
## O ciclo de dados próprio: aprender com o que o modelo não sabe

> É o mecanismo que transforma cada voo em dataset. Sem ele, o modelo fica
> congelado no que o Quênia e Cornell ensinaram; com ele, cada semana de
> operação deixa a lavoura brasileira dentro dos pesos.

### O problema que ele resolve

Um softmax é **fechado**: ele sempre devolve uma das classes que aprendeu. Diante
de cigarrinha, de uma deficiência nutricional, de uma doença fora do catálogo ou
de uma foto de solo, ele não diz "não sei" — diz `Common_Rust`, com confiança
alta. Essas imagens são exatamente as mais valiosas: são o que falta no dataset.

### Como funciona

A penúltima camada do modelo (`embedding`, 128 dimensões) é a assinatura visual
da imagem. Imagens da mesma classe caem perto umas das outras nesse espaço. Ao
fim do treino, o `fit_novelty` mede onde fica o centro de cada classe conhecida
e calibra um limiar. Em produção, cada imagem ganha um `novelty_score`:

| Score | Significado | Destino |
| --- | --- | --- |
| `< 1.0` | Parecido com algo conhecido | Diagnóstico normal |
| `> 1.0` | **Não se parece com nada que o modelo viu** | Fila do agrônomo |

O embedding sai da **mesma passada** que as probabilidades — a detecção de
novidade não custa uma segunda inferência.

**Uma covariância por classe, e um limiar por classe.** A primeira versão usava
uma covariância compartilhada, e ela quebrou quando o dataset cresceu: com
laboratório, campo e *in-the-wild* misturados, a mesma classe passou a ocupar
regiões distantes do espaço, a covariância comum inchou, e covariância inchada
encolhe a distância de **tudo** — inclusive do que é estranho. Cada classe agora
tem o seu próprio formato de nuvem, e o score é a distância dividida pelo limiar
*daquela* classe, o que impede que uma classe de nuvem larga vire um atrator.
Classes com menos amostras que dimensões recuam para a covariância compartilhada.

### Quanto ela realmente pega — medido em 08/09/2026

Sete entradas que o modelo de milho nunca deveria ver, com o dataset de 19.787
imagens:

| Limiar | Pega fora de escopo | Marca dado bom |
| ---: | :---: | ---: |
| | compartilhada → **por classe** | compartilhada → **por classe** |
| 1,0 | 1/7 → **3/7** | 0% → **0%** |
| 0,8 | 3/7 → **5/7** | 5% → **2%** |
| 0,7 | 4/7 → **6/7** | 23% → **10%** |

O caso que mais importava: uma foto de **colmo** era classificada como `Blight`
com **100% de confiança** e passava sem ser marcada. Agora é marcada.

**A novidade sozinha não basta — combine com `MIN_CONFIDENCE`.** Os dois filtros
pegam coisas diferentes: a novidade pega o que é visualmente estranho *com*
confiança alta (o colmo); o `MIN_CONFIDENCE` pega o que o modelo já responde
com dúvida (folha de café vira `Healthy` a 0,53). Juntos, com
`MIN_CONFIDENCE=0.7` e `NOVELTY_SENSITIVITY=1.0`, **6 dos 7 casos** são
interceptados. O que escapa dos dois é imagem de cor chapada com confiança
média — para esse caso não há configuração, só uma classe "fora de escopo"
treinada com negativos.

### O ciclo completo

```
drone/rover → POST /predict → needs_review? → review/pending/<data>/<id>.jpg
                                                     + <id>.json (palpite do modelo)
                                                          ↓
                                        agrônomo diagnostica e move o arquivo
                                                          ↓
                                        review/labeled/<Classe>/<id>.jpg
                                                          ↓
                              python -m training.prepare_dataset --source review/labeled
                                                          ↓
                                            data/<modelo>/<Classe>/
                                                          ↓
                                    python -m training.train --model <modelo>
                                    python -m training.fit_novelty --model <modelo>
                                                          ↓
                                 modelo novo, agora com dado da lavoura de verdade
```

A ação do agrônomo é literalmente **arrastar o arquivo para a pasta certa**. Todo
nome de classe do catálogo é reconhecido como nome de pasta pelo
`prepare_dataset` — não precisa de interface, e há teste garantindo essa
invariante.

### Ativando

```bash
# 1. depois de treinar, calibre uma vez por modelo
python -m training.fit_novelty --model coffee-foliar

# 2. no .env, ligue a fila
REVIEW_DIR=review
NOVELTY_SENSITIVITY=1.0
```

Sem o passo 1 o modelo funciona normalmente — só devolve `novelty_score: null` e
nunca marca nada. Sem `REVIEW_DIR` a API marca `needs_review` na resposta mas não
grava nada em disco (o cliente decide o que fazer). Guardar imagem de cliente é
decisão de quem opera, então o padrão é desligado.

### O incidente: a novidade do `leaf-foliar` ficou desligada de 08/09 a 26/09

Encontrado em 26/09/2026, ao medir o TTA. A referência
`corn_disease_model.novelty.npz` foi gravada em 08/09 às 22:25 — **depois** dos
pesos de 4 classes (19:48) — com 3 classes e 17.772 imagens: exatamente o
dataset sem `Gray_Leaf_Spot` do experimento da doença retida. A calibração do
experimento gravou por cima da de produção.

A API fez o que devia: recusou uma referência cujas classes não batem com as dos
pesos. Mas a recusa só aparecia numa linha de log de erro, `/health` respondia
normalmente, e o modelo principal serviu por semanas sem detecção de novidade —
`novelty_score` nulo e `needs_review` sempre falso.

Três mudanças para não repetir:

- o `fit_novelty` **recusa** calibrar com classes diferentes das do
  `.labels.json` dos pesos (antes, só avisava);
- `GET /models` expõe `safeguards` por modelo — `novelty`, `quality`,
  `conformal`, `framing` —, então uma referência recusada aparece como `false`
  na primeira consulta, não num log;
- a referência de 3 classes foi guardada como
  `corn_disease_model.novelty.npz.bak-3classes` e a de produção, recalibrada.

### Regulando o volume

`NOVELTY_SENSITIVITY` é o **orçamento do tempo do agrônomo**, não um parâmetro de
acurácia:

- `1.0` — usa o limiar calibrado (~1% das imagens de treino).
- `0.7` — marca bem mais, inclusive falso alarme. Bom no começo, quando você
  quer volume para rotular.
- `1.5` — só o que é gritantemente estranho. Bom quando a fila está maior que a
  capacidade de revisão.

Em campo a taxa é sempre maior que no treino — é dado que o modelo nunca viu.
Comece folgado, aperte conforme a fila encher.

### A variante mais agressiva, para os primeiros dias

O que descrevemos acima marca o que é diferente de **tudo que o modelo conhece**.
Se no início você quiser que o agrônomo veja **tudo que não é saudável** — para
bootstrap rápido do dataset —, isso não precisa de modelo nenhum: é
`predicted_class != "Healthy"` no cliente. Mais volume, menos seletividade. A
detecção de novidade é o que faz isso escalar quando o volume de voo crescer.

---

<a id="abstenção"></a>
## Abstenção por limiar de confiança

Um softmax sempre devolve uma classe. Foto de solo, de céu ou fora de foco sai
como doença com confiança alta — e errar com confiança na frente de um agrônomo
custa credibilidade.

```bash
MIN_CONFIDENCE=0.75    # no .env
```

Abaixo do limiar a resposta vem com `"abstained": true`. O cliente deve tratar
como **inconclusivo** e não exibir diagnóstico. No lote, `abstained_count`
resume quantas ficaram assim.

Padrão é `0.0` (desligado). Calibre pela distribuição de confiança do seu
conjunto de teste, não por chute — e por modelo: o `aerial-canopy`, otimizando
recall, deve usar limiar mais baixo que o `rover-understory`.

---

## Aprendizado contínuo da cabeça — medido em 15/10/2026

Como usar: README, "Aprendizado contínuo no campo". Aqui, a medição e o que ela ensinou.

**Protocolo** (`python -m training.mede_continuo`, CPU, ~7 s por rodada): cabeça de partida
treinada só no `coffee-foliar` (a dNATY de 01/10: 1176→64→256, GELU/ReLU); o BRACOL faz a
fazenda nova — 1.122 folhas depois de tirar 220 quase cópias do café (cosseno > 0,97), 30%
num teste final que ninguém vê, o resto em 3 rodadas de ~197 folhas e um lote de controle;
cada rodada é dividida 60/15/25 por grupo como o `adapt_campo` faz. Três sementes (42, 7,
2026: divisões diferentes do BRACOL). Memória: todo o treino do café (2.158, 1.000 vagas por
classe). Portão e seleção nos padrões do `adapt_campo`.

| estado final (média ± desvio, 3 sementes) | café | AF café | BRACOL | AF BRACOL | externo | rodadas aprovadas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| partida | 0,968 | 2,6% | 0,300 | 100% | 0,165 | — |
| ingênuo (só o campo) | — | — | — | — | — | 0/9 |
| ingênuo promovido à força | 0,813 ± 0,030 | 3,2% | 0,850 | 0% | 0,861 | (0/9) |
| replay simples | 0,967 ± 0,005 | 2,9% | 0,840 ± 0,011 | 3,6% ± 5,1 | 0,819 ± 0,041 | 3/9 |
| replay + destilação + L2-SP | 0,971 ± 0,001 | 2,0% ± 1,6 | 0,830 ± 0,042 | 0% | 0,706 ± 0,142 | 4/9 |
| **+ seleção na validação original** | **0,967 ± 0,003** | **1,2% ± 0,4** | **0,847 ± 0,015** | **0%** | **0,841 ± 0,035** | 4/9 |

Acurácia balanceada; AF = alarme falso em folha sadia; externo = `coffee-leaves-disease`
(só mede). Rótulos ruins (40% trocados, lote novo, depois das rodadas): 3/3 recusados.
Controle com os mesmos rótulos certos: 1/3 aprovado — depois de aprender o BRACOL, um lote a
mais quase não tem o que ensinar ("sem ganho").

**Referência: o treino conjunto.** A cabeça treinada de uma vez com café + BRACOL (01/10) e
a de produção sobem o AF no café de 2,6% para 4,3% e 5,2%. Não existe cabeça que aprenda o
BRACOL sem mexer em nenhuma folha sadia do café; o aprendizado contínuo com o portão terminou
abaixo das duas.

### O que a medição corrigiu no desenho (não repetir)

1. **Margem menor que a resolução recusa tudo.** O primeiro portão pedia o limite superior
   do IC do aumento de AF ≤ 1%. Com 116 folhas sadias no teste do café, duas folhas que mudam
   já põem esse limite em ~3%: recusou todas as candidatas boas, inclusive uma que *baixava*
   o AF de 2,6% para 1,7%. Hoje: não-inferioridade só no **agregado** (amostra que decide);
   AF, classes e conjuntos por piora demonstrada e margem pontual onde a amostra resolve.
2. **Conjunto pequeno não aguenta margem pontual.** O teste de uma rodada tem ~49 folhas; uma
   queda de 3% é uma folha e meia. Uma rodada foi recusada por "cair 6,6%" ali enquanto
   **subia** de 0,860 para 0,896 no teste final de 336 folhas. Por conjunto, só piora
   demonstrada.
3. **Validar no que a cabeça decorou não protege nada.** A primeira seleção de época usava
   15% da memória — exemplos do treino original, que a cabeça já tinha visto. Nunca recusou
   uma época (com e sem ela, as linhas saíam idênticas). Com a validação original (o split
   que escolheu a arquitetura no `fit_heads`) mais a validação de cada rodada promovida,
   contra não selecionar: AF no café 2,0% ± 1,6 → 1,2% ± 0,4, BRACOL 0,830 ± 0,042 → 0,847 ±
   0,015, externo 0,706 ± 0,142 → 0,841 ± 0,035 — menos variância em tudo.
4. **Memória: tamanho e destilação.** Com 300 vagas por classe (~60% do treino do café)
   nenhuma rodada de replay simples passou (0/3, uma semente, portão da época); com 1.000
   (todo o treino, 7,7 MB) a primeira passa. Mas o replay simples, mesmo com 1.000, esqueceu
   a rodada 1 nas seguintes (−3,6 e −8,2 pontos no teste dela) — é o que a destilação dos
   logits e a âncora seguram. Padrão: 1.000 vagas.
5. **Época 0 é a cabeça de partida, byte a byte.** Sem época melhor, a "candidata" saía com
   pesos regravados em float64 e outro SHA — o perfil da borda a trataria como cabeça nova.

**Limites**: o BRACOL são folhas inteiras fotografadas, não recortes do Rover; o primeiro
número de campo de verdade sai da primeira rodada com a validação do agrônomo. Três sementes
dão a ordem de grandeza, não um intervalo fino. A cabeça de partida e a arquitetura vêm do
dNATY (licença não comercial; ver a seção da cabeça). O `adapt_campo` em produção não usa o
dNATY; o motor com a plasticidade dele é opcional e foi medido abaixo.

### Plasticidade do dNATY no ciclo contínuo — medido em 18/10/2026

O dNATY 2.2 traz o *continual backprop* (Dohare et al., 2024): mede a utilidade de cada
neurônio oculto (`|ativação| × ||pesos de saída||`), renasce os menos úteis com pesos de
entrada novos e de saída zerados, e acompanha o rank efetivo da última camada. O
`training/motor_dnaty.py` remonta a `DynamicMLP` a partir da cabeça (BatchNorm identidade —
o inverso do `from_dnaty`; ida e volta com erro < 1e-4 nos logits), usa o tracker e o
`reinit_dormant` do próprio dNATY, devolve à cabeça só as linhas e colunas renascidas (o resto
com os mesmos bits) e treina com o `adapt` de sempre; o portão compara com a cabeça
instalada. `adapt_campo adapta --motor dnaty-plasticidade`.

Mesmo protocolo da medição de 15/10 (BRACOL como fazenda nova, 3 rodadas, 3 sementes;
`mede_continuo --estrategias proposto proposto_dnaty proposto_dnaty_puro`):

| | café, teste | AF sadia café | BRACOL, teste final | AF sadia BRACOL | externo | aprovadas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `proposto` (produção) | 96,7% ± 0,3 | 1,2% ± 0,4 | 84,7% ± 1,5 | 0% | 84,1% ± 3,5 | 4/9 |
| dNATY, só dormentes (< 10% da mediana) | 96,7% ± 0,3 | 1,2% ± 0,4 | 84,7% ± 1,5 | 0% | 84,1% ± 3,5 | 4/9 |
| dNATY, fração pura (5% por rodada) | 96,4% ± 0,2 | 2,6% ± 1,4 | 82,9% ± 3,7 | 2,5% ± 2,0 | 74,6% ± 4,9 | 4/9 |

* **Não há neurônio dormente** na cabeça do café: em nenhuma das 9 rodadas algum ficou
  abaixo de 10% da mediana da camada — o motor com limiar renasceu 0 e saiu idêntico.
* **Não há perda de plasticidade para remediar** em 3 rodadas: o rank efetivo da última
  camada **sobe** com a adaptação (≈ 58 → 70–75).
* Renascer os 5% menos úteis sempre (73 neurônios por rodada: 58, 3 e 12 nas três camadas)
  joga fora capacidade útil: piora o BRACOL, o externo (−9,5 pontos) e o alarme falso em
  sadia nas duas fazendas. O portão segurou o pior, mas o estado final é pior.
* Rótulos ruins: o portão recusou 3/3 também com o motor dNATY.

**Decisão**: produção continua no motor NumPy. O dNATY fica como opção para quando a
cabeça tiver muitas rodadas: o laudo registra o rank efetivo e os renascidos quando o motor
roda, e a fração pura não deve ser usada. Reproduzir: WSL,
`/home/pedro/medicoes/continuo/mede_continuo_v8_dnaty.{log,json}`.

## Café em vídeo real de lavoura — medido em 16/10/2026

**O teste.** 20 vídeos de café do YouTube (reportagens, closes de ferrugem, bicho-mineiro e
cercóspora, lavoura carregada vista de lado, colheita, plantio; dois clipes de música e um
jardim como controle), 1.024 quadros pela API e 511 pelo motor do Rover. **Só medição**: os
quadros não treinam nem calibram nada.

| | resultado |
| --- | --- |
| API (portão + cabeça + conformal) | 42% dos quadros passam no portão; desses, **89% saem Rust**, quase sempre com conjunto conformal de 1 classe |
| Motor do Rover (sem portão) | **511/511 quadros viram evento** (Rust em 473), inclusive os clipes de música |
| Percepção ativa | quase não age: o erro é confiante, não é dúvida. O alinhamento funciona em filmagem real: 88% dos pares a 0,4 s, 79% a 0,8 s |

Conferido a olho: quadros de lavoura, colheita e plantio chamados de doença não mostram
ferrugem (são cereja madura, pano de derriça, gente; uma xícara de café no coador saiu
"Rust 0,97"). Nos vídeos de doença a lesão aparece, mas tudo vira Rust — bicho-mineiro com
a larva à vista, cercóspora, mancha aureolada.

**Por quê** (medido): *Rust é o ralo da cabeça quando a foto piora* — folhas limpas do teste
degradadas como vídeo: cercóspora 98% → 35% (160 px, JPEG forte), bicho-mineiro 98% → 40%
(borrão de movimento), a diferença virando Rust. Recortes que o SigLIP chama de "fruto" saem
93% Rust (cereja madura). E o portão do café aceitava foto com 6,5% de probabilidade de folha.

### Portão do café: a cena vencedora tem de ser folha (instalado)

`FramingReference.leaf_winner_crops` (padrão do `fit_framing --folha-vencedora`: café):
além do limiar, a cena mais provável tem de ser folha (o escopo, mais "folha de outra
planta", que vence em ~10% das folhas de café boas). Calibrado só com fotos boas:

| regra | folha de café boa recusada | BRACOL recusado | quadros de vídeo aceitos |
| --- | ---: | ---: | --- |
| antes | 1,8% | 0% | 435 (96 de pessoa, 13 de fruto, 11 de planta inteira) |
| **cena vencedora é folha** | **3,4%** | **0%** | **298** — só cenas de folha |
| limiar no percentil 5 do café | 5,7% | 3,0% | 330 |

O portão tira o lixo; não conserta a cabeça: dos quadros de folha que passam, ~90% seguem Rust.

### Cabeça robusta à captura: três tentativas, nenhuma promovida

`training/robustez_captura.py`: cada foto do treino ganha 2 versões degradadas como vídeo
(resolução e JPEG, movimento, desfoque, exposição, ruído), a cabeça é ajustada com replay,
destilação e âncora, e o portão do aprendizado contínuo decide (teste limpo não pode piorar;
teste degradado — de fotos que a produção nunca treinou — tem de ganhar).

| tentativa | teste degradado | café limpo | cauda de doença em sadia | decisão |
| --- | ---: | ---: | --- | --- |
| 1 (sem suavização) | 0,698 → **0,850** | 0,970 → 0,964 | não medida pelo portão | APROVADA → promovida → **revertida** |
| 2 (suavização 0,1) | 0,698 → **0,884** | 0,970 → 0,964 | 0,7% → 0,7% | REPROVADA: ferrugem −5,2, mineiro −5,9 (piora demonstrada) |
| 3 (dose escolhida na validação) | 0,698 → **0,853** | 0,970 → 0,967 | 0,7% → **2,1%** | REPROVADA: cauda e não-inferioridade |

**A lição (não repetir): a cauda em folha sadia.** A tentativa 1 passou no portão de folha
— mesma acurácia por folha — e piorou o Rover. Recalibrado o perfil no mosaico (o mesmo
código e semente reproduzem a cabeça atual exatamente):

| Rover, mosaico fora da amostra | cabeça atual | tentativa 1 |
| --- | ---: | ---: |
| plantas doentes pegas | 640/720 | 615/720 |
| quadros sadios que viram evento | **103/900** | **163/900** |
| fusão por planta: detecta / alarme falso | 631/720 · 12/300 | **432/720 · 45/300** |

A cabeça atual nunca passa de 0,98; a robusta passa, e folhas sadias com P(doença) > 0,9 foram
de 1,2% para 2,4% — duas folhas. O Rover lê ~70 recortes por quadro: um recorte sadio com
doença "certa" vira evento, e a chance disso num quadro sadio sobe de ~57% para ~82%. O portão
ganhou o critério `GateCriteria.margem_cauda` (a cauda não pode subir mais de 0,5 ponto nem de
forma demonstrada) — foi ele que recusou a tentativa 3. A dose (suavização, destilação) agora
sai da validação (`selecao_na_validacao`), nunca do portão; parado em 3 tentativas para não
gastar o conjunto de teste.

**Estado de produção**: cabeça, perfil do Rover, fusão da borda e conformal **de antes**
(backup em `models/_backup-20261016/`); só o portão mudou. A memória do aprendizado contínuo
foi restaurada ao estado anterior à tentativa 1 (os logits gravados eram da cabeça revertida).

**O que resolve de verdade**: (1) uma saída "não é folha / fruto" na cabeça, treinada com
captura própria (cereja, céu, solo, caule, pessoa — YouTube não serve); (2) robustez à
captura medida e cobrada **no nível do Rover** (o mosaico no portão, não só por folha); (3) no
Rover, foto parada com exposição curta: a resolução de um recorte de 12 MP é muito maior que a
do vídeo, e o borrão é o que mais derruba o bicho-mineiro.

<a id="gemeo"></a>
## Gêmeo digital espaço-temporal — medido em 16/10/2026 (talhão sintético)

Como usar: README, "Gêmeo digital do talhão". Aqui, o desenho, a medição e o que ela corrigiu.
**Nada aqui foi medido em lavoura real**: o protocolo é sintético, com a verdade conhecida, e
o piloto é que vai dizer quanto o gêmeo acerta no campo.

**Protocolo** (`python -m training.mede_gemeo`, CPU): talhão de 20 ruas × 80 pés (1.600
plantas, 0,7 m × 3,5 m); ferrugem com 4 focos; epidemia **estocástica e binária** planta a
planta (cada pé em S, E, I ou R, sorteando as transições com a física do gêmeo e os vizinhos
que de fato esporulam — não o próprio modelo de campo médio); passada a cada 14 dias, 90% de
cobertura, em dois dias; Rover com sensibilidade 0,85 e alarme falso 3% por planta; fusão
**calibrada** (LLR gaussiano verdadeiro, prior da cultura 0,1). `--baixada 5` multiplica por 5
a suscetibilidade de 6 ruas × 30 m — o que é do talhão e a física não conhece. Três sementes.
O ajuste vê as k primeiras passadas e prevê a k+1, que não viu.

**Linhas de base**: persistência (a última passada repetida) e a heurística de contágio que a
prescrição já usa (`app.prescricao._risco`, alcance 3,0 / 1,5 m), as duas recalibradas
(Platt) no par anterior — o que um sistema em campo conseguiria. Métricas: log-verossimilhança
da evidência da passada por planta (maior é melhor; é a do próprio filtro), AUC do sinal novo
(entre as plantas que estavam limpas) e AUC da **infecção oculta** — entre as limpas, as que
estão de fato infectadas (latente ou esporulando), inclusive as que o Rover ainda não vê.

| cenário | passadas no ajuste | gêmeo | heurística do produto | persistência | AUC sinal novo (gêmeo / heurística) | AUC oculta (gêmeo / heurística) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| homogêneo | 5 | 0,114 | 0,114 | 0,105 | 0,67 / 0,70 | 0,81 / 0,84 |
| homogêneo | 7 | **0,183** | 0,177 | 0,168 | **0,70** / 0,68 | 0,84 / 0,85 |
| baixada 5× | 5 | 0,172 | **0,180** | 0,161 | 0,70 / **0,74** | 0,82 / 0,84 |
| baixada 5× | 7 | **0,282** | 0,280 | 0,258 | **0,72** / 0,70 | **0,85** / 0,84 |

"Gêmeo" é o ajustado com a seleção no par retido (abaixo). Leitura honesta: com 7 passadas o
gêmeo empata ou ganha nas três medidas, sem recalibração externa; com 5, a heurística
recalibrada no par anterior ainda leva vantagem, mais no talhão com baixada (a estrutura que a
física não conhece). As duas ordenam as plantas escondidas parecido (as duas usam contágio pela
vizinhança; a persistência, que não usa, fica em 0,53–0,56). O que o gêmeo dá e as bases não dão:
o **latente** (quantas plantas pegaram e ainda não mostram), o **tempo** (janelas de infecção
pela previsão do clima — o sintético tem tempo quase uniforme, então isso não aparece nesta
tabela), horizonte além de uma passada, e os **cenários** de aplicação.

**Seleção no par retido** (`app.gemeo.rede`): três candidatas disputam o último par de
passadas, que nenhuma viu — física pura, física recalibrada (Platt, dois números ajustados no
par anterior) e física com a rede residual. Nas 12 rodadas (2 cenários × 2 tamanhos × 3
sementes), a física ficou em 9, a rede em 3 (ganho retido +0,002 a +0,006 por planta) e a
recalibração em nenhuma. Onde a rede entrou, a previsão subiu até 0,04 (baixada, 7 passadas,
semente 1: 0,415 → 0,454) e uma vez perdeu 0,001 — o par retido é um só, e o portão erra
para os dois lados por esse tanto.

**Recuperação dos parâmetros** (homogêneo, média de 3 sementes):

| parâmetro | verdade | 5 passadas | IC95 cobre | 7 passadas | IC95 cobre |
| --- | ---: | ---: | ---: | ---: | ---: |
| β | 0,12 | 0,079 | 3/3 | 0,050 | 0/3 |
| alcance na rua | 1,8 m | 1,98 m | 3/3 | 2,22 m | 3/3 |
| alcance entre ruas | 0,9 m | 0,76 m | 3/3 | 1,07 m | 3/3 |
| fundo | 3·10⁻⁵ | 3,3·10⁻⁵ | 3/3 | 2,4·10⁻⁵ | 3/3 |
| alarme falso | 0,03 | 0,032 | 2/3 | 0,032 | 2/3 |
| sensibilidade | 0,85 | 0,957 | 2/3 | 0,964 | 0/3 |

Alcances, fundo e alarme falso saem certos. β sai baixo e a sensibilidade alta pelo erro de
modelo do campo médio (abaixo), e com mais passadas o intervalo encolhe em volta do valor
viesado: o IC95 de Laplace é **condicional ao modelo**, não a incerteza do mundo.

### O desenho que a medição impôs (não repetir)

1. **Estado em área foliar não liga com sinal binário.** A primeira versão carregava fração
   de área (S, E, I, R da folha) e invertia a posterior em ``I = -ln(1 - y)/κ`` a cada
   passada. Com posterior exata recuperava a verdade; com o sinal binário da fusão, zerava a
   planta doente que não foi sinalizada naquela passada, fazia de cada alarme falso um foco
   (3% de 1.600 pés = 48 "focos" contra 4 de verdade) e batia num teto; o ajuste compensava
   com β dez vezes menor e o alcance no limite, e κ era só calibre (não identificável). Hoje:
   S, E, I, R são a **probabilidade** de cada estado da planta (campo médio de um SEIR
   individual), P(sinal) = a + (s − a)·I, e a passada entra por **Bayes exato por planta**
   com a razão odds(posterior)/odds(prior da cultura) — por isso o
   `pass_observations` devolve o prior (`priors`).
2. **O teste não pode ser o próprio modelo.** Gerar o talhão com o campo médio e ajustar o
   campo médio media o modelo contra ele mesmo. O gerador é estocástico binário, e a fusão,
   calibrada (o gerador "Beta(8,2) se sinalizada" punha posterior 0,8 em qualquer planta
   doente, o que nenhuma fusão calibrada faz, e enviesava tudo).
3. **O limite do campo médio.** Com os parâmetros verdadeiros, o filtro previa P(esporulando)
   cinco vezes maior que a verdade nas plantas limpas: o campo médio põe em cada pé a força
   esperada de todas as vizinhas, e a reboleira de verdade é aglomerada. O ajuste compensa
   (β ~½). A correção da literatura, incidência não linear (ΣwI)^q (Liu, Hethcote & Levin,
   1987), foi testada e **recusada**: o ajuste levou q ao limite de cima (1,5) e a previsão
   piorou (0,113 → 0,062 com 5 passadas). Descontar os pares antigos no ajuste deu resultado
   misto (melhor no talhão com baixada, pior no homogêneo) e ficou desligado.
4. **Hessiana perto do limite.** Um parâmetro a menos de um passo do limite faz a diferença
   finita ler a penalidade de fora do domínio; a Laplace caía no prior e o conjunto sorteava β
   num fator 7 — AUC 0,58. Hoje a Hessiana é só nos parâmetros no interior **no ponto final**,
   com a diagonal como recurso. O objetivo é mal condicionado (o alarme falso é ~10⁴ vezes mais
   curvo que o resto): Nelder-Mead com reinício e polimento Levenberg-Marquardt.
5. **Um filtro só.** Ajuste, rede e previsão percorrem as passadas pelo mesmo gerador
   (`epidemia.filtrar`): a primeira versão tinha três laços, e um deles simulava de novo um
   ou dois dias por passada (a passada leva dias; o relógio recomeçava da mediana).
6. **O EM da prevalência colapsa** com poucas plantas e evidência fraca: de zero o Bayes não
   sai (três sinais seguidos no mesmo pé deixavam P(I) em 0,002). Piso de uma planta (1/n).
7. **A rede residual instável.** Com ~150 pesos e 4 a 6 pares de passadas, ganhava num par e
   perdia no seguinte (−0,05, +0,01, −0,01); ponderar os pares recentes piorou nos três. O
   portão recusa — é o que ele está lá para fazer. Daí a seleção com a recalibração de dois
   números (acima).
8. **"Identificado" não troca o modelo.** Com ganho abaixo do limiar, a primeira versão
   devolvia o prior **com a covariância do prior**, e o conjunto sorteava β num fator 7: a
   previsão de uma semente caiu de 0,111 para 0,022 por um ganho de 0,00099 por planta. O MAP
   com a Laplace é a posterior, movam os dados muito ou pouco; "identificado" só avisa. E o
   critério é o ganho **total** (3 nats), não por planta: 40.000 pés com 20 focos diluíam o
   ganho por planta e davam "não identificado" com os dados movendo o ajuste.

### Custo

A soma do contágio é exata em O(N) (`grafo.Vizinhanca`: o núcleo exponencial ao longo da rua
vira duas somas acumuladas em log). A primeira versão guardava as arestas até 12 m — num
talhão de 40.000 pés (~10 ha de café) eram 6,7 milhões (~215 MB) e 5,4 s por avaliação do
ajuste. Hoje, CPU, uma thread:

| talhão | avaliação do ajuste (4 passadas) | ajuste completo | previsão 21 dias, 16 membros |
| --- | ---: | ---: | ---: |
| 1.600 pés, 7 passadas | — | 20 a 30 s | ~1 s |
| 40.000 pés, 4 passadas | 0,36 s (era 5,41 s) | 229 s (com a máquina ocupada) | 9 s |

O SaaS espera até 15 min pelo ajuste e 5 min pela previsão e pela prescrição.

### O que falta para valer no campo

- **Piloto**: passadas reais de café com a calibração da fusão por planta, e a estação da
  fazenda (ou o Open-Meteo, que é reanálise: molhamento estimado por UR ≥ 90%).
- **Os presets são qualitativos** (faixas da literatura para ferrugem, cercosporiose, phoma e
  bicho-mineiro) e devem ser revistos pelo agrônomo; só β, alcances, fundo, alarme falso e
  sensibilidade são ajustados ao talhão.
- **Prescrição densa**: `app.prescricao` monta o núcleo n × n; a prescrição por risco limita
  as candidatas (1.500 de maior risco, com aviso).

<a id="enxame"></a>
## O gêmeo no enxame: fusão entre fontes e prioridade de inspeção — medido em 17/10/2026

O Nível 4 (coordenação de Rover, drone e base) mora no SaaS (`VALTEXA-SAAS/docs/ENXAME.md`).
A parte do gêmeo é esta: **várias fontes** num filtro só e **onde vale olhar**. Talhão
sintético (`python -m training.mede_enxame`), o gerador de `mede_gemeo`; 3 sementes.

**Canal por fonte** (`app/gemeo/epidemia.py`: `Canal`, `CANAIS`, `canal_de`). Cada passada tem
a `fonte` (`rover`, `drone`, `agronomo`) e entra no filtro com o canal dela — sensibilidade e
alarme falso. Dada a planta, as fontes veem independentes: a fusão é o produto das
verossimilhanças. O do Rover é ajustado ao talhão; os outros vêm da tabela — o do drone
(0,45 / 0,06) é **suposição de projeto**, não medida: a copa esconde o terço inferior, onde a
ferrugem começa. Evidência de outra fonte entra por planta em
`POST /fields/{id}/gemeo/observacoes` (LLR por doença); `fonte=rover` é recusada — a do Rover
já chega pela fusão, e entrar de novo seria contá-la duas vezes.

| previsão da passada seguinte do Rover | log-v (homogêneo) | AUC infectada oculta | log-v (baixada 5×) | AUC | sensibilidade do Rover ajustada (verdade 0,85) |
| --- | ---: | ---: | ---: | ---: | ---: |
| só o Rover (a cada 14 dias) | 0,142 | 0,838 | 0,321 | 0,914 | 0,94–0,97 |
| Rover + drone (a cada 7), cada um com o seu canal | **0,152** | **0,861** | 0,333 | **0,927** | 0,89–0,96 |
| Rover + drone, os dois com o canal do Rover | 0,152 | 0,855 | 0,338 | 0,916 | **0,64–0,77** |

A fusão ingênua parece igual na verossimilhança, mas ordena pior as plantas escondidas e
**estraga o canal do Rover**: para explicar os sinais fracos do drone, o ajuste rebaixa a
sensibilidade dele. É o argumento para o canal por fonte, e para medir o do drone antes de
confiar nele.

**Prioridade de inspeção** (`app/gemeo/prioridade.py`, `GET /fields/{id}/gemeo/prioridades`).
O valor da informação da decisão tratar / não tratar, por planta, somado por trecho de rua
(Rover) ou bloco (drone); a consequência inclui as vizinhas que a planta infectaria (pelo
núcleo ajustado). Sem os números do produtor, a informação mútua em bits. Inspeção simulada
no dia da visita, com a verdade, e a mesma regra de decisão para todas as políticas:

| valor realizado (R$ poupados contra não inspecionar) | orçamento | valor da informação | varredura em ordem | sorteio | só risco | só incerteza |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| limiar ~11%, homogêneo | 15% dos trechos | **64** | 53 | 45 | 13 | 13 |
| limiar ~11%, homogêneo | 30% | **120** | 98 | 74 | 33 | 38 |
| limiar ~11%, baixada | 30% | **19** | 11 | 5 | 1 | 2 |
| limiar ~4%, homogêneo | 30% | **27** | 9 | 9 | 4 | 1 |

Em 24 combinações (3 sementes × 2 economias × 2 talhões × 2 orçamentos) o valor da
informação nunca perdeu para nenhuma alternativa. **Só risco e só incerteza perdem até da
varredura**: mandam o Rover às plantas cuja decisão já está tomada (com 60% de risco e
sensibilidade 0,9, nem o "não vi nada" desce abaixo do limiar — trata-se de qualquer jeito).
O ganho absoluto é pequeno (2–4% da perda): o gêmeo de campo médio superestima o risco e manda
tratar quase tudo; com o gêmeo calibrado no campo, a inspeção deve valer mais — a medir.

**O que não fazer com a prioridade.** Ela desce para o Rover (dentro da missão assinada)
**só para decidir a bordo** — onde gastar a percepção ativa. Nunca como evidência: a mesma
observação voltaria ao gêmeo duas vezes, e a certeza cresceria sem dado novo.

## Inspeção adaptativa, dispersão pelo vento e aplicação localizada — medido em 19/10/2026

### Novidade no embedding da torre (o Rover reconhece quando não sabe?)

`app/inference/novidade_cabeca.py`, ajustado por `training/fit_novidade_cabeca.py` no treino do
café + 60% do BRACOL (a cabeça de produção treinou com os dois), limiares no percentil 99 da
calibração (validação do café + 20% do BRACOL), 64 componentes fixados antes de olhar os vídeos.
Referência instalada: `models/coffee_foliar.novidade.npz` (sha256 `0fb62d71…`).

| conjunto | recortes acima do limiar 1 | acima de 1,2 (o do Rover) |
| --- | ---: | ---: |
| teste do café + BRACOL (fotos) | 0,7% | — |
| acervo externo de folhas de café | 0% | — |
| BRACOL separado, pelo caminho do quadro (redimensiona + JPEG) | **31%** (15/48) | **0%** (0/48) |
| vídeo do YouTube, todos os 7.665 recortes | 66% | 16% |
| vídeo, doença com ≥ 90% de confiança dentro do portão | 41% | 7% |

Escore mediano das folhas legítimas não vistas: 0,96; dos recortes de vídeo com doença
confiante: 0,97. **A novidade não separa o erro confiante em folha de folha nova legítima** — só
a cena que não é folha (com 1,2: dossel 46%, planta inteira 32%, solo 27%, inseto 26% dos
recortes; folha de café 11%). O limiar do Rover é 1,2, calibrado no café não visto; no ensaio
ponta a ponta com pesos reais, com limiar 1 os originais guardados dobravam (29 → 58), com 1,2
voltaram a 29 (34/34 passos OK). O erro confiante em folha de vídeo continua sendo assunto de
dado de campo validado (aprendizado contínuo), não de novidade.

### Dispersão pelo vento (`app/gemeo/dispersao.py`, `training/mede_dispersao.py`)

Talhão sintético (30 ruas x 100 plantas), SEIR estocástico com parte do contágio pelo ar e vento
de oeste atravessando as ruas; cada modelo ajusta o seu β numa semana e prevê quem pega na
seguinte (estado verdadeiro de quem esporula; 5 sementes):

| talhão | modelo | log-perda | AUC |
| --- | --- | ---: | ---: |
| com vento | contato (o núcleo de hoje) | 0,321 ± 0,065 | 0,815 ± 0,037 |
| com vento | vento certo | 0,265 ± 0,058 | 0,841 ± 0,040 |
| com vento | vento ao contrário | 0,313 ± 0,059 | 0,759 ± 0,033 |
| com vento | **ω escolhido pelo talhão** | **0,265** | **0,841** |
| sem vento | contato | 0,052 ± 0,010 | 0,972 ± 0,006 |
| sem vento | vento | 0,075 ± 0,015 | 0,928 ± 0,020 |
| sem vento | **ω escolhido pelo talhão** | **0,052** | **0,972** |

O vento ajuda quando existe e com a direção certa; vento errado é pior que nada; ar onde não há
atrapalha. Escolhendo ω ∈ {0; 0,3; 0,6; 0,9} pela verossimilhança da semana de ajuste, o
talhão acertou o verdadeiro nas 10 execuções (0,6 e 0). No gêmeo:
`POST /gemeo/ajuste?dispersao=true&azimute_rua_graus=` faz essa escolha depois do ajuste
(`ajuste.escolher_dispersao`); desligado por padrão. Dado sintético com a mesma física do
modelo: mede a peça, não que a ferrugem do café se espalha assim.

### PINN para a dispersão (`training/pinn_dispersao.py`)

Campo verdadeiro resolvido numericamente (D = 0,8 m²/s, λ = 0,04/s, vento conhecido, 4 focos),
40 armadilhas com 10% de ruído, erro do campo em 400 pontos não medidos (3 sementes):

| método | erro em D | erro em λ | erro mediano do campo (log) |
| --- | --- | --- | ---: |
| PINN (resíduo em C) | -25%, -1%, +141% | +46%, +79%, +147% | 0,10 / 0,16 / 0,18 |
| PINN (resíduo em log C) | -76%, -52%, -62% | -29%, -54%, -34% | 0,61 / 0,75 / 0,38 |
| rede só de dados | — | — | 0,06 / 0,15 / 0,17 |
| **ajuste pelo solucionador** (Nelder-Mead em log D, log λ) | **-0,6%, -1,3%, -0,3%** | **+0,3%, -1,9%, +1,1%** | **0,010 / 0,021 / 0,012** |

Com dado esparso, a PINN não recuperou os parâmetros de forma confiável e não superou a rede sem
física; o ajuste pelo solucionador numérico da mesma equação recuperou D e λ com erro de 1–2%.
É o caso otimista (sem erro de modelo, fontes e vento conhecidos). Para o café, faltam a série
de armadilha de esporos e o anemômetro no talhão, os mecanismos do patógeno e a comparação com
as passadas seguintes — até lá D, λ e ω ficam como hipótese.

## Convenções compartilhadas

Valem para os quatro modelos.

**Pré-processamento idêntico ao treino.** Redimensionamento bilinear para o
quadrado, sem preservar aspect ratio, e escala `[-1, 1]`. Treino e inferência
importam a mesma função de
[`app/inference/preprocess.py`](app/inference/preprocess.py) — divergir aqui não
gera erro, só derruba a acurácia em silêncio.

**TTA igual na avaliação e na API.** As acurácias "+ TTA" deste documento vêm de
quatro orientações (original, espelho horizontal, vertical e as duas) com a média
das probabilidades. A API aplica exatamente as mesmas com `?tta=true` ou
`INFERENCE_TTA=true` — sem isso, o número que vale em produção é a linha "sem
TTA" de cada tabela. Ver [TTA em produção](README.md#tta-em-produção).

**Qualidade e atenção são medidas por modelo.** `fit_quality` grava
`models/<pesos>.quality.json` com os limites de nitidez, brilho e contraste do
próprio dataset de treino; `audit_attention` mostra, por classe, se a decisão
depende de uma região da imagem ou da foto inteira. Rode os dois a cada pesos
novos, como o `fit_novelty`.

**Os rótulos vêm do `.labels.json`.** O treino grava a ordem real das classes ao
lado dos pesos; a API lê esse arquivo. A lista no `registry.py` é só fallback.
Se a contagem de rótulos não bater com as saídas do modelo, o carregamento é
recusado — servir classe trocada é pior que não servir.

**Treino em duas fases.** Cabeça com a base congelada (LR 1e-3), depois
fine-tuning da base (LR 1e-5). O checkpoint da fase 2 só sobrescreve o da fase 1
se realmente melhorar.

**Modelos independentes.** Cada um carrega e falha sozinho: `leaf-foliar` e
`coffee-foliar` podem estar em produção enquanto os outros dois respondem `503`.

**`prepare_dataset` reporta, nunca adivinha.** Uma pasta de origem cujo nome não
está mapeado em `SOURCE_CLASS_ALIASES` aparece na lista de ignoradas — nunca é
encaixada por palpite. Vale também para os datasets de café, cujos nomes de
pasta reais nós não inspecionamos.

---

## Roadmap

- [ ] Treinar `leaf-foliar` e `coffee-foliar`, medir cada um num conjunto de teste próprio.
- [x] Segunda arquitetura do `leaf-foliar` para a dupla validação, com a
      independência medida — ver [`leaf-foliar-b`](#leaf-foliar-b).
- [ ] Repetir a medição de independência com imagens da própria frota.
- [x] TTA na API, com as mesmas quatro orientações da avaliação — ver
      [TTA em produção](README.md#tta-em-produção).
- [ ] Medir quanto `view_agreement < 1` pega de erro no teste isolado, ao lado da
      discordância entre `leaf-foliar` e `leaf-foliar-b`.
- [x] Verificação de qualidade da captura, calibrada pelo dataset de cada modelo —
      ver [qualidade da captura](README.md#qualidade-da-captura).
- [ ] Recalibrar `fit_quality` sobre os datasets consolidados e deduplicados.
- [x] Explicação por oclusão (`/predict/explain`) e auditoria de atenção por classe.
- [ ] Rodar `audit_attention` no `rover-understory` em quarentena — é o caso de
      controle: se ela não denunciar o atalho da sessão de fotografia, não serve.
- [x] Severidade por segmentação de lesão, como área no quadro — ver
      [severidade](#severidade).
- [ ] Severidade sobre a área da folha (segmentar a folha além da lesão) e
      comparação com a escala diagramática de um fitopatologista.
- [ ] Coleta própria de café no Sul de Minas — maior prioridade de diferenciação.
- [ ] `aerial-canopy` v1 a partir do `disease-nlb-drone` + tiling.
- [ ] Coleta de campo para as 4 classes exclusivas do rover.
- [ ] Migrar de classificação de folha inteira para **detecção por lesão**
      (YOLO), usando PlantDoc + `disease-nlb-drone` + `bracol` + PlantSeg — ver
      [zoom e detecção por lesão](#zoom-e-deteccao).
- [ ] Modelo de daninhas em milho e soja a partir do `droneweed` (67k imgs) +
      `weed2c` (já tem versão YOLOv11 pronta) — as duas culturas de linha
      cobertas com dataset de drone real.
- [ ] Testar o parâmetro de voo publicado para café (~2,8 m, seguindo a rua de
      plantio) no piloto comercial desta semana.
- [ ] Tronco compartilhado com múltiplas cabeças, em vez de modelos independentes por cultura.
- [ ] Trocar softmax por sigmoid multi-label — uma planta pode ter duas doenças.
- [ ] Camada de fusão espacial: risco do drone → tasking do rover → diagnóstico confirmado.
- [ ] Série temporal por talhão: progressão da doença entre voos.
- [ ] Modelo de pragas a partir do IP102, calibrado depois com dado próprio.
