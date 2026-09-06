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
| [`leaf-foliar`](#leaf-foliar) | Milho | qualquer | 4 | **Treinável hoje** | Piloto comercial |
| [`coffee-foliar`](#coffee-foliar) | Café | qualquer | 5 | **Treinável hoje** | Diferencial estratégico |
| [`rover-understory`](#rover-understory) | Milho | rover | 8 | Precisa de coleta | Produto principal |
| [`aerial-canopy`](#aerial-canopy) | Milho | drone | 2 | Precisa de coleta | Triagem |

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
python -m training.prepare_dataset --model leaf-foliar \
    --source "C:/datasets/corn-or-maize-leaf-disease" \
    --source "C:/datasets/PlantVillage" \
    --dry-run                      # confira o plano antes

python -m training.prepare_dataset --model leaf-foliar \
    --source "C:/datasets/corn-or-maize-leaf-disease" \
    --source "C:/datasets/PlantVillage"

# 3. treine
python -m training.train --model leaf-foliar --epochs 15 --fine-tune-epochs 5
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

### Limitações

- Exige a folha preenchendo o quadro. Em imagem nadir a média/grande altitude
  ele não funciona (ver [zoom e detecção por lesão](#zoom-e-deteccao)).
- Softmax de 4 classes: uma folha com duas doenças ao mesmo tempo é forçada a uma só.
- Sem classe "outro": foto de solo ou de céu sai como doença. Use `MIN_CONFIDENCE`
  (ver [abstenção](#abstenção)).

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

python -m training.train --model coffee-foliar --epochs 15 --fine-tune-epochs 5
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

### Por que este é o modelo estratégico

Cruzando com a leitura de mercado: café puxa turismo/renda para boa parte do Sul
de Minas, a ferrugem é a doença mais economicamente relevante da cultura no
Brasil, e não existe concorrente óbvio já servindo diagnóstico por imagem
calibrado para lavoura brasileira. Um modelo público (JMuBEN + RoCoLe) tira o
produto do zero esta semana; **10-20 mil imagens próprias, capturadas em
lavouras do Sul de Minas com drone/rover**, é o que transforma isso de "um YOLO
treinado em dataset público" em vantagem real e defensável.

### Limitações

- As mesmas do `leaf-foliar`: exige folha preenchendo o quadro, softmax de
  classe única, sem categoria "outro".
- `Leaf_Miner_Damage` mistura semanticamente doença e dano de inseto na mesma
  cabeça de classificação — funciona porque o dataset de origem já assim o
  trata, mas vale revisar se um modelo de pragas separado (ver `IP102` no
  catálogo) compensar mais à frente.
- Nenhum dos três datasets públicos foi capturado no Brasil. Trate a
  `val_accuracy` deles como piso, não como meta.

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
| `bracol` | [BRACOL](https://data.mendeley.com/datasets/yy2k5y8mxg/1) — Brazilian Arabica Coffee Leaf | 1.744 imgs, bounding box, 4 classes (Cercospora, Miner, Phoma, Rust) | **Campo, Espírito Santo (BR)** | Confirmado nesta rodada — é o dataset brasileiro de café que faltava. **Formato de detecção (YOLO), não serve ao classificador atual** — ver [zoom e detecção por lesão](#zoom-e-deteccao). Já baixado em `downloads/_bracol-raw/` |

**Sobre café brasileiro**: `bracol` fecha a lacuna que antes ficava em aberto —
já existe dataset público **do Brasil** (Espírito Santo, mesma faixa de altitude
e clima montanhoso Arabica que o Sul de Minas). A ressalva que continua valendo:
é anotação por bounding box, então só entra em produção quando o item de
detecção por lesão do roadmap sair do papel — para o `coffee-foliar` de
classificação, os três primeiros datasets continuam sendo a base, e nenhum
deles é brasileiro. A coleta própria no Sul de Minas continua sendo o que vira
produto defensável.

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

## Convenções compartilhadas

Valem para os quatro modelos.

**Pré-processamento idêntico ao treino.** Redimensionamento bilinear para o
quadrado, sem preservar aspect ratio, e escala `[-1, 1]`. Treino e inferência
importam a mesma função de
[`app/inference/preprocess.py`](app/inference/preprocess.py) — divergir aqui não
gera erro, só derruba a acurácia em silêncio.

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
