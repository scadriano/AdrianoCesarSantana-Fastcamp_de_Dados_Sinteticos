# Detecção de talheres com dados sintéticos gerados no Blender

Projeto final do **Fastcamp de Dados Sintéticos para IA e Visão Computacional**, desenvolvido por **Adriano César Santana**. O projeto implementa um pipeline completo para geração de imagens sintéticas de garfos, facas e colheres no Blender, anotação automática no padrão YOLO, treinamento de um detector YOLOv8n e avaliação em um conjunto sintético de teste.

## Objetivo

Desenvolver um detector de três classes de talheres (`garfo`, `faca` e `colher`) utilizando exclusivamente dados sintéticos. A proposta demonstra como uma cena 3D simples, combinada com automação em Python, pode reduzir o trabalho manual de captura e anotação de imagens.

## Tecnologias utilizadas

- Blender 5.2 e API Python `bpy`;
- Python 3;
- Ultralytics YOLOv8n;
- PyTorch;
- Google Colab com GPU;
- Formato de anotação YOLO.

## Estrutura do projeto

```text
Projeto Final/
├── README.md
├── relatorio_tecnico.pdf
├── requirements.txt
├── blender/
│   ├── Projeto_Final.blend
│   ├── gerar_dataset_talheres.py
│   └── modelos/
│       ├── garfo_simples.fbx
│       ├── faca_simples.fbx
│       └── colher_simples.fbx
├── dataset/
│   ├── dataset.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── treinamento/
│   ├── treinamento_yolo.ipynb
│   └── best.pt
└── resultados/
    ├── results.csv
    ├── results.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── BoxPR_curve.png
    ├── val_batch0_pred.jpg
    └── predicoes/
```

## Dataset sintético

O dataset possui **100 imagens de 640 x 640 pixels**, cada uma contendo os três talheres. As caixas delimitadoras foram projetadas automaticamente a partir das caixas 3D dos objetos e gravadas no formato YOLO.

| Divisão | Imagens | Rótulos | Objetos anotados |
|---|---:|---:|---:|
| Treinamento | 70 | 70 | 210 |
| Validação | 20 | 20 | 60 |
| Teste | 10 | 10 | 30 |
| **Total** | **100** | **100** | **300** |

Durante a geração foram variadas a posição, a rotação e a escala dos objetos, além da intensidade das luzes e da cor do fundo. A câmera ortográfica permaneceu em vista superior para manter todos os objetos dentro do enquadramento.

## Geração do dataset no Blender

1. Abra `blender/Projeto_Final.blend` no Blender.
2. Acesse a área de trabalho **Scripting**.
3. Abra `blender/gerar_dataset_talheres.py`.
4. Confirme que os objetos estão nomeados como `garfo`, `faca` e `colher`.
5. Execute o script com `Alt + P`.

O script cria automaticamente as pastas `train`, `valid` e `test`, renderiza as imagens e produz um arquivo `.txt` para cada imagem.

## Treinamento no Google Colab

Abra `treinamento/treinamento_yolo.ipynb` no Google Colab, habilite uma GPU e execute as células em sequência. Também é possível executar o treinamento diretamente:

```python
from ultralytics import YOLO

modelo = YOLO("yolov8n.pt")
modelo.train(
    data="dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    pretrained=True
)
```

## Resultados

Após 50 épocas, o desempenho no conjunto sintético de validação foi:

| Métrica | Resultado |
|---|---:|
| Precisão | 0,9902 |
| Revocação | 1,0000 |
| mAP@50 | 0,9950 |
| mAP@50-95 | 0,8892 |

Os resultados indicam que o modelo aprendeu adequadamente as características visuais das três classes no domínio sintético. O arquivo `treinamento/best.pt` contém os melhores pesos obtidos.

## Executar uma predição

```python
from ultralytics import YOLO

modelo = YOLO("treinamento/best.pt")
modelo.predict(
    source="dataset/test/images",
    conf=0.25,
    imgsz=640,
    save=True
)
```

## Limitações e domain gap

As métricas elevadas foram obtidas em imagens sintéticas produzidas pelo mesmo pipeline utilizado no treinamento. Portanto, não comprovam desempenho equivalente em fotografias reais. Os modelos 3D são simples, a câmera apresenta predominantemente uma vista superior e o conjunto possui somente 100 imagens. Como continuidade, recomenda-se ampliar materiais, texturas, fundos, oclusões e perspectivas, além de avaliar o modelo em imagens reais e aplicar ajuste fino com dados reais.

## Principais artefatos

- `blender/Projeto_Final.blend`: cena configurada;
- `blender/gerar_dataset_talheres.py`: geração e anotação automática;
- `dataset/`: imagens e rótulos utilizados;
- `treinamento/treinamento_yolo.ipynb`: treinamento e avaliação;
- `treinamento/best.pt`: modelo treinado;
- `resultados/`: métricas, gráficos e exemplos de detecção;
- `relatorio_tecnico.pdf`: documentação técnica completa.

## Autor

**Adriano César Santana**  
Fastcamp de Dados Sintéticos para IA e Visão Computacional - 2026
