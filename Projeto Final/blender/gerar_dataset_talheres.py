"""
Geração de dataset sintético de talheres no Blender.

Classes YOLO:
0 = garfo
1 = faca
2 = colher

Saída: //dataset_talheres/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
  dataset.yaml

Como executar:
1. Abra o arquivo .blend que contém os objetos garfo, faca e colher.
2. Abra a área de trabalho Scripting.
3. Carregue este arquivo e clique em Executar Script.
"""

import bpy
import math
import random
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


# -----------------------------------------------------------------------------
# CONFIGURAÇÕES DO PROJETO
# -----------------------------------------------------------------------------

QUANTIDADE_IMAGENS = 100
RESOLUCAO = 640
SEMENTE_ALEATORIA = 42

CLASSES = {
    "garfo": 0,
    "faca": 1,
    "colher": 2,
}

# Diretório criado ao lado do arquivo .blend.
DIRETORIO_SAIDA = Path(bpy.path.abspath("//dataset_talheres"))

# Divisão das 100 imagens: 70 para treino, 20 para validação e 10 para teste.
DIVISOES = {
    "train": (0, 70),
    "valid": (70, 90),
    "test": (90, 100),
}


# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------

def localizar_objeto(nome):
    """Localiza o objeto mesmo quando o Blender acrescenta .001 ao nome."""
    nome = nome.lower()
    for objeto in bpy.data.objects:
        if objeto.name.lower() == nome or objeto.name.lower().startswith(nome + "."):
            return objeto
    raise RuntimeError(f"Objeto '{nome}' não encontrado na cena.")


def criar_diretorios():
    for divisao in DIVISOES:
        (DIRETORIO_SAIDA / divisao / "images").mkdir(parents=True, exist_ok=True)
        (DIRETORIO_SAIDA / divisao / "labels").mkdir(parents=True, exist_ok=True)


def definir_divisao(indice):
    for divisao, (inicio, fim) in DIVISOES.items():
        if inicio <= indice < fim:
            return divisao
    return "test"


def obter_shader_principled(material):
    """Localiza o Principled BSDF pelo tipo, independentemente do idioma."""
    for no in material.node_tree.nodes:
        if no.type == "BSDF_PRINCIPLED":
            return no

    # Caso o material existente não possua o nó, cria um novo.
    shader = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    saida = next(
        (no for no in material.node_tree.nodes if no.type == "OUTPUT_MATERIAL"),
        None,
    )
    if saida is None:
        saida = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(shader.outputs["BSDF"], saida.inputs["Surface"])
    return shader


def criar_material(nome, cor, metalico=0.0, rugosidade=0.5):
    material = bpy.data.materials.get(nome) or bpy.data.materials.new(nome)
    material.use_nodes = True
    shader = obter_shader_principled(material)
    shader.inputs["Base Color"].default_value = (*cor, 1.0)
    shader.inputs["Metallic"].default_value = metalico
    shader.inputs["Roughness"].default_value = rugosidade
    return material


def configurar_renderizacao():
    cena = bpy.context.scene
    cena.render.resolution_x = RESOLUCAO
    cena.render.resolution_y = RESOLUCAO
    cena.render.resolution_percentage = 100
    cena.render.image_settings.file_format = "PNG"
    cena.render.film_transparent = False

    # Compatibilidade entre versões recentes do Blender.
    try:
        cena.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        cena.render.engine = "BLENDER_EEVEE"

    cena.render.image_settings.color_mode = "RGB"
    cena.render.image_settings.color_depth = "8"
    # O nome das opções de contraste muda entre versões do Blender.
    for opcao in ("AgX - Medium High Contrast", "Medium High Contrast", "None"):
        try:
            cena.view_settings.look = opcao
            break
        except TypeError:
            continue


def configurar_fundo():
    """Cria ou reutiliza um plano horizontal como fundo da cena."""
    fundo = bpy.data.objects.get("Fundo_Dataset")
    if fundo is None:
        bpy.ops.mesh.primitive_plane_add(size=18, location=(0, 0, 0))
        fundo = bpy.context.object
        fundo.name = "Fundo_Dataset"

    fundo.location = (0, 0, 0)
    fundo.scale = (1, 1, 1)
    fundo.rotation_euler = (0, 0, 0)

    material = criar_material("Material_Fundo", (0.55, 0.55, 0.55), 0.0, 0.75)
    fundo.data.materials.clear()
    fundo.data.materials.append(material)
    return fundo, material


def configurar_camera():
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 9))
        camera = bpy.context.object
        camera.name = "Camera"

    camera.location = (0, 0, 9)
    camera.rotation_euler = (0, 0, 0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 8.2
    bpy.context.scene.camera = camera
    return camera


def configurar_luzes():
    """Cria uma iluminação principal e uma luz de preenchimento."""
    luz_principal = bpy.data.objects.get("Luz_Principal")
    if luz_principal is None:
        dados = bpy.data.lights.new("Luz_Principal", type="AREA")
        luz_principal = bpy.data.objects.new("Luz_Principal", dados)
        bpy.context.collection.objects.link(luz_principal)
    luz_principal.location = (-3.5, -3.0, 7.0)
    luz_principal.data.shape = "DISK"
    luz_principal.data.size = 5.0

    luz_auxiliar = bpy.data.objects.get("Luz_Auxiliar")
    if luz_auxiliar is None:
        dados = bpy.data.lights.new("Luz_Auxiliar", type="AREA")
        luz_auxiliar = bpy.data.objects.new("Luz_Auxiliar", dados)
        bpy.context.collection.objects.link(luz_auxiliar)
    luz_auxiliar.location = (4.0, 3.0, 5.0)
    luz_auxiliar.data.size = 4.0

    return luz_principal, luz_auxiliar


def dimensao_maxima_xy(objeto):
    return max(objeto.dimensions.x, objeto.dimensions.y)


def normalizar_objetos(objetos):
    """Redimensiona os modelos para tamanhos semelhantes e aplicáveis à cena."""
    alvos = {"garfo": 2.55, "faca": 2.75, "colher": 2.60}
    for nome, objeto in objetos.items():
        objeto.hide_render = False
        objeto.hide_set(False)
        atual = dimensao_maxima_xy(objeto)
        if atual <= 0:
            raise RuntimeError(f"O objeto '{nome}' possui dimensões inválidas.")
        fator = alvos[nome] / atual
        objeto.scale *= fator
        bpy.context.view_layer.objects.active = objeto
        objeto.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        objeto.select_set(False)


def randomizar_cena(objetos, fundo_material, luz_principal, luz_auxiliar):
    """Varia posição, rotação, escala, iluminação e cor do fundo."""
    posicoes_base = [(-2.15, 0.0), (0.0, 0.0), (2.15, 0.0)]
    random.shuffle(posicoes_base)

    for (nome, objeto), (base_x, base_y) in zip(objetos.items(), posicoes_base):
        objeto.location.x = base_x + random.uniform(-0.32, 0.32)
        objeto.location.y = base_y + random.uniform(-1.35, 1.35)
        objeto.location.z = random.uniform(0.10, 0.22)
        objeto.rotation_euler = (
            random.uniform(math.radians(-5), math.radians(5)),
            random.uniform(math.radians(-5), math.radians(5)),
            random.uniform(0, 2 * math.pi),
        )
        escala = random.uniform(0.82, 1.10)
        objeto.scale = (escala, escala, escala)

    cor_fundo = tuple(random.uniform(0.18, 0.82) for _ in range(3))
    shader_fundo = obter_shader_principled(fundo_material)
    shader_fundo.inputs["Base Color"].default_value = (*cor_fundo, 1.0)
    shader_fundo.inputs["Roughness"].default_value = random.uniform(0.60, 0.95)

    luz_principal.data.energy = random.uniform(650, 1250)
    luz_principal.data.color = (
        random.uniform(0.88, 1.0),
        random.uniform(0.88, 1.0),
        random.uniform(0.88, 1.0),
    )
    luz_principal.rotation_euler.z = random.uniform(0, 2 * math.pi)
    luz_auxiliar.data.energy = random.uniform(250, 650)

    mundo = bpy.context.scene.world
    mundo.use_nodes = True
    fundo_mundo = mundo.node_tree.nodes.get("Background")
    fundo_mundo.inputs["Strength"].default_value = random.uniform(0.20, 0.45)


def caixa_delimitadora_yolo(objeto, camera, cena):
    """Projeta os oito cantos da caixa 3D para gerar a caixa 2D normalizada."""
    cantos = [objeto.matrix_world @ Vector(canto) for canto in objeto.bound_box]
    pontos = [world_to_camera_view(cena, camera, canto) for canto in cantos]

    xs = [p.x for p in pontos]
    ys = [p.y for p in pontos]

    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    largura = x_max - x_min
    altura = y_max - y_min
    if largura <= 0 or altura <= 0:
        return None

    centro_x = (x_min + x_max) / 2
    # O sistema YOLO começa no canto superior esquerdo; o Blender usa o inferior.
    centro_y = 1.0 - ((y_min + y_max) / 2)
    return centro_x, centro_y, largura, altura


def escrever_anotacoes(caminho, objetos, camera):
    cena = bpy.context.scene
    linhas = []
    for nome, objeto in objetos.items():
        caixa = caixa_delimitadora_yolo(objeto, camera, cena)
        if caixa is None:
            continue
        x, y, largura, altura = caixa
        linhas.append(
            f"{CLASSES[nome]} {x:.6f} {y:.6f} {largura:.6f} {altura:.6f}"
        )
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def escrever_yaml():
    yaml = (
        f"path: {DIRETORIO_SAIDA.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: garfo\n"
        "  1: faca\n"
        "  2: colher\n"
    )
    (DIRETORIO_SAIDA / "dataset.yaml").write_text(yaml, encoding="utf-8")


# -----------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# -----------------------------------------------------------------------------

def gerar_dataset():
    random.seed(SEMENTE_ALEATORIA)
    criar_diretorios()
    configurar_renderizacao()

    objetos = {nome: localizar_objeto(nome) for nome in CLASSES}
    normalizar_objetos(objetos)

    fundo, fundo_material = configurar_fundo()
    camera = configurar_camera()
    luz_principal, luz_auxiliar = configurar_luzes()

    # Oculta a luz padrão para evitar variações não controladas.
    luz_padrao = bpy.data.objects.get("Light")
    if luz_padrao and luz_padrao not in (luz_principal, luz_auxiliar):
        luz_padrao.hide_render = True

    # A cena da captura possui um cubo usado como parede. O novo plano já faz
    # o papel de fundo, por isso o cubo é ocultado para não cobrir os talheres.
    cubo_padrao = bpy.data.objects.get("Cube")
    if cubo_padrao:
        cubo_padrao.hide_render = True

    for indice in range(QUANTIDADE_IMAGENS):
        divisao = definir_divisao(indice)
        nome_arquivo = f"talheres_{indice + 1:03d}"

        randomizar_cena(
            objetos, fundo_material, luz_principal, luz_auxiliar
        )
        bpy.context.view_layer.update()

        caminho_imagem = DIRETORIO_SAIDA / divisao / "images" / f"{nome_arquivo}.png"
        caminho_rotulo = DIRETORIO_SAIDA / divisao / "labels" / f"{nome_arquivo}.txt"

        bpy.context.scene.render.filepath = str(caminho_imagem)
        bpy.ops.render.render(write_still=True)
        escrever_anotacoes(caminho_rotulo, objetos, camera)

        print(f"[{indice + 1:03d}/{QUANTIDADE_IMAGENS}] {caminho_imagem.name}")

    escrever_yaml()
    print("\nDataset concluído!")
    print(f"Diretório: {DIRETORIO_SAIDA}")


gerar_dataset()
