# Importa as classes necessárias
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Lista de modelos de reconhecimento de fala a serem utilizados
MODELOS = [
    "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2",
    "facebook/wav2vec2-base-960h",
    "Edresson/wav2vec2-large-xlsr-coraa-portuguese"
]

# Função para inicializar um modelo específico
def iniciar_modelo(nome_modelo, dispositivo="cpu"):
    print(f"Iniciando Modelo: {nome_modelo}")
    
    try:
        # Carrega o modelo pré-treinado
        modelo = Wav2Vec2ForCTC.from_pretrained(nome_modelo).to(dispositivo)
        # Carrega o processador
        processador = Wav2Vec2Processor.from_pretrained(nome_modelo)
        
        # Retorna True indicando que o modelo foi carregado com sucesso
        return True, processador, modelo, []
    except Exception as e:
        # Em caso de erro ao carregar o modelo, imprime a mensagem de erro
        print(f"Erro ao iniciar o modelo {nome_modelo}: {str(e)}")
        
        # Retorna False e o erro ocorrido
        return False, None, None, [e]

# Função para inicializar vários modelos
def iniciar_modelos(modelos=MODELOS, dispositivo="cpu"):
    iniciados = True  # Inicializa uma variável para verificar se todos os modelos foram carregados com sucesso
    erros = []  # Lista para armazenar erros de inicialização dos modelos

    # Itera sobre a lista de modelos
    for nome_modelo in modelos:
        # Tenta iniciar cada modelo e obter o processador, modelo e erros
        iniciado, processador, modelo, erros_modelo = iniciar_modelo(nome_modelo, dispositivo)
        
        # Atualiza a variável iniciados para False se algum modelo falhar ao ser iniciado
        iniciados = iniciados and iniciado
        
        # Adiciona os erros do modelo à lista de erros
        erros.extend(erros_modelo)

    # Retorna se todos os modelos foram carregados com sucesso, o último processador e modelo, e a lista de erros
    return iniciados, processador, modelo, erros
