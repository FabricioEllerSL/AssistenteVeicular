# Importando os módulos necessários
from inicializador import * 
from transcritor import *  
from nltk import word_tokenize, corpus
import secrets
import pyaudio 
import wave 
import json 
import os 

# Configurações do áudio e do sistema
TAXAS = 1024  # Tamanho do buffer de áudio
FORMATO = pyaudio.paInt16  # Formato de áudio
CANAIS = 1  # Número de canais (mono)
TEMPO_GRAVACAO = 4  # Duração da gravação em segundos
CAMINHO_FALA = "temp"  # Pasta para salvar arquivos de áudio
IDIOMA_CORPUS = "portuguese"  # Idioma para remover palavras irrelevantes
CAMINHO_CONFIG = "interacoes.json"  # Caminho do arquivo de configuração

# Função para capturar áudio do microfone
def capturar_fala(gravador):
    # Abre o stream para gravar áudio
    gravacao = gravador.open(format=FORMATO, channels=CANAIS, rate=TAXA_AMOSTRAGEM, input=True, frames_per_buffer=TAXAS)
    print("O que desejas fazer com seu veículo?")
    
    # Lista para armazenar os frames de áudio
    fala = []
    # Grava áudio por TEMPO_GRAVACAO segundos
    for _ in range(0, int(TAXA_AMOSTRAGEM / TAXAS * TEMPO_GRAVACAO)):
        fala.append(gravacao.read(TAXAS))
    
    # Finaliza o stream de gravação
    gravacao.stop_stream()
    gravacao.close()

    print("Humm... Entendi")
    return fala  # Retorna os frames de áudio capturados

# Função para salvar o áudio capturado em um arquivo WAV
def gravar_fala(fala):
    gravado, arquivo = False, f"{CAMINHO_FALA}/{secrets.token_hex(32).lower()}.wav"
    try:
        # Cria um arquivo WAV para salvar os dados de áudio
        wav = wave.open(arquivo, 'wb')
        wav.setframerate(TAXA_AMOSTRAGEM)
        wav.setnchannels(CANAIS)
        wav.setsampwidth(gravador.get_sample_size(FORMATO))
        wav.writeframes(b''.join(fala))
        wav.close()
        gravado = True  # Marca o arquivo como gravado com sucesso
    except Exception as e:
        print(f"Erro na gravação do arquivo: {str(e)}")
    return gravado, arquivo  # Retorna o status e o caminho do arquivo

# Remove palavras irrelevantes (stopwords) da transcrição
def remover_stopwords(transcricao, stopwords):
    comando = []
    # Tokeniza a transcrição em palavras
    tokens = word_tokenize(transcricao)
    # Adiciona apenas palavras que não estão na lista de stopwords
    for token in tokens:
        if token not in stopwords:
            comando.append(token)
    return comando  # Retorna a lista de palavras relevantes

# Valida um comando de acordo com as ações esperadas
def validar_comando(comando, acoes):
    valido, acao, objeto = False, None, None
    # Verifica se o comando tem o tamanho esperado
    if len(comando) in [2, 3]:
        acao = comando[0]
        # Extrai o objeto dependendo do tamanho do comando
        if len(comando) == 2:
            objeto = comando[1]
        else:
            objeto = f"{comando[1]} {comando[2]}"
        # Verifica se a ação e o objeto estão na lista de ações permitidas
        for acao_esperada in acoes:
            if acao == acao_esperada["nome"]:
                if objeto in acao_esperada["objetos"]:
                    valido = True
                    break
    return valido, acao, objeto  # Retorna se é válido, a ação e o objeto

# Inicializa os componentes do sistema
def iniciar(dispositivo):
    gravador = pyaudio.PyAudio()  # Inicializa o gravador de áudio
    # Inicializa o modelo e o processador de áudio
    iniciado, processador, modelo, _ = iniciar_modelo(MODELOS[0], dispositivo)
    stopwords, acoes = None, None

    if iniciado:
        # Carrega as palavras irrelevantes do idioma especificado
        stopwords = corpus.stopwords.words(IDIOMA_CORPUS)
        # Carrega as ações disponíveis do arquivo de configuração
        with open(CAMINHO_CONFIG, "r", encoding="utf-8") as arquivo_config:
            config = json.load(arquivo_config)
            acoes = config["acoes"]
    return iniciado, processador, modelo, gravador, stopwords, acoes

# Inicia a captura e o processamento de comandos de voz
def iniciar_captura():
    while True:
        fala = capturar_fala(gravador)  # Captura a fala do usuário
        gravado, arquivo = gravar_fala(fala)  # Salva a fala em um arquivo WAV
        if gravado:
            # Transcreve o áudio gravado
            transcricao = transcrever(dispositivo, carregar_audio(arquivo), modelo, processador)
            os.remove(arquivo)  # Remove o arquivo após a transcrição
            print(f"Você disse: {transcricao}\n")
            # Processa o comando transcrito
            comando = remover_stopwords(transcricao, stopwords)
            valido, acao, objeto = validar_comando(comando, acoes)
            if valido:
                print(
                    "Ligando veículo..." if acao == "ligar" and objeto == "veículo" else
                    "Desligando veículo..." if acao == "desligar" and objeto == "veículo" else
                    "Abrindo porta malas do veículo..." if acao == "abrir" and objeto == "porta malas" else
                    "Levantando suspensão do veículo..." if acao == "levantar" and objeto == "suspensão" else
                    "Abaixando suspensão do veículo..." if acao == "abaixar" and objeto == "suspensão" else
                    ""
                )  # Executa a ação correspondente
    

# Ponto de entrada do programa
if __name__ == "__main__":
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"  # Define o dispositivo de processamento
    # Inicializa o sistema
    iniciado, processador, modelo, gravador, stopwords, acoes = iniciar(dispositivo)

    if iniciado:
        iniciar_captura()  # Inicia o loop de captura e processamento de comandos
