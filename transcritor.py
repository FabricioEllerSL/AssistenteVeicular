# Importando os módulos necessários
from inicializador import * 
import torchaudio
import torch 

# Lista de arquivos de áudio para processar
AUDIOS = []

# Taxa de amostragem padrão esperada pelo modelo
TAXA_AMOSTRAGEM = 16000

# Função para carregar e processar um arquivo de áudio
def carregar_audio(localizacao):
    # Carrega o arquivo de áudio
    audio, taxa_amostragem = torchaudio.load(localizacao)
    
    # Se o áudio tiver mais de um canal, converte para mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # Adapta a taxa de amostragem do áudio, se necessário
    if taxa_amostragem != TAXA_AMOSTRAGEM:
        adaptador_amostragem = torchaudio.transforms.Resample(taxa_amostragem, TAXA_AMOSTRAGEM)
        audio = adaptador_amostragem(audio)
        
    # Remove dimensões extras e retorna o tensor de áudio processado
    return audio.squeeze()

# Função para transcrever um áudio usando o modelo e o processador
def transcrever(dispositivo, audio, modelo, processador):
    # Converte o áudio em valores de entrada para o modelo
    valores_entrada = processador(audio, return_tensors="pt", sampling_rate=TAXA_AMOSTRAGEM).input_values.to(dispositivo)
    
    # Passa os valores de entrada pelo modelo para obter os logits (saída do modelo antes da decisão final)
    logits = modelo(valores_entrada).logits

    # Obtém as previsões (índices das palavras mais prováveis)
    predicao = torch.argmax(logits, dim=-1)
    
    # Decodifica a predição em texto e retorna em minúsculas
    transcricao = processador.batch_decode(predicao)[0]

    return transcricao.lower()

# Ponto de entrada do script
if __name__ == "__main__":
    # Define o dispositivo de computação: GPU se disponível, caso contrário, CPU
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Inicializa os modelos e processadores
    iniciado, processador, modelo, erros = iniciar_modelos(MODELOS, dispositivo)

    # Verifica se os modelos foram inicializados com sucesso
    if iniciado:
        # Itera sobre os arquivos de áudio na lista
        for audio in AUDIOS:
            # Carrega e processa o áudio
            fala = carregar_audio(audio)
            
            # Transcreve o áudio
            transcricao = transcrever(dispositivo, fala, modelo, processador)
            
            # Imprime a transcrição resultante
            print(f"Transcrição resultante: {transcricao}")
    else:
        # Imprime os erros caso a inicialização dos modelos falhe
        print("Falha na inicialização dos modelos. Erros:", erros)
