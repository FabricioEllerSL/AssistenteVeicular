# Importando os módulos necessários
import unittest
from assistente_veicular import *

# Caminhos para os arquivos de áudio de teste com os comandos
LIGAR_VEICULO = "audios_comandos/ligar-veiculo.wav"
DESLIGAR_VEICULO = "audios_comandos/desligar-veiculo.wav"
ABRIR_PORTA_MALAS = "audios_comandos/abrir-malas.wav"
LEVANTAR_SUSPENSAO = "audios_comandos/levantar-susp.wav"
ABAIXAR_SUSPENSAO = "audios_comandos/abaixar-susp.wav"

# Classe de testes
class ASSISTENTE_VEICULAR(unittest.TestCase):

    # Método de configuração executado antes de todos os testes
    @classmethod
    def setUpClass(cls):
        # Verifica se há suporte para CUDA e define o dispositivo apropriado (GPU ou CPU)
        cls.dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Inicializa o modelo, processador e outros componentes necessários
        cls.iniciado, cls.processador, cls.modelo, cls.gravador, cls.palavras_irrelevantes, cls.acoes = iniciar(cls.dispositivo)

    # Teste do comando para ligar o veículo
    def test_ligar_veiculo(self):
        # Transcreve o áudio para texto (IDEM PARA OS DEMAIS TESTES)
        transcricao = transcrever(self.dispositivo, carregar_audio(LIGAR_VEICULO), self.modelo, self.processador)
        
        # Verifica se a transcrição não é nula (IDEM PARA OS DEMAIS TESTES)
        self.assertIsNotNone(transcricao)

        # Remove as palavras irrelevantes (stop words) da transcrição (IDEM PARA OS DEMAIS TESTES)
        comando = remover_stopwords(transcricao, self.palavras_irrelevantes)
        
        # Valida o comando (ação e objeto) (IDEM PARA OS DEMAIS TESTES)
        valido, acao, objeto = validar_comando(comando, self.acoes)

        # Verifica se o comando é válido e se a ação e o objeto estão corretos (IDEM PARA OS DEMAIS TESTES)
        self.assertTrue(valido)
        self.assertEqual(acao, "ligar")
        self.assertEqual(objeto, "veículo")

        print(f"Comando validado! Ação: {acao}, Objeto: {objeto}")

    # Teste do comando para desligar o veículo
    def test_desligar_veiculo(self):
        transcricao = transcrever(self.dispositivo, carregar_audio(DESLIGAR_VEICULO), self.modelo, self.processador)
        self.assertIsNotNone(transcricao)
        comando = remover_stopwords(transcricao, self.palavras_irrelevantes)
        valido, acao, objeto = validar_comando(comando, self.acoes)
        self.assertTrue(valido)
        self.assertEqual(acao, "desligar")
        self.assertEqual(objeto, "veículo")
        print(f"Comando validado! Ação: {acao}, Objeto: {objeto}")


    # Teste do comando para abrir a porta-malas do veículo
    def test_abrir_porta_malas(self):
        transcricao = transcrever(self.dispositivo, carregar_audio(ABRIR_PORTA_MALAS), self.modelo, self.processador)
        self.assertIsNotNone(transcricao)
        comando = remover_stopwords(transcricao, self.palavras_irrelevantes)
        valido, acao, objeto = validar_comando(comando, self.acoes)
        self.assertTrue(valido)
        self.assertEqual(acao, "abrir")
        self.assertEqual(objeto, "porta malas")
        print(f"Comando validado! Ação: {acao}, Objeto: {objeto}")


    # Teste do comando para levantar a suspensão do veículo
    def test_levantar_suspensao(self):
        transcricao = transcrever(self.dispositivo, carregar_audio(LEVANTAR_SUSPENSAO), self.modelo, self.processador)
        self.assertIsNotNone(transcricao)
        comando = remover_stopwords(transcricao, self.palavras_irrelevantes)
        valido, acao, objeto = validar_comando(comando, self.acoes)
        self.assertTrue(valido)
        self.assertEqual(acao, "levantar")
        self.assertEqual(objeto, "suspensão")
        print(f"Comando validado! Ação: {acao}, Objeto: {objeto}")


    # Teste do comando para abaixar a suspensão do veículo
    def test_abaixar_suspensao(self):
        transcricao = transcrever(self.dispositivo, carregar_audio(ABAIXAR_SUSPENSAO), self.modelo, self.processador)
        self.assertIsNotNone(transcricao)
        comando = remover_stopwords(transcricao, self.palavras_irrelevantes)
        valido, acao, objeto = validar_comando(comando, self.acoes)
        self.assertTrue(valido)
        self.assertEqual(acao, "abaixar")
        self.assertEqual(objeto, "suspensão")
        print(f"Comando validado! Ação: {acao}, Objeto: {objeto}")



# Verifica se o script está sendo executado diretamente e executa os testes
if __name__ == "__main__":
    unittest.main()
