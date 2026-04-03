import requests
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Configurações para o cliente Ollama
class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 256


class OllamaClient:
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.health_checked = False
        logger.info(f"OllamaClient inicializado: {self.config.base_url}/{self.config.model}")
    
    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"].split(":")[0] for m in models]
                
                if self.config.model in model_names:
                    self.health_checked = True
                    logger.info(f"✓ Ollama saudável. Modelo '{self.config.model}' disponível")
                    return True
                else:
                    logger.warning(f"⚠ Modelo '{self.config.model}' não encontrado em Ollama")
                    logger.warning(f"  Modelos disponíveis: {model_names}")
                    return False
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Não conseguiu conectar ao Ollama em {self.config.base_url}")
            logger.error("  Certifique-se de que Ollama está rodando: ollama serve")
            return False
        except Exception as e:
            logger.error(f"✗ Erro ao verificar saúde do Ollama: {str(e)}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            # Preparar mensagens
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Chamar Ollama
            logger.info(f"Chamando Ollama: {self.config.model}")
            
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.num_predict,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("message", {}).get("content", "")
                
                if generated_text:
                    logger.info(f"Resposta gerada: {len(generated_text)} caracteres")
                    return generated_text.strip()
                else:
                    logger.warning("Ollama retornou resposta vazia")
                    return "Desculpe, não consegui gerar uma resposta."
            else:
                logger.error(f"Ollama retornou status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return "Erro ao chamar Ollama."
                
        except requests.exceptions.Timeout:
            logger.error("Timeout ao chamar Ollama (modelo pode estar processando)")
            return "Timeout na resposta. Tente novamente."
        except requests.exceptions.ConnectionError:
            logger.error(f"Erro de conexão com Ollama em {self.config.base_url}")
            return "Ollama não está disponível. Inicie com: ollama serve"
        except Exception as e:
            logger.error(f"Erro ao gerar com Ollama: {str(e)}")
            return f"Erro: {str(e)}"
    
    def list_available_models(self) -> list:
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
            return []
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {str(e)}")
            return []


# Exports
__all__ = ["OllamaClient", "OllamaConfig"]
