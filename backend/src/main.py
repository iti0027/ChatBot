from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

#Instância da aplicação
app = FastAPI(
    title="ChatBot API",
    description="API do chatbot com embeddings e busca vetorial",
    version="0.1.0"
)

# Configurar CORS para aceitar requisições front-end
app.dd_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#rotas de verificação básicas
@app.get("/")
def read_root():
    return{
        "message": "API esta online",
        "version": "0.1.0"
    }

@app.get("/health")
def health_check():
    return{
        "status": "ok",
        "message": "API saudável"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
