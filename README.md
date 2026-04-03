# ChatBot
Criação de um ChatBot, contendo o código para a criação do LLM em Python e a interface em TypeScirpt.

Desenvoltura do BackEnd:
1- instalar as dependências do requirements.txt

rodar a aplicação:
pip install -r requirements.txt

## Configuração do banco de dados
O backend suporta `DATABASE_URL` via variável de ambiente.
Por padrão usa SQLite em `sqlite:///./chatbot.db`.

Para usar um banco público acessível de qualquer computador com o código, defina uma URL pública em um arquivo `.env` ou como variável de ambiente.

Exemplos de URLs públicas:

- PostgreSQL:
  `DATABASE_URL=postgresql://user:password@host:5432/dbname`
- MySQL:
  `DATABASE_URL=mysql+pymysql://user:password@host:3306/dbname`

Crie um arquivo `backend/.env` com o valor correto antes de rodar a aplicação.