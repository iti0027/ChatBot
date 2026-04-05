'use client';

import { useState } from 'react';
import "./app.css";

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input;
    setInput('');
    setLoading(true);

    // Adicionar mensagem do usuário
    setMessages(prev => [...prev, { question: userMessage, answer: null, loading: true }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          session_id: 'frontend-session' // ou gerar um ID único
        }),
      });

      if (!response.ok) {
        throw new Error(`Erro: ${response.status}`);
      }

      const data = await response.json();

      // Atualizar a última mensagem com a resposta
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIndex = newMessages.length - 1;
        newMessages[lastIndex] = {
          question: userMessage,
          answer: data.response,
          loading: false,
          retrieved_docs: data.retrieved_docs,
          model_used: data.model_used
        };
        return newMessages;
      });
    } catch (error) {
      console.error('Erro ao enviar mensagem:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIndex = newMessages.length - 1;
        newMessages[lastIndex] = {
          question: userMessage,
          answer: 'Erro ao conectar com o servidor. Verifique se o backend está rodando.',
          loading: false
        };
        return newMessages;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">ChatBot Inteligente</h1>
      <div className="input-section">
        <label className="label">Digite aqui sua pergunta:</label>
        <input
          className="input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Digite sua pergunta..."
          disabled={loading}
        />
        <button className="button" onClick={handleSubmit} disabled={loading || !input.trim()}>
          {loading ? 'Enviando...' : 'Enviar'}
        </button>
      </div>
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className="message">
            <p className="question"><strong>Você:</strong> {msg.question}</p>
            {msg.loading ? (
              <p className="answer"><em>Carregando resposta...</em></p>
            ) : (
              <p className="answer"><strong>ChatBot:</strong> {msg.answer}</p>
            )}
            {msg.retrieved_docs && (
              <p className="info">Documentos recuperados: {msg.retrieved_docs} | Modelo: {msg.model_used}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
