'use client';

import { useState } from 'react';
import "./app.css";

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Adicionar pergunta e resposta dummy por enquanto
    setMessages([...messages, { question: input, answer: 'Resposta do chatbot aqui.' }]);
    setInput('');
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
        />
        <button className="button" onClick={handleSubmit}>Enviar</button>
      </div>
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className="message">
            <p className="question">{msg.question}</p>
            <p className="answer">{msg.answer}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
