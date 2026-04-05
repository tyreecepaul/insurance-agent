import React, { useState, useEffect, useRef } from 'react';
import api from '../api/client';
import './ChatInterface.css';

export default function ChatInterface({ sessionId, onClaimDraftUpdate }) {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryType, setQueryType] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversation on mount
  useEffect(() => {
    if (sessionId) {
      loadConversation();
    }
  }, [sessionId]);

  const loadConversation = async () => {
    try {
      const data = await api.getConversation(sessionId);
      setMessages(data.messages || []);
      onClaimDraftUpdate(data.claim_draft);
      setError(null);
    } catch (err) {
      setError(`Failed to load conversation: ${err.message}`);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMsg = inputValue.trim();
    setInputValue('');
    setError(null);
    setLoading(true);

    try {
      // Upload image if selected
      let imagePath = null;
      if (selectedFile) {
        const uploadResp = await api.uploadImage(sessionId, selectedFile);
        imagePath = uploadResp.file_path;
        setSelectedFile(null);
      }

      // Send chat message
      const response = await api.chat(sessionId, userMsg, imagePath);

      // Add user message
      setMessages(prev => [...prev, {
        role: 'user',
        content: userMsg,
        timestamp: new Date().toISOString()
      }]);

      // Add agent response
      setMessages(prev => [...prev, {
        role: 'agent',
        content: response.agent_response,
        query_type: response.query_type,
        timestamp: new Date().toISOString()
      }]);

      setQueryType(response.query_type);
      onClaimDraftUpdate(response.claim_draft);
    } catch (err) {
      setError(`Failed to send message: ${err.message}`);
      // Re-add the user message for retry
      setInputValue(userMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  return (
    <div className="chat-interface">
      {/* Messages */}
      <div className="messages-list">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>Start a conversation about your insurance claim</p>
            <p className="text-sm text-gray-500">Ask about coverage, file a claim, or check status</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message message-${msg.role}`}>
              <div className="message-header">
                <span className="role-badge">{msg.role === 'agent' ? '🤖 Agent' : '👤 You'}</span>
                {msg.query_type && (
                  <span className="query-type-badge">{msg.query_type}</span>
                )}
                <span className="timestamp">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="message-content">{msg.content}</div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="close-btn">×</button>
        </div>
      )}

      {/* Input Area */}
      <form onSubmit={sendMessage} className="input-form">
        <div className="input-wrapper">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message..."
            disabled={loading}
            className="message-input"
          />
          <button type="submit" disabled={loading || !inputValue.trim()} className="send-button">
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>

        {/* File Upload */}
        <div className="file-upload">
          <label htmlFor="file-input" className="file-label">
            {selectedFile ? selectedFile.name : 'Attach Image'}
          </label>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            disabled={loading}
            className="hidden"
          />
          {selectedFile && (
            <button
              type="button"
              onClick={() => setSelectedFile(null)}
              className="clear-file-btn"
            >
              Remove
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
