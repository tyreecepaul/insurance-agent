import React, { useState, useEffect } from 'react';
import api from './api/client';
import ChatInterface from './components/ChatInterface';
import ClaimDraftPanel from './components/ClaimDraftPanel';
import './styles/App.css';

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [claimDraft, setClaimDraft] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  // Initialize session on component mount
  useEffect(() => {
    initializeSession();
  }, []);

  const initializeSession = async () => {
    setLoading(true);
    try {
      const response = await api.createSession();
      setSessionId(response.session_id);
      setStatusMessage('Session started');
      setTimeout(() => setStatusMessage(''), 3000);
    } catch (error) {
      setStatusMessage(`Failed to start session: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetSession = async () => {
    if (!sessionId) return;

    const confirmed = window.confirm('Clear session and start fresh?');
    if (!confirmed) return;

    setLoading(true);
    try {
      await api.resetSession(sessionId);
      setClaimDraft(null);
      setStatusMessage('Session reset');
      // Initialize new session after reset
      await new Promise(resolve => setTimeout(resolve, 1000));
      await initializeSession();
    } catch (error) {
      setStatusMessage(`Failed to reset: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) {
    return (
      <div className="app-container">
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          textAlign: 'center'
        }}>
          <div>
            <h1>🚀 Insurance Claims Agent</h1>
            <p>Initializing session...</p>
            {statusMessage && <p style={{ color: '#fee' }}>{statusMessage}</p>}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <h1>Insurance Claims Agent</h1>
          <span className="session-badge">
            Session: {sessionId.substring(0, 8)}...
          </span>
        </div>
        <button
          onClick={resetSession}
          disabled={loading}
          className="reset-button"
        >
          {loading ? 'Resetting...' : 'Reset'}
        </button>
      </header>

      {/* Status Message */}
      {statusMessage && (
        <div className="status-banner">
          {statusMessage}
        </div>
      )}

      {/* Main Content */}
      <div className="content-container">
        <div className="chat-column">
          <ChatInterface
            sessionId={sessionId}
            onClaimDraftUpdate={setClaimDraft}
          />
        </div>
        <div className="panel-column">
          <ClaimDraftPanel
            sessionId={sessionId}
            claimDraft={claimDraft}
            onUpdate={setClaimDraft}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Tip: Upload claim images to help the agent understand your situation better.
          Ask questions about your policy, file claims, or check status.
        </p>
      </footer>
    </div>
  );
}
