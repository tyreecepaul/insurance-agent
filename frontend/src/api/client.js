/**
 * api/client.js
 * Wrapper around axios for API calls to the insurance-agent backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 second timeout for chat (LLM inference)
  headers: {
    'Content-Type': 'application/json'
  }
});

// Session Management
export const api = {
  // Health check
  health: () => client.get('/health'),

  // Sessions
  createSession: () => client.post('/api/session/create'),
  getSessionCount: () => client.get('/api/sessions/count'),
  resetSession: (sessionId) => client.delete('/api/session/reset', { params: { session_id: sessionId } }),
  cleanupSessions: () => client.post('/api/sessions/cleanup'),

  // Chat
  chat: (sessionId, userMessage, imagePath = null) =>
    client.post('/api/chat', {
      session_id: sessionId,
      user_message: userMessage,
      image_path: imagePath
    }),

  // Conversation
  getConversation: (sessionId) =>
    client.get('/api/conversation', { params: { session_id: sessionId } }),

  // Claim Draft
  getClaimDraft: (sessionId) =>
    client.get('/api/claim-draft', { params: { session_id: sessionId } }),

  updateClaimDraft: (sessionId, claimDraft) =>
    client.put('/api/claim-draft', {
      session_id: sessionId,
      claim_draft: claimDraft
    }),

  // Image Upload
  uploadImage: async (sessionId, file) => {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('file', file);

    return client.post('/api/upload-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  }
};

// Error handling
client.interceptors.response.use(
  response => response.data, // Return just the data
  error => {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      console.error(`API Error [${status}]:`, data);

      if (status === 404) {
        throw new Error(`Not found: ${data.error || 'Resource not found'}`);
      } else if (status >= 500) {
        throw new Error(`Server error: ${data.error || 'Internal server error'}`);
      } else {
        throw new Error(data.error || 'An error occurred');
      }
    } else if (error.request) {
      console.error('Network error:', error.request);
      throw new Error('Network error: Could not reach server');
    } else {
      console.error('Error:', error.message);
      throw error;
    }
  }
);

export default api;
