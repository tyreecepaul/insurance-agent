import React, { useState } from 'react';
import api from '../api/client';

export default function ClaimDraftPanel({ sessionId, claimDraft, onUpdate }) {
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState(claimDraft || {});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Update form when claimDraft prop changes
  React.useEffect(() => {
    setFormData(claimDraft || {});
  }, [claimDraft]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSave = async () => {
    setLoading(true);
    setError(null);

    try {
      const updated = await api.updateClaimDraft(sessionId, formData);
      onUpdate(updated);
      setEditMode(false);
    } catch (err) {
      setError(`Failed to save: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setFormData(claimDraft || {});
    setEditMode(false);
    setError(null);
  };

  const fields = [
    { key: 'policy_number', label: 'Policy Number', type: 'text' },
    { key: 'insurance_type', label: 'Insurance Type', type: 'text' },
    { key: 'claimant_name', label: 'Claimant Name', type: 'text' },
    { key: 'incident_type', label: 'Incident Type', type: 'text' },
    { key: 'incident_date', label: 'Incident Date', type: 'date' },
    { key: 'description', label: 'Description', type: 'textarea' }
  ];

  return (
    <div className="claim-draft-panel">
      <div className="panel-header">
        <h2>Claim Draft</h2>
        {!editMode && (
          <button onClick={() => setEditMode(true)} className="edit-button">
            Edit
          </button>
        )}
      </div>

      {error && (
        <div className="error-message">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="draft-content">
        {Object.keys(formData).length === 0 ? (
          <p className="empty-draft">No claim data yet. Ask questions to build your claim.</p>
        ) : editMode ? (
          <form className="draft-form">
            {fields.map(field => (
              <div key={field.key} className="form-group">
                <label htmlFor={field.key}>{field.label}</label>
                {field.type === 'textarea' ? (
                  <textarea
                    id={field.key}
                    name={field.key}
                    value={formData[field.key] || ''}
                    onChange={handleChange}
                    rows="3"
                    className="form-input"
                  />
                ) : (
                  <input
                    id={field.key}
                    type={field.type}
                    name={field.key}
                    value={formData[field.key] || ''}
                    onChange={handleChange}
                    className="form-input"
                  />
                )}
              </div>
            ))}

            <div className="form-buttons">
              <button
                type="button"
                onClick={handleSave}
                disabled={loading}
                className="save-button"
              >
                {loading ? 'Saving...' : 'Save'}
              </button>
              <button
                type="button"
                onClick={handleCancel}
                disabled={loading}
                className="cancel-button"
              >
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <div className="draft-display">
            {fields.map(field => {
              const value = formData[field.key];
              return (
                <div key={field.key} className={`draft-field ${value ? 'filled' : 'empty'}`}>
                  <span className="field-label">{field.label}</span>
                  <span className="field-value">{value || '—'}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
