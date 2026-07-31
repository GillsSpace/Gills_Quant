import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  RefreshCw, 
  ExternalLink, 
  Key, 
  Clock, 
  CheckCircle2, 
  Server,
  Copy,
  Check,
  Wifi
} from 'lucide-react';

export default function App() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [authUrl, setAuthUrl] = useState(null);
  const [redirectUrlInput, setRedirectUrlInput] = useState('');
  const [logs, setLogs] = useState([]);
  const [copied, setCopied] = useState(false);

  const SERVER_ENDPOINT = "https://gill-01.taileb5b7d.ts.net/update-token";

  const addLog = (text, type = 'info') => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [{ time, text, type }, ...prev]);
  };

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/status');
      const data = await res.json();
      setStatus(data);
      if (data.refresh_token_valid) {
        addLog('Refresh token active & valid', 'success');
      } else {
        addLog('Refresh token expired. OAuth re-authentication required.', 'error');
      }
    } catch (err) {
      addLog(`Failed to connect to gill-01: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchAuthUrl = async () => {
    try {
      const res = await fetch('/api/auth-url');
      const data = await res.json();
      setAuthUrl(data.auth_url);
    } catch (err) {
      addLog(`Failed to retrieve Schwab Auth URL: ${err.message}`, 'error');
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchAuthUrl();
  }, []);

  const handleCompleteAuth = async () => {
    if (!redirectUrlInput.trim()) return;
    setUpdating(true);
    addLog('Completing Schwab OAuth authentication & token sync...', 'info');
    try {
      const res = await fetch('/api/update-tokens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ redirect_url: redirectUrlInput.trim() })
      });
      const data = await res.json();

      if (data.upload_result && data.upload_result.success) {
        addLog('OAuth successful! Tokens synced to gill-01 server.', 'success');
      } else {
        addLog(`Tokens updated. Sync status: ${JSON.stringify(data.upload_result)}`, 'info');
      }

      if (data.token_status) {
        setStatus(data.token_status);
      }
      setRedirectUrlInput('');
    } catch (err) {
      addLog(`OAuth sync failed: ${err.message}`, 'error');
    } finally {
      setUpdating(false);
    }
  };

  const formatRefreshTime = (sec) => {
    if (!sec || sec <= 0) return 'Expired';
    const d = Math.floor(sec / (3600 * 24));
    const h = Math.floor((sec % (3600 * 24)) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    if (d > 0) return `${d}d ${h}h remaining`;
    if (h > 0) return `${h}h ${m}m remaining`;
    return `${m}m remaining`;
  };

  const getProgressPercent = (sec) => {
    if (!sec || sec <= 0) return 0;
    const maxSec = 7 * 86400; // 7 days max
    return Math.min(100, Math.max(0, (sec / maxSec) * 100));
  };

  const copyEndpoint = () => {
    navigator.clipboard.writeText(SERVER_ENDPOINT);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isTokenValid = status?.refresh_token_valid;
  const isServerOnline = status !== null;

  return (
    <div>
      {/* Minimalist Header */}
      <header className="app-header">
        <div className="brand-group">
          <div className="gq-logo-badge">GQ</div>
          <div className="brand-text">
            <h1>GQ Schwab Sync</h1>
            <p>Tailscale Server Token Manager</p>
          </div>
        </div>

        {/* Server Connection Indicator */}
        <div className="status-pill">
          <div className={`status-dot ${isServerOnline ? (isTokenValid ? 'active' : 'warning') : 'danger'}`}></div>
          <span>{isServerOnline ? (isTokenValid ? 'Connected' : 'Re-Auth Required') : 'Offline'}</span>
        </div>
      </header>

      {/* Refresh Token Status Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <ShieldCheck size={18} color="var(--accent-emerald)" />
            Refresh Token Status
          </div>
          <button 
            onClick={fetchStatus} 
            disabled={loading}
            style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
            title="Refresh Status"
          >
            <RefreshCw size={15} className={loading ? 'spinner-white' : ''} />
          </button>
        </div>

        <div className="metric-box">
          <div className="metric-label">7-Day Schwab Refresh Token</div>
          <div className={`metric-value ${isTokenValid ? '' : 'danger'}`}>
            {isTokenValid ? formatRefreshTime(status?.refresh_token_expires_in) : 'Expired'}
          </div>

          {/* Progress bar */}
          <div className="progress-bar-bg">
            <div 
              className={`progress-bar-fill ${isTokenValid ? (status?.refresh_token_expires_in > 86400 ? 'active' : 'warning') : 'danger'}`}
              style={{ width: `${getProgressPercent(status?.refresh_token_expires_in)}%` }}
            ></div>
          </div>
        </div>

        {status?.refresh_token_issued && (
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            Issued: {new Date(status.refresh_token_issued).toLocaleString()}
          </div>
        )}
      </div>

      {/* Re-Authentication Section */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Key size={18} color="var(--accent-blue)" />
            Schwab OAuth Re-Authentication
          </div>
        </div>

        <div className="card-subtitle">
          Schwab requires full OAuth renewal every 7 days. Tap below to log in on Schwab's portal, grant permission, then paste the redirected URL to sync tokens.
        </div>

        {authUrl && (
          <a 
            href={authUrl} 
            target="_blank" 
            rel="noreferrer"
            className="btn btn-secondary"
            style={{ marginBottom: '14px' }}
          >
            <ExternalLink size={16} /> Open Schwab Login Portal
          </a>
        )}

        <div className="metric-label" style={{ marginBottom: '6px' }}>Redirected Callback URL</div>
        <input 
          type="text"
          className="input-field"
          placeholder="https://127.0.0.1/?code=..."
          value={redirectUrlInput}
          onChange={(e) => setRedirectUrlInput(e.target.value)}
        />

        <button 
          className="btn btn-primary"
          onClick={handleCompleteAuth}
          disabled={updating || !redirectUrlInput.trim()}
        >
          {updating ? <div className="spinner"></div> : <CheckCircle2 size={16} />}
          {updating ? 'Authenticating & Syncing...' : 'Complete Re-Auth & Sync'}
        </button>
      </div>

      {/* Target Server Endpoint Card */}
      <div className="card">
        <div className="card-header" style={{ marginBottom: '10px' }}>
          <div className="card-title">
            <Server size={16} color="var(--text-secondary)" />
            Server Endpoint
          </div>
        </div>
        
        <div className="endpoint-box">
          <span>{SERVER_ENDPOINT}</span>
          <button 
            onClick={copyEndpoint}
            style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
            title="Copy Endpoint"
          >
            {copied ? <Check size={14} color="var(--accent-emerald)" /> : <Copy size={14} />}
          </button>
        </div>
      </div>

      {/* Activity Log */}
      <div className="card">
        <div className="card-header" style={{ marginBottom: '10px' }}>
          <div className="card-title" style={{ fontSize: '13px' }}>
            <Clock size={15} color="var(--text-secondary)" />
            Activity Log
          </div>
        </div>

        <div className="log-box">
          {logs.length === 0 ? (
            <div style={{ fontStyle: 'italic', opacity: 0.5 }}>System ready. Status checked.</div>
          ) : (
            logs.map((item, idx) => (
              <div key={idx} className={`log-item ${item.type}`}>
                <span style={{ color: 'var(--text-muted)' }}>[{item.time}]</span> {item.text}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
