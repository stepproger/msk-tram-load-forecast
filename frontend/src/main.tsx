import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import Login from './Login';
import { getAuth, onAuthChange } from './api';
import 'leaflet/dist/leaflet.css';
import './styles.css';
import './transport.css';

document.documentElement.dataset.theme = 'dark';

function Root() {
  const [authed, setAuthed] = useState(() => getAuth() != null);
  const [notice, setNotice] = useState('');
  useEffect(() => onAuthChange(() => {
    const next = getAuth() != null;
    setNotice(next ? '' : 'Сессия завершена. Войдите снова.');
    setAuthed(next);
  }), []);
  return authed ? <App /> : <Login notice={notice} />;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><Root /></React.StrictMode>);
