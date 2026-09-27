import { useState } from 'react';
import type { FormEvent } from 'react';
import { login } from './api';

export default function Login({ notice }: { notice?: string }) {
  const [user, setUser] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!user || !password) { setError('Введите логин и пароль'); return; }
    setBusy(true); setError('');
    try { await login(user, password); }
    catch (e: any) { setError(e.status === 401 ? 'Неверный логин или пароль' : e.message || 'Сервис недоступен, попробуйте позже'); }
    finally { setBusy(false); }
  };
  return <main className="login-screen">
    <div className="login-intro"><span className="login-line" aria-hidden="true"><i/><i/><i/></span><div className="login-brand"><span className="brand-mark"><span /></span><b>Ход города</b></div><h1>Город<br/>в движении.</h1><p>Прогноз трамвайной сети Москвы</p></div>
    <form className="login-card" onSubmit={submit} aria-label="Вход в систему">
      <span className="login-kicker">ДИСПЕТЧЕРСКАЯ</span>
      <h2>Войти</h2>
      {notice && !error && <div className="login-notice">{notice}</div>}
      <label>Логин<input autoFocus autoComplete="username" value={user} onChange={(e) => setUser(e.target.value)} /></label>
      <label>Пароль<input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
      {error && <div className="login-error" role="alert">{error}</div>}
      <button type="submit" disabled={busy}>{busy ? 'Проверяем…' : 'Войти'}<span aria-hidden="true">↗</span></button>
    </form>
  </main>;
}
