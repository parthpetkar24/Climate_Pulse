import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Wind, Menu, X, Sparkles } from 'lucide-react';
import './Navbar.css';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [location]);

  const links = [
    { to: '/',               label: 'Home' },
    { to: '/dashboard',      label: 'Dashboard' },
    { to: '/predict',        label: 'AQI Predict & XAI' },
    { to: '/forecast',       label: '7-Day Forecast' },
    { to: '/simulator',      label: 'Climate Simulator' },
    { to: '/carbon',         label: 'Carbon Audit' },
  ];

  return (
    <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
      <div className="navbar__inner container">
        <Link to="/" className="navbar__brand">
          <div className="navbar__logo-icon">
            <Wind size={22} strokeWidth={2.2} />
          </div>
          <span className="navbar__logo-text">
            Climate<span className="navbar__logo-accent">Pulse</span>
          </span>
        </Link>

        <ul className={`navbar__links ${mobileOpen ? 'navbar__links--open' : ''}`}>
          {links.map(({ to, label }) => {
            const isActive = location.pathname === to;
            return (
              <li key={to}>
                <Link
                  to={to}
                  className={`navbar__link ${isActive ? 'navbar__link--active' : ''}`}
                >
                  {label}
                  {isActive && <span className="navbar__link-dot" />}
                </Link>
              </li>
            );
          })}
        </ul>

        <button
          className="navbar__toggle"
          onClick={() => setMobileOpen(prev => !prev)}
          aria-label="Toggle navigation"
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </nav>
  );
}
