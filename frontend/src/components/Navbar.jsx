import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Wind, Menu, X } from 'lucide-react';
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
    { to: '/predict',        label: 'Predict AQI' },
    { to: '/explainability', label: 'Explainability' },
    { to: '/about',          label: 'About' },
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
          {links.map(({ to, label }) => (
            <li key={to}>
              <Link
                to={to}
                className={`navbar__link ${location.pathname === to ? 'navbar__link--active' : ''}`}
              >
                {label}
                {location.pathname === to && <span className="navbar__link-dot" />}
              </Link>
            </li>
          ))}
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
