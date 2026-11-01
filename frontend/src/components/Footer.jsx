import { Link } from 'react-router-dom';
import { Wind, GitFork, Mail, ExternalLink } from 'lucide-react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer__wave">
        <svg viewBox="0 0 1440 120" preserveAspectRatio="none">
          <path
            d="M0,60 C360,120 720,0 1080,60 C1260,90 1380,50 1440,60 L1440,120 L0,120 Z"
            fill="var(--bg-dark)"
          />
        </svg>
      </div>

      <div className="footer__body">
        <div className="container">
          <div className="footer__grid">
            <div className="footer__col footer__col--brand">
              <div className="footer__brand">
                <div className="footer__brand-icon">
                  <Wind size={20} />
                </div>
                <span className="footer__brand-text">
                  Climate<span>Pulse</span>
                </span>
              </div>
              <p className="footer__desc">
                An environmental intelligence platform combining multi-source data,
                machine learning, and AI-powered explanations to deliver actionable insights.
              </p>
            </div>

            <div className="footer__col">
              <h4 className="footer__heading">Platform</h4>
              <ul className="footer__list">
                <li><Link to="/dashboard">AQI Dashboard</Link></li>
                <li><Link to="/predict">Predict AQI</Link></li>
                <li><Link to="/explainability">Explainability</Link></li>
                <li><Link to="/about">About Us</Link></li>
              </ul>
            </div>

            <div className="footer__col">
              <h4 className="footer__heading">Data Sources</h4>
              <ul className="footer__list">
                <li><a href="https://openaq.org" target="_blank" rel="noopener noreferrer">OpenAQ <ExternalLink size={12} /></a></li>
                <li><a href="https://cpcb.nic.in" target="_blank" rel="noopener noreferrer">CPCB India <ExternalLink size={12} /></a></li>
                <li><a href="https://power.larc.nasa.gov" target="_blank" rel="noopener noreferrer">NASA POWER <ExternalLink size={12} /></a></li>
                <li><a href="https://www.who.int" target="_blank" rel="noopener noreferrer">WHO Guidelines <ExternalLink size={12} /></a></li>
              </ul>
            </div>

            <div className="footer__col">
              <h4 className="footer__heading">Team</h4>
              <ul className="footer__list">
                <li className="footer__team-name">CodeCrew</li>
                <li>PCCOE Hackathon 2026</li>
              </ul>
              <div className="footer__social">
                <a href="#" className="footer__social-link" aria-label="GitHub">
                  <GitFork size={18} />
                </a>
                <a href="#" className="footer__social-link" aria-label="Email">
                  <Mail size={18} />
                </a>
              </div>
            </div>
          </div>

          <div className="footer__bottom">
            <p>&copy; {new Date().getFullYear()} ClimatePulse by CodeCrew. Built for a better tomorrow.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
