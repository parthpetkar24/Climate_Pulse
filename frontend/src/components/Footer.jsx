import { Link } from 'react-router-dom';
import { Wind, Activity, Cpu, Shield, Sparkles } from 'lucide-react';
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
                Unified environmental intelligence platform combining multi-source sensory data,
                leakage-audited machine learning, and cited AI explanations for actionable air and climate resilience.
              </p>
              <div className="footer__badge-row">
                <span className="footer__tech-pill"><Cpu size={12} /> PyTorch & XGBoost</span>
                <span className="footer__tech-pill"><Sparkles size={12} /> SHAP XAI</span>
                <span className="footer__tech-pill"><Shield size={12} /> WHO Grounded</span>
              </div>
            </div>

            <div className="footer__col">
              <h4 className="footer__heading">Platform MVPs</h4>
              <ul className="footer__list">
                <li><Link to="/predict">MVP 1: AQI Predict & SHAP</Link></li>
                <li><Link to="/forecast">MVP 2: 7-Day Forecaster</Link></li>
                <li><Link to="/simulator">MVP 3: Climate Simulator</Link></li>
                <li><Link to="/carbon">MVP 4: Carbon Audit Copilot</Link></li>
              </ul>
            </div>

            <div className="footer__col">
              <h4 className="footer__heading">Analytics & Tools</h4>
              <ul className="footer__list">
                <li><Link to="/dashboard">City Analytics Dashboard</Link></li>
                <li><Link to="/explainability">Global SHAP Attribution</Link></li>
                <li><Link to="/predict">Cited Health Advisory</Link></li>
                <li><Link to="/carbon">Scope 1–3 GHG Accounting</Link></li>
              </ul>
            </div>

            <div className="footer__col">
              <h4 className="footer__heading">Standards & Benchmarks</h4>
              <ul className="footer__list">
                <li><span>CPCB National AQI Scale</span></li>
                <li><span>NCAP 40% Reduction Target</span></li>
                <li><span>WHO Air Quality Guidelines 2021</span></li>
                <li><span>GHG Protocol Corporate Standard</span></li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
