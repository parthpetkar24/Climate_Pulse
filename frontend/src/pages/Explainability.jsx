import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ScatterChart, Scatter,
  ZAxis
} from 'recharts';
import { Brain, Info, ShieldCheck, AlertTriangle, TrendingUp, Eye } from 'lucide-react';
import './Explainability.css';

/* ── Scroll reveal hook ────────────────────────────────────────── */
function useReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { el.classList.add('revealed'); obs.unobserve(el); } },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return ref;
}

function Reveal({ children, className = '', delay = 0 }) {
  const ref = useReveal();
  return (
    <div ref={ref} className={`reveal-section ${className}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

/* ── Mock SHAP data ────────────────────────────────────────────── */
const globalImportance = [
  { feature: 'pm25', importance: 42.8, color: '#5e8a5e' },
  { feature: 'pm10', importance: 28.3, color: '#7da37d' },
  { feature: 'pm_sum', importance: 12.1, color: '#5ba3a3' },
  { feature: 'max_pm_proxy', importance: 8.7, color: '#c4a65a' },
  { feature: 'no2', importance: 6.4, color: '#c48a5a' },
  { feature: 'pm_ratio', importance: 5.2, color: '#7aaec4' },
  { feature: 'co', importance: 3.9, color: '#c47a7a' },
  { feature: 'o3', importance: 3.1, color: '#a3bfa3' },
  { feature: 'so2', importance: 2.8, color: '#edd5a3' },
  { feature: 'day_of_year', importance: 1.5, color: '#c9d9c9' },
];

const localExplanation = [
  { feature: 'pm25', shap: 38.5, value: 85.2, direction: 'up' },
  { feature: 'pm10', shap: 22.1, value: 145.0, direction: 'up' },
  { feature: 'pm_sum', shap: 8.4, value: 230.2, direction: 'up' },
  { feature: 'no2', shap: -4.2, value: 18.0, direction: 'down' },
  { feature: 'o3', shap: -2.8, value: 22.5, direction: 'down' },
  { feature: 'day_of_year', shap: -1.5, value: 152, direction: 'down' },
];

const beeswarmData = [
  { feature: 'pm25', points: Array.from({ length: 30 }, (_, i) => ({ x: (Math.random() - 0.3) * 60, y: Math.random() * 100, z: Math.random() * 10 + 2 })) },
  { feature: 'pm10', points: Array.from({ length: 30 }, (_, i) => ({ x: (Math.random() - 0.35) * 40, y: Math.random() * 100, z: Math.random() * 8 + 2 })) },
  { feature: 'no2', points: Array.from({ length: 30 }, (_, i) => ({ x: (Math.random() - 0.5) * 20, y: Math.random() * 100, z: Math.random() * 6 + 2 })) },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="explain-tooltip">
      <p><strong>{label || payload[0]?.payload?.feature}</strong></p>
      {payload.map((e, i) => (
        <p key={i} style={{ color: e.color }}>
          {e.name}: {typeof e.value === 'number' ? e.value.toFixed(2) : e.value}
        </p>
      ))}
    </div>
  );
};

export default function Explainability() {
  const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5 },
  };

  return (
    <div className="explain">
      {/* ── Header ─────────────────────────────────── */}
      <section className="explain__header">
        <div className="container">
          <motion.div {...fadeUp}>
            <span className="section__tag">Model Explainability</span>
            <h1 className="explain__title">Understanding the Predictions</h1>
            <p className="explain__subtitle">
              Transparent AI: see exactly which features drive AQI predictions
              using SHAP (SHapley Additive exPlanations) analysis.
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container explain__body">
        {/* ── What Is SHAP ─────────────────────────── */}
        <Reveal>
          <div className="explain-intro">
            <div className="explain-intro__content">
              <h2>What is SHAP?</h2>
              <p>
                SHAP (SHapley Additive exPlanations) is a game-theoretic approach
                to explain machine learning predictions. It assigns each feature
                an importance value (SHAP value) for a particular prediction.
              </p>
              <div className="explain-intro__points">
                <div className="explain-intro__point">
                  <div className="explain-intro__point-icon" style={{ background: 'rgba(94,138,94,0.1)', color: 'var(--sage-600)' }}>
                    <TrendingUp size={18} />
                  </div>
                  <div>
                    <strong>Positive SHAP</strong>
                    <span>Pushes AQI prediction higher</span>
                  </div>
                </div>
                <div className="explain-intro__point">
                  <div className="explain-intro__point-icon" style={{ background: 'rgba(91,163,163,0.1)', color: 'var(--accent-teal)' }}>
                    <TrendingUp size={18} style={{ transform: 'rotate(180deg)' }} />
                  </div>
                  <div>
                    <strong>Negative SHAP</strong>
                    <span>Pulls AQI prediction lower</span>
                  </div>
                </div>
                <div className="explain-intro__point">
                  <div className="explain-intro__point-icon" style={{ background: 'rgba(196,166,90,0.1)', color: 'var(--accent-amber)' }}>
                    <Eye size={18} />
                  </div>
                  <div>
                    <strong>Influence ≠ Cause</strong>
                    <span>SHAP shows association, not causation</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="explain-intro__visual">
              <div className="explain-intro__formula">
                <span className="explain-intro__formula-label">Prediction Breakdown</span>
                <div className="explain-intro__equation">
                  <span className="explain-intro__eq-part explain-intro__eq-base">Base AQI</span>
                  <span className="explain-intro__eq-op">+</span>
                  <span className="explain-intro__eq-part explain-intro__eq-shap">Σ SHAP</span>
                  <span className="explain-intro__eq-op">=</span>
                  <span className="explain-intro__eq-part explain-intro__eq-result">Predicted AQI</span>
                </div>
                <div className="explain-intro__equation-detail">
                  <span>94.5</span>
                  <span>+</span>
                  <span>60.5</span>
                  <span>=</span>
                  <span>155.0</span>
                </div>
              </div>
            </div>
          </div>
        </Reveal>

        {/* ── Global Feature Importance ────────────── */}
        <Reveal delay={100}>
          <div className="explain-card">
            <div className="explain-card__header">
              <h3>
                <Brain size={18} />
                Global Feature Importance
              </h3>
              <p>Mean absolute SHAP value across all test samples — shows which features matter most overall.</p>
            </div>
            <ResponsiveContainer width="100%" height={380}>
              <BarChart
                data={[...globalImportance].reverse()}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
                <XAxis type="number" tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 12, fill: 'var(--text-secondary)' }}
                  width={80}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="importance" name="Mean |SHAP|" radius={[0, 4, 4, 0]}>
                  {[...globalImportance].reverse().map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Reveal>

        {/* ── Local Explanation (Waterfall) ─────────── */}
        <Reveal delay={200}>
          <div className="explain-card">
            <div className="explain-card__header">
              <h3>
                <Info size={18} />
                Local Explanation — Sample Prediction
              </h3>
              <p>How each feature contributed to a single prediction (AQI = 155, Moderate).</p>
            </div>
            <div className="waterfall-chart">
              <div className="waterfall-base">
                <span className="waterfall-label">Base Value</span>
                <span className="waterfall-val">94.5</span>
              </div>
              {localExplanation.map((f, i) => (
                <div key={i} className={`waterfall-row waterfall-row--${f.direction}`}>
                  <span className="waterfall-feature">{f.feature}</span>
                  <div className="waterfall-bar-wrap">
                    <div
                      className={`waterfall-bar waterfall-bar--${f.direction}`}
                      style={{ width: `${Math.min(Math.abs(f.shap) * 2, 100)}%` }}
                    >
                      <span className="waterfall-bar__val">
                        {f.shap > 0 ? '+' : ''}{f.shap.toFixed(1)}
                      </span>
                    </div>
                  </div>
                  <span className="waterfall-fval">{f.value}</span>
                </div>
              ))}
              <div className="waterfall-result">
                <span className="waterfall-label">Prediction</span>
                <span className="waterfall-val waterfall-val--result">155.0</span>
              </div>
            </div>
          </div>
        </Reveal>

        {/* ── Feature Interaction ──────────────────── */}
        <Reveal delay={100}>
          <div className="explain-card">
            <div className="explain-card__header">
              <h3>
                <Eye size={18} />
                PM2.5 Feature Dependence
              </h3>
              <p>How PM2.5 concentration relates to its SHAP contribution. Each dot is one prediction.</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
                <XAxis
                  type="number"
                  dataKey="y"
                  name="PM2.5"
                  tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }}
                  label={{ value: 'PM2.5 Value (µg/m³)', position: 'bottom', fontSize: 12, fill: 'var(--text-tertiary)' }}
                />
                <YAxis
                  type="number"
                  dataKey="x"
                  name="SHAP"
                  tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }}
                  label={{ value: 'SHAP Value', angle: -90, position: 'insideLeft', fontSize: 12, fill: 'var(--text-tertiary)' }}
                />
                <ZAxis type="number" dataKey="z" range={[20, 80]} />
                <Tooltip content={<CustomTooltip />} />
                <Scatter data={beeswarmData[0].points} fill="var(--sage-500)" opacity={0.6} name="PM2.5" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Reveal>

        {/* ── Leakage Audit ────────────────────────── */}
        <Reveal delay={100}>
          <div className="explain-card explain-card--audit">
            <div className="explain-card__header">
              <h3>
                <ShieldCheck size={18} />
                Data Leakage Audit
              </h3>
              <p>Our model has been audited to prevent inflated accuracy metrics.</p>
            </div>
            <div className="audit-grid">
              <div className="audit-item audit-item--pass">
                <div className="audit-item__icon">✓</div>
                <div>
                  <strong>Temporal Split</strong>
                  <p>Train/val/test split is chronological — no future data leaks into training.</p>
                </div>
              </div>
              <div className="audit-item audit-item--pass">
                <div className="audit-item__icon">✓</div>
                <div>
                  <strong>Sub-index Columns Removed</strong>
                  <p>sub_pm25, sub_pm10, etc. are derived from AQI and were excluded as features.</p>
                </div>
              </div>
              <div className="audit-item audit-item--pass">
                <div className="audit-item__icon">✓</div>
                <div>
                  <strong>Dominant Pollutant Removed</strong>
                  <p>This column directly reveals the answer and was dropped before training.</p>
                </div>
              </div>
              <div className="audit-item audit-item--info">
                <div className="audit-item__icon">i</div>
                <div>
                  <strong>Known Limitation</strong>
                  <p>AQI is calculated from pollutant concentrations used as inputs. The model learns the CPCB formula mapping — this is acceptable for explainability.</p>
                </div>
              </div>
            </div>
          </div>
        </Reveal>

        {/* ── Disclaimer ───────────────────────────── */}
        <Reveal delay={100}>
          <div className="explain-disclaimer">
            <AlertTriangle size={20} />
            <div>
              <strong>Important Disclaimer</strong>
              <p>
                SHAP values indicate how much each feature <em>influenced</em> the
                model's prediction. They do not prove that a pollutant directly
                <em> caused</em> an AQI change — external factors, measurement
                errors, and complex interactions may also play a role.
              </p>
            </div>
          </div>
        </Reveal>
      </div>
    </div>
  );
}
