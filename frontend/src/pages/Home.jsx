import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Wind, BarChart3, Brain, ShieldCheck,
  ArrowRight, Leaf, CloudSun, Activity,
  Gauge, Microscope, Users, Building2
} from 'lucide-react';
import './Home.css';

/* ── Scroll-triggered visibility hook ──────────────────────────── */
function useReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add('revealed');
          observer.unobserve(el);
        }
      },
      { threshold: 0.15 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return ref;
}

function RevealSection({ children, className = '', delay = 0 }) {
  const ref = useReveal();
  return (
    <div
      ref={ref}
      className={`reveal-section ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/* ── Animated counter ──────────────────────────────────────────── */
function Counter({ end, suffix = '', duration = 2000 }) {
  const ref = useRef(null);
  const counted = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !counted.current) {
        counted.current = true;
        let start = 0;
        const startTime = performance.now();
        const step = (now) => {
          const progress = Math.min((now - startTime) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          el.textContent = Math.floor(eased * end) + suffix;
          if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      }
    }, { threshold: 0.5 });
    observer.observe(el);
    return () => observer.disconnect();
  }, [end, suffix, duration]);

  return <span ref={ref}>0{suffix}</span>;
}

export default function Home() {
  return (
    <div className="home">
      {/* ═══ Hero Section ═══ */}
      <section className="hero">
        <div className="hero__bg">
          <div className="hero__orb hero__orb--1" />
          <div className="hero__orb hero__orb--2" />
          <div className="hero__orb hero__orb--3" />
          <div className="hero__grid-lines" />
        </div>

        <div className="container hero__content">
          <motion.div
            className="hero__badge"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Leaf size={14} />
            <span>Environmental Intelligence Platform</span>
          </motion.div>

          <motion.h1
            className="hero__title"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.35 }}
          >
            Understand Your Air.
            <br />
            <span className="hero__title-accent">Protect Your Health.</span>
          </motion.h1>

          <motion.p
            className="hero__subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.55 }}
          >
            Climate Pulse fuses satellite data, ground sensors, and machine learning
            into one clear picture of your local air quality — with explanations
            you can actually trust.
          </motion.p>

          <motion.div
            className="hero__actions"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.7 }}
          >
            <Link to="/predict" className="btn btn--primary">
              Predict AQI Now
              <ArrowRight size={18} />
            </Link>
            <Link to="/dashboard" className="btn btn--outline">
              View Dashboard
            </Link>
          </motion.div>

          <motion.div
            className="hero__visual"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.9 }}
          >
            <div className="hero__card-stack">
              <div className="hero__aqi-card">
                <div className="hero__aqi-header">
                  <Wind size={18} />
                  <span>Current AQI Estimate</span>
                </div>
                <div className="hero__aqi-value">
                  <span className="hero__aqi-number">127</span>
                  <span className="hero__aqi-label">Moderate</span>
                </div>
                <div className="hero__aqi-bar">
                  <div className="hero__aqi-bar-fill" style={{ width: '42%' }} />
                </div>
                <div className="hero__aqi-pollutants">
                  <div className="hero__pollutant"><span>PM2.5</span><strong>68.4</strong></div>
                  <div className="hero__pollutant"><span>PM10</span><strong>112.0</strong></div>
                  <div className="hero__pollutant"><span>NO₂</span><strong>34.2</strong></div>
                  <div className="hero__pollutant"><span>O₃</span><strong>28.7</strong></div>
                </div>
              </div>

              <div className="hero__float-card hero__float-card--shap">
                <Brain size={16} />
                <div>
                  <span className="hero__float-label">SHAP Insight</span>
                  <span className="hero__float-value">PM2.5 drives 68% of prediction</span>
                </div>
              </div>

              <div className="hero__float-card hero__float-card--who">
                <ShieldCheck size={16} />
                <div>
                  <span className="hero__float-label">WHO Guideline</span>
                  <span className="hero__float-value">15 µg/m³ annual mean</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ═══ Stats Bar ═══ */}
      <section className="stats-bar">
        <div className="container">
          <div className="stats-bar__grid">
            {[
              { value: 6, suffix: '+', label: 'Pollutants Tracked' },
              { value: 22, suffix: '', label: 'ML Features' },
              { value: 95, suffix: '%', label: 'Model Accuracy' },
              { value: 5, suffix: '', label: 'Data Sources' },
            ].map((stat, i) => (
              <div key={i} className="stats-bar__item">
                <span className="stats-bar__value">
                  <Counter end={stat.value} suffix={stat.suffix} />
                </span>
                <span className="stats-bar__label">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ Problem Section ═══ */}
      <section className="section problem-section">
        <div className="container">
          <RevealSection>
            <div className="section__header">
              <span className="section__tag">The Challenge</span>
              <h2 className="section__title">
                Environmental Data Is Fragmented
              </h2>
              <p className="section__desc">
                India generates vast environmental data through AQI monitoring stations,
                satellites, weather systems, and IoT sensors — but it sits scattered
                across dozens of disconnected sources, making unified analysis nearly impossible.
              </p>
            </div>
          </RevealSection>

          <div className="problem-grid">
            {[
              {
                icon: BarChart3,
                title: 'AQI Prediction Tools',
                problem: 'Limited to a single city or pollutant, mostly descriptive rather than prescriptive.',
              },
              {
                icon: Brain,
                title: 'LLM-based Systems',
                problem: 'Often lack grounded, citable environmental evidence and hallucinate data.',
              },
              {
                icon: Gauge,
                title: 'Carbon Calculators',
                problem: 'Depend heavily on manual questionnaires, covering only Scope 1 emissions.',
              },
              {
                icon: CloudSun,
                title: 'Climate ML Models',
                problem: 'Rarely audited for data leakage — misleading accuracy without temporal splits.',
              },
            ].map((item, i) => (
              <RevealSection key={i} delay={i * 100}>
                <div className="problem-card">
                  <div className="problem-card__icon">
                    <item.icon size={22} />
                  </div>
                  <h3 className="problem-card__title">{item.title}</h3>
                  <p className="problem-card__text">{item.problem}</p>
                </div>
              </RevealSection>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ Solution Section ═══ */}
      <section className="section solution-section">
        <div className="container">
          <RevealSection>
            <div className="section__header">
              <span className="section__tag">Our Solution</span>
              <h2 className="section__title">
                One Platform. Full Picture.
              </h2>
              <p className="section__desc">
                Climate Pulse combines multi-source environmental data, leakage-audited
                machine learning, SHAP-powered explanations, and WHO-cited health
                intelligence into a single modular platform.
              </p>
            </div>
          </RevealSection>

          <div className="solution-features">
            {[
              {
                icon: Activity,
                title: 'Multi-Source Data Fusion',
                desc: 'Government stations, satellite imagery, and IoT sensors merged into one view.',
                color: 'sage',
              },
              {
                icon: Brain,
                title: 'Explainable AI',
                desc: 'SHAP explanations show exactly which pollutants drove each prediction.',
                color: 'teal',
              },
              {
                icon: Microscope,
                title: 'WHO Evidence Retrieval',
                desc: 'Health explanations grounded in WHO guidelines with proper page citations.',
                color: 'amber',
              },
              {
                icon: ShieldCheck,
                title: 'Leakage-Audited Models',
                desc: 'Temporal train/test splits and automatic leakage detection prevent inflated metrics.',
                color: 'rose',
              },
            ].map((feat, i) => (
              <RevealSection key={i} delay={i * 120}>
                <div className={`solution-card solution-card--${feat.color}`}>
                  <div className="solution-card__icon-wrap">
                    <feat.icon size={24} />
                  </div>
                  <div className="solution-card__body">
                    <h3>{feat.title}</h3>
                    <p>{feat.desc}</p>
                  </div>
                  <div className="solution-card__line" />
                </div>
              </RevealSection>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ How It Works ═══ */}
      <section className="section how-section">
        <div className="container">
          <RevealSection>
            <div className="section__header">
              <span className="section__tag">How It Works</span>
              <h2 className="section__title">
                From Raw Data to Actionable Intelligence
              </h2>
            </div>
          </RevealSection>

          <div className="pipeline">
            {[
              { step: '01', title: 'Collect', desc: 'Multi-source environmental data ingestion from CPCB, OpenAQ, NASA POWER, and satellite feeds.' },
              { step: '02', title: 'Process', desc: 'Leakage-safe preprocessing with temporal feature engineering, forward-fill imputation, and domain features.' },
              { step: '03', title: 'Predict', desc: 'Ensemble ML models (RandomForest, XGBoost) trained with chronological splits for honest evaluation.' },
              { step: '04', title: 'Explain', desc: 'SHAP analysis identifies influential factors. WHO knowledge base retrieves cited health evidence.' },
              { step: '05', title: 'Act', desc: 'Clear dashboards, health advisories, and prescriptive recommendations delivered to users.' },
            ].map((item, i) => (
              <RevealSection key={i} delay={i * 100}>
                <div className="pipeline__step">
                  <div className="pipeline__number">{item.step}</div>
                  <div className="pipeline__connector" />
                  <div className="pipeline__content">
                    <h3>{item.title}</h3>
                    <p>{item.desc}</p>
                  </div>
                </div>
              </RevealSection>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ Impact Section ═══ */}
      <section className="section impact-section">
        <div className="container">
          <RevealSection>
            <div className="section__header">
              <span className="section__tag">Impact & Benefits</span>
              <h2 className="section__title">Built for Everyone</h2>
            </div>
          </RevealSection>

          <div className="impact-grid">
            {[
              {
                icon: Users,
                title: 'Citizens',
                points: [
                  'Understand local air quality in plain language',
                  'Receive clearer health-risk insights',
                  'Get personalised safety recommendations',
                ],
              },
              {
                icon: Building2,
                title: 'Local Governments',
                points: [
                  'Simulate interventions before allocating resources',
                  'Compare potential environmental outcomes',
                  'Data-driven urban planning support',
                ],
              },
              {
                icon: Leaf,
                title: 'SMEs & Individuals',
                points: [
                  'Calculate carbon emissions for Scopes 1–3',
                  'Skip lengthy manual questionnaires',
                  'Track sustainability progress over time',
                ],
              },
              {
                icon: Microscope,
                title: 'Researchers & Policy',
                points: [
                  'Audit climate ML models for data leakage',
                  'Standardised model cards for peer review',
                  'Link India\'s climate targets with intelligence',
                ],
              },
            ].map((group, i) => (
              <RevealSection key={i} delay={i * 100}>
                <div className="impact-card">
                  <div className="impact-card__icon">
                    <group.icon size={24} />
                  </div>
                  <h3 className="impact-card__title">{group.title}</h3>
                  <ul className="impact-card__list">
                    {group.points.map((pt, j) => (
                      <li key={j}>{pt}</li>
                    ))}
                  </ul>
                </div>
              </RevealSection>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ CTA Section ═══ */}
      <section className="cta-section">
        <div className="container">
          <RevealSection>
            <div className="cta-box">
              <div className="cta-box__bg" />
              <div className="cta-box__content">
                <h2>Ready to explore your air quality?</h2>
                <p>
                  Try our AQI prediction engine with real-time SHAP explanations
                  and WHO-backed health insights.
                </p>
                <div className="cta-box__actions">
                  <Link to="/predict" className="btn btn--primary btn--lg">
                    Start Predicting
                    <ArrowRight size={18} />
                  </Link>
                  <Link to="/about" className="btn btn--ghost btn--lg">
                    Learn More About Us
                  </Link>
                </div>
              </div>
            </div>
          </RevealSection>
        </div>
      </section>
    </div>
  );
}
