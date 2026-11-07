import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Wind, Database, Brain, ShieldCheck, Leaf,
  Code2, Cpu, BarChart3, Globe, BookOpen,
  GitFork, Globe2, Mail, ExternalLink,
  Server, Layers, FlaskConical
} from 'lucide-react';
import './About.css';

/* ── Scroll reveal ─────────────────────────────────────────────── */
function useReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { el.classList.add('revealed'); obs.unobserve(el); } },
      { threshold: 0.12 }
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

/* ── Data ──────────────────────────────────────────────────────── */
const teamMembers = [
  {
    name: 'Team CodeCrew',
    role: 'Full-stack Development',
    desc: 'Building the bridge between environmental data and actionable intelligence at PCCOE Hackathon 2026.',
    initials: 'CC',
    gradient: 'linear-gradient(135deg, var(--sage-500), var(--sage-700))',
  },
];

const techStack = [
  {
    layer: 'Frontend',
    icon: Code2,
    color: 'sage',
    techs: ['React', 'Vite', 'JavaScript', 'Recharts', 'Framer Motion'],
  },
  {
    layer: 'Backend',
    icon: Server,
    color: 'teal',
    techs: ['Python', 'FastAPI'],
  },
  {
    layer: 'ML / AI',
    icon: Cpu,
    color: 'amber',
    techs: ['RandomForest', 'XGBoost', 'SVM', 'SHAP', 'Groq LLM', 'LangChain'],
  },
  {
    layer: 'Data & Deploy',
    icon: Database,
    color: 'rose',
    techs: ['PostgreSQL', 'pgvector', 'Supabase', 'Docker', 'Render'],
  },
];

const dataSources = [
  { name: 'OpenAQ', desc: 'Global open air quality data from monitoring stations worldwide', url: 'https://openaq.org' },
  { name: 'CPCB India', desc: 'Central Pollution Control Board — official Indian AQI monitoring', url: 'https://cpcb.nic.in' },
  { name: 'NASA POWER', desc: 'Meteorological and solar data from NASA satellite observations', url: 'https://power.larc.nasa.gov' },
  { name: 'WHO Guidelines', desc: 'World Health Organization global air quality guidelines 2021', url: 'https://www.who.int' },
  { name: 'Sentinel-2 / Landsat', desc: 'Satellite imagery for land cover and vegetation monitoring', url: '#' },
  { name: 'Climate TRACE', desc: 'Independent greenhouse gas emissions tracking', url: 'https://climatetrace.org' },
];

const milestones = [
  { phase: 'MVP 1', title: 'AQI Explainability Engine', status: 'complete', desc: 'SHAP-based model explanations with WHO evidence retrieval and leakage-audited ML pipeline.' },
  { phase: 'MVP 2', title: 'Hyperlocal AQI Forecasting', status: 'in-progress', desc: 'Multi-station predictions with attention-based CNN-LSTM and satellite data fusion.' },
  { phase: 'MVP 3', title: 'Carbon Footprint Calculator', status: 'planned', desc: 'Scope 1–3 emissions tracking without manual questionnaires using LLM-driven data extraction.' },
  { phase: 'MVP 4', title: 'Intervention Simulator', status: 'planned', desc: 'Prescriptive tool to simulate policy interventions and predict environmental outcomes.' },
];

export default function About() {
  const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 },
  };

  return (
    <div className="about">
      {/* ── Hero ───────────────────────────────────── */}
      <section className="about__hero">
        <div className="about__hero-bg">
          <div className="about__hero-orb about__hero-orb--1" />
          <div className="about__hero-orb about__hero-orb--2" />
        </div>
        <div className="container about__hero-content">
          <motion.div {...fadeUp}>
            <span className="section__tag">About Climate Pulse</span>
            <h1 className="about__hero-title">
              Environmental Intelligence,
              <br />
              <span>Made Transparent.</span>
            </h1>
            <p className="about__hero-desc">
              Climate Pulse is an environmental intelligence platform built by Team CodeCrew
              at the PCCOE Hackathon. We combine multi-source data, leakage-audited machine learning,
              and explainable AI to deliver clear, trustworthy environmental insights.
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container about__body">
        {/* ── Mission ──────────────────────────────── */}
        <Reveal>
          <div className="about-mission">
            <div className="about-mission__content">
              <h2>Our Mission</h2>
              <p>
                India generates enormous volumes of environmental data through monitoring stations,
                satellites, weather systems, and IoT sensors. But this data is fragmented across
                dozens of disconnected sources, making unified analysis nearly impossible.
              </p>
              <p>
                Existing solutions are either limited to a single pollutant, lack citable evidence,
                or use ML models that haven't been audited for data leakage — producing misleadingly
                high accuracy metrics.
              </p>
              <p className="about-mission__highlight">
                <strong>Climate Pulse changes this.</strong> We fuse multiple data sources into
                one coherent picture, explain every prediction with SHAP analysis, cite WHO
                health evidence with page-level precision, and audit our models for temporal
                leakage to ensure honest metrics.
              </p>
            </div>
            <div className="about-mission__values">
              {[
                { icon: Brain, title: 'Explainable', desc: 'Every prediction comes with human-readable SHAP explanations.' },
                { icon: ShieldCheck, title: 'Trustworthy', desc: 'Leakage-audited models with temporal train/test splits.' },
                { icon: BookOpen, title: 'Evidence-Based', desc: 'Health claims grounded in WHO guidelines with page citations.' },
                { icon: Globe, title: 'Multi-Source', desc: 'Fusing government, satellite, and IoT sensor data.' },
              ].map((v, i) => (
                <div key={i} className="about-value">
                  <div className="about-value__icon">
                    <v.icon size={20} />
                  </div>
                  <div>
                    <strong>{v.title}</strong>
                    <span>{v.desc}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>

        {/* ── Roadmap ──────────────────────────────── */}
        <Reveal>
          <div className="about-section">
            <div className="about-section__header">
              <span className="section__tag">Roadmap</span>
              <h2>Platform Development</h2>
              <p>Four modular MVPs, each addressing a distinct environmental challenge.</p>
            </div>
            <div className="roadmap">
              {milestones.map((m, i) => (
                <Reveal key={i} delay={i * 80}>
                  <div className={`roadmap-card roadmap-card--${m.status}`}>
                    <div className="roadmap-card__phase">{m.phase}</div>
                    <div className="roadmap-card__body">
                      <h3>{m.title}</h3>
                      <p>{m.desc}</p>
                    </div>
                    <div className={`roadmap-card__status roadmap-card__status--${m.status}`}>
                      {m.status === 'complete' ? '✓ Complete' : m.status === 'in-progress' ? '● In Progress' : '○ Planned'}
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </Reveal>

        {/* ── Tech Stack ───────────────────────────── */}
        <Reveal>
          <div className="about-section">
            <div className="about-section__header">
              <span className="section__tag">Technology</span>
              <h2>Tech Stack</h2>
              <p>A modern, modular architecture designed for reliability and scale.</p>
            </div>
            <div className="tech-grid">
              {techStack.map((layer, i) => (
                <Reveal key={i} delay={i * 80}>
                  <div className={`tech-card tech-card--${layer.color}`}>
                    <div className="tech-card__header">
                      <div className={`tech-card__icon tech-card__icon--${layer.color}`}>
                        <layer.icon size={20} />
                      </div>
                      <h3>{layer.layer}</h3>
                    </div>
                    <div className="tech-card__tags">
                      {layer.techs.map((t, j) => (
                        <span key={j} className="tech-tag">{t}</span>
                      ))}
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </Reveal>

        {/* ── Data Sources ─────────────────────────── */}
        <Reveal>
          <div className="about-section">
            <div className="about-section__header">
              <span className="section__tag">Data</span>
              <h2>Data Sources</h2>
              <p>Multi-source environmental data for comprehensive intelligence.</p>
            </div>
            <div className="datasource-grid">
              {dataSources.map((ds, i) => (
                <Reveal key={i} delay={i * 60}>
                  <a
                    href={ds.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="datasource-card"
                  >
                    <div className="datasource-card__header">
                      <Database size={16} />
                      <strong>{ds.name}</strong>
                      {ds.url !== '#' && <ExternalLink size={12} className="datasource-card__ext" />}
                    </div>
                    <p>{ds.desc}</p>
                  </a>
                </Reveal>
              ))}
            </div>
          </div>
        </Reveal>

        {/* ── Team ─────────────────────────────────── */}
        <Reveal>
          <div className="about-section about-team-section">
            <div className="about-section__header">
              <span className="section__tag">The Team</span>
              <h2>Built by CodeCrew</h2>
              <p>A multidisciplinary team from PCCOE tackling environmental challenges with technology.</p>
            </div>
            <div className="team-showcase">
              <div className="team-crest">
                <div className="team-crest__avatar" style={{ background: teamMembers[0].gradient }}>
                  <span>{teamMembers[0].initials}</span>
                </div>
                <h3>Team CodeCrew</h3>
                <p className="team-crest__role">PCCOE Hackathon 2026</p>
                <p className="team-crest__desc">{teamMembers[0].desc}</p>
                <div className="team-crest__social">
                  <a href="#" className="team-social-link" aria-label="GitHub"><GitFork size={18} /></a>
                  <a href="#" className="team-social-link" aria-label="LinkedIn"><Globe2 size={18} /></a>
                  <a href="#" className="team-social-link" aria-label="Email"><Mail size={18} /></a>
                </div>
              </div>

              <div className="team-stats">
                {[
                  { icon: Layers, label: 'MVPs Planned', value: '4' },
                  { icon: FlaskConical, label: 'ML Models', value: '6+' },
                  { icon: BarChart3, label: 'Features Engineered', value: '22' },
                  { icon: Leaf, label: 'Data Sources', value: '5+' },
                ].map((s, i) => (
                  <div key={i} className="team-stat">
                    <div className="team-stat__icon"><s.icon size={18} /></div>
                    <div>
                      <span className="team-stat__value">{s.value}</span>
                      <span className="team-stat__label">{s.label}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Reveal>

        {/* ── Open Source Notice ────────────────────── */}
        <Reveal>
          <div className="about-opensource">
            <div className="about-opensource__icon">
              <Leaf size={28} />
            </div>
            <h3>Built for a Better Tomorrow</h3>
            <p>
              Climate Pulse is an open-source project developed during the PCCOE Hackathon 2026.
              We believe environmental intelligence should be transparent, accessible, and
              grounded in trusted scientific evidence.
            </p>
            <a href="#" className="btn btn--primary">
              <GitFork size={18} />
              View on GitHub
            </a>
          </div>
        </Reveal>
      </div>
    </div>
  );
}
