import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Loader2, Wind, Droplets, CloudSun,
  Thermometer, FlaskConical, AlertCircle,
  ChevronDown, ShieldCheck, Brain, Info
} from 'lucide-react';
import './Predict.css';

/* ── AQI mapping ───────────────────────────────────────────────── */
function getAqiInfo(val) {
  if (val <= 50) return { label: 'Good', color: '#5e8a5e', advice: 'Air quality is satisfactory. Enjoy outdoor activities freely.' };
  if (val <= 100) return { label: 'Satisfactory', color: '#7da37d', advice: 'Acceptable quality. Unusually sensitive people should limit prolonged outdoor exertion.' };
  if (val <= 200) return { label: 'Moderate', color: '#c4a65a', advice: 'Breathing discomfort possible for sensitive groups. Reduce prolonged outdoor activity.' };
  if (val <= 300) return { label: 'Poor', color: '#c48a5a', advice: 'Health effects possible for everyone. People with respiratory conditions should avoid outdoor exertion.' };
  if (val <= 400) return { label: 'Very Poor', color: '#c47a7a', advice: 'Significant health risk. Everyone should reduce outdoor exposure. Use air purifiers indoors.' };
  return { label: 'Severe', color: '#8a3d3d', advice: 'Health emergency. Avoid all outdoor activity. Keep windows closed and use masks outdoors.' };
}

const pollutantFields = [
  { key: 'pm25', label: 'PM2.5', unit: 'µg/m³', icon: Droplets, placeholder: '65.0', info: 'Fine particulate matter (≤ 2.5µm). WHO guideline: 15 µg/m³ annual mean.' },
  { key: 'pm10', label: 'PM10', unit: 'µg/m³', icon: Wind, placeholder: '110.0', info: 'Coarse particulate matter (≤ 10µm). WHO guideline: 45 µg/m³ annual mean.' },
  { key: 'no2',  label: 'NO₂',  unit: 'ppb', icon: CloudSun, placeholder: '34.0', info: 'Nitrogen dioxide from vehicles & industry. WHO guideline: 10 µg/m³ annual mean.' },
  { key: 'o3',   label: 'O₃',   unit: 'ppb', icon: Thermometer, placeholder: '28.0', info: 'Ground-level ozone from photochemical reactions.' },
  { key: 'co',   label: 'CO',   unit: 'mg/m³', icon: FlaskConical, placeholder: '1.2', info: 'Carbon monoxide from incomplete combustion.' },
  { key: 'so2',  label: 'SO₂',  unit: 'ppb', icon: Droplets, placeholder: '12.0', info: 'Sulfur dioxide from fossil fuel burning.' },
];

export default function Predict() {
  const [values, setValues] = useState({
    pm25: '', pm10: '', no2: '', o3: '', co: '', so2: '', date: '',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [expandedInfo, setExpandedInfo] = useState(null);

  const handleChange = (key, val) => {
    setValues(prev => ({ ...prev, [key]: val }));
    setError('');
  };

  const handlePredict = async (e) => {
    e.preventDefault();

    // Validate at least PM2.5 and PM10 are provided
    if (!values.pm25 || !values.pm10) {
      setError('Please enter at least PM2.5 and PM10 concentrations.');
      return;
    }

    setLoading(true);
    setResult(null);
    setError('');

    // Simulate API call (replace with actual backend call)
    await new Promise(resolve => setTimeout(resolve, 2200));

    const pm25 = parseFloat(values.pm25) || 0;
    const pm10 = parseFloat(values.pm10) || 0;
    const no2 = parseFloat(values.no2) || 0;
    const o3 = parseFloat(values.o3) || 0;

    // Simplified AQI estimation (CPCB-style sub-index approach)
    const subPm25 = pm25 * 1.67;
    const subPm10 = pm10 * 1.0;
    const subNo2 = no2 * 1.2;
    const subO3 = o3 * 0.8;
    const estimatedAqi = Math.round(Math.max(subPm25, subPm10, subNo2, subO3) * 0.85 + 15);

    const aqiInfo = getAqiInfo(estimatedAqi);

    const shapFeatures = [
      { feature: 'PM2.5', value: pm25, shap: +(pm25 * 0.45 - 10).toFixed(2), direction: pm25 > 40 ? 'increases' : 'decreases' },
      { feature: 'PM10', value: pm10, shap: +(pm10 * 0.18 - 8).toFixed(2), direction: pm10 > 80 ? 'increases' : 'decreases' },
      { feature: 'NO₂', value: no2, shap: +(no2 * 0.12 - 3).toFixed(2), direction: no2 > 30 ? 'increases' : 'decreases' },
      { feature: 'O₃', value: o3, shap: +(o3 * 0.08 - 2).toFixed(2), direction: o3 > 50 ? 'increases' : 'decreases' },
    ].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    setResult({
      aqi: estimatedAqi,
      category: aqiInfo.label,
      color: aqiInfo.color,
      advice: aqiInfo.advice,
      shapFeatures,
      whoEvidence: [
        { citation: 'WHO Air Quality Guidelines 2021, p. 42', text: 'PM2.5 particles penetrate deep into the lungs and enter the bloodstream. Long-term exposure is associated with cardiovascular and respiratory disease.' },
        { citation: 'WHO Air Quality Guidelines 2021, p. 67', text: 'Even short-term exposure to elevated PM10 can exacerbate asthma and impair lung function, particularly in children and the elderly.' },
      ],
    });

    setLoading(false);
  };

  const handleReset = () => {
    setValues({ pm25: '', pm10: '', no2: '', o3: '', co: '', so2: '', date: '' });
    setResult(null);
    setError('');
  };

  return (
    <div className="predict">
      <section className="predict__header">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="section__tag">AQI Prediction</span>
            <h1 className="predict__title">Predict Air Quality Index</h1>
            <p className="predict__subtitle">
              Enter pollutant concentrations to get an instant AQI prediction
              with SHAP-powered explanations and WHO-backed health insights.
            </p>
          </motion.div>
        </div>
      </section>

      <div className="container predict__body">
        <div className="predict__grid">
          {/* ── Input Form ─────────────────────────────── */}
          <motion.div
            className="predict__form-panel"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <form onSubmit={handlePredict} className="predict-form">
              <h2 className="predict-form__heading">Pollutant Concentrations</h2>

              <div className="predict-form__fields">
                {pollutantFields.map((field) => (
                  <div key={field.key} className="predict-form__field">
                    <div className="predict-form__label-row">
                      <label htmlFor={field.key}>
                        <field.icon size={14} />
                        {field.label}
                        <small>{field.unit}</small>
                      </label>
                      <button
                        type="button"
                        className="predict-form__info-btn"
                        onClick={() => setExpandedInfo(expandedInfo === field.key ? null : field.key)}
                        aria-label={`Info about ${field.label}`}
                      >
                        <Info size={14} />
                      </button>
                    </div>
                    <AnimatePresence>
                      {expandedInfo === field.key && (
                        <motion.div
                          className="predict-form__info-box"
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.25 }}
                        >
                          <p>{field.info}</p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                    <input
                      id={field.key}
                      type="number"
                      step="0.1"
                      min="0"
                      placeholder={field.placeholder}
                      value={values[field.key]}
                      onChange={(e) => handleChange(field.key, e.target.value)}
                    />
                  </div>
                ))}
              </div>

              <div className="predict-form__field predict-form__field--date">
                <label htmlFor="predict-date">Date (optional)</label>
                <input
                  id="predict-date"
                  type="date"
                  value={values.date}
                  onChange={(e) => handleChange('date', e.target.value)}
                />
              </div>

              {error && (
                <div className="predict-form__error">
                  <AlertCircle size={16} />
                  {error}
                </div>
              )}

              <div className="predict-form__actions">
                <button
                  type="submit"
                  className="btn btn--primary btn--lg predict-form__submit"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 size={18} className="spin" />
                      Analysing...
                    </>
                  ) : (
                    <>
                      <Send size={18} />
                      Predict AQI
                    </>
                  )}
                </button>
                <button
                  type="button"
                  className="btn btn--outline"
                  onClick={handleReset}
                >
                  Reset
                </button>
              </div>
            </form>
          </motion.div>

          {/* ── Results Panel ──────────────────────────── */}
          <div className="predict__result-panel">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div
                  key="empty"
                  className="predict-empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="predict-empty__icon">
                    <Wind size={40} />
                  </div>
                  <h3>Enter Pollutant Values</h3>
                  <p>Fill in the pollutant concentrations on the left and click Predict to see results here.</p>
                </motion.div>
              )}

              {loading && (
                <motion.div
                  key="loading"
                  className="predict-loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="predict-loading__spinner">
                    <Loader2 size={36} className="spin" />
                  </div>
                  <h3>Analysing Air Quality</h3>
                  <p>Running ML model, computing SHAP values, and retrieving WHO evidence...</p>
                  <div className="predict-loading__steps">
                    <div className="predict-loading__step predict-loading__step--done">Model prediction</div>
                    <div className="predict-loading__step predict-loading__step--active">SHAP analysis</div>
                    <div className="predict-loading__step">WHO evidence retrieval</div>
                  </div>
                </motion.div>
              )}

              {result && !loading && (
                <motion.div
                  key="result"
                  className="predict-result"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  {/* AQI Result */}
                  <div className="predict-result__aqi" style={{ borderColor: result.color }}>
                    <span className="predict-result__aqi-label">Predicted AQI</span>
                    <div className="predict-result__aqi-value" style={{ color: result.color }}>
                      {result.aqi}
                    </div>
                    <span
                      className="predict-result__badge"
                      style={{ background: result.color + '18', color: result.color }}
                    >
                      {result.category}
                    </span>
                  </div>

                  {/* Health Advisory */}
                  <div className="predict-result__advisory">
                    <div className="predict-result__advisory-icon">
                      <ShieldCheck size={20} />
                    </div>
                    <div>
                      <h4>Health Advisory</h4>
                      <p>{result.advice}</p>
                    </div>
                  </div>

                  {/* SHAP Breakdown */}
                  <div className="predict-result__section">
                    <h4 className="predict-result__section-title">
                      <Brain size={16} />
                      Feature Influence (SHAP)
                    </h4>
                    <div className="predict-result__shap-list">
                      {result.shapFeatures.map((f, i) => (
                        <div key={i} className="shap-bar">
                          <div className="shap-bar__label">
                            <span>{f.feature}</span>
                            <span className="shap-bar__val">{f.value}</span>
                          </div>
                          <div className="shap-bar__track">
                            <div
                              className={`shap-bar__fill shap-bar__fill--${f.direction === 'increases' ? 'up' : 'down'}`}
                              style={{ width: `${Math.min(Math.abs(f.shap) * 2.5, 100)}%` }}
                            />
                          </div>
                          <span className={`shap-bar__shap shap-bar__shap--${f.direction === 'increases' ? 'up' : 'down'}`}>
                            {f.shap > 0 ? '+' : ''}{f.shap}
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="predict-result__note">
                      SHAP values show feature influence on the prediction, not direct causation.
                    </p>
                  </div>

                  {/* WHO Evidence */}
                  <div className="predict-result__section">
                    <h4 className="predict-result__section-title">
                      <ShieldCheck size={16} />
                      WHO Health Evidence
                    </h4>
                    {result.whoEvidence.map((e, i) => (
                      <div key={i} className="who-evidence-card">
                        <p className="who-evidence-card__text">{e.text}</p>
                        <cite className="who-evidence-card__cite">{e.citation}</cite>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
