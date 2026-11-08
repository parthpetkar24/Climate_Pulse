import { useState, useEffect, useMemo } from 'react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Wind, Droplets, CloudSun, Thermometer,
  TrendingUp, TrendingDown, AlertTriangle, MapPin
} from 'lucide-react';
import './Dashboard.css';

/* ── Initial Fallback Data ─────────────────────────────────────── */
const initialWeeklyData = [
  { day: 'Mon', aqi: 98, pm25: 42, pm10: 88, no2: 28 },
  { day: 'Tue', aqi: 112, pm25: 55, pm10: 96, no2: 32 },
  { day: 'Wed', aqi: 145, pm25: 72, pm10: 130, no2: 40 },
  { day: 'Thu', aqi: 127, pm25: 64, pm10: 112, no2: 35 },
  { day: 'Fri', aqi: 135, pm25: 68, pm10: 118, no2: 38 },
  { day: 'Sat', aqi: 89, pm25: 38, pm10: 75, no2: 22 },
  { day: 'Sun', aqi: 76, pm25: 30, pm10: 60, no2: 18 },
];

const initialMonthlyTrend = [
  { month: 'Jun', aqi: 68 },
  { month: 'Jul', aqi: 72 },
  { month: 'Aug', aqi: 85 },
  { month: 'Sep', aqi: 110 },
  { month: 'Oct', aqi: 155 },
  { month: 'Nov', aqi: 210 },
  { month: 'Dec', aqi: 265 },
  { month: 'Jan', aqi: 245 },
  { month: 'Feb', aqi: 195 },
  { month: 'Mar', aqi: 140 },
  { month: 'Apr', aqi: 95 },
  { month: 'May', aqi: 72 },
];

const initialCategoryDistribution = [
  { name: 'Good', value: 85, color: '#5e8a5e' },
  { name: 'Satisfactory', value: 120, color: '#7da37d' },
  { name: 'Moderate', value: 95, color: '#c4a65a' },
  { name: 'Poor', value: 40, color: '#c48a5a' },
  { name: 'Very Poor', value: 18, color: '#c47a7a' },
  { name: 'Severe', value: 7, color: '#8a3d3d' },
];

const initialPollutants = [
  { name: 'PM2.5', value: 68.4, unit: 'µg/m³', limit: 60, icon: Droplets },
  { name: 'PM10', value: 112.0, unit: 'µg/m³', limit: 100, icon: Wind },
  { name: 'NO₂', value: 34.2, unit: 'ppb', limit: 80, icon: CloudSun },
  { name: 'O₃', value: 28.7, unit: 'ppb', limit: 100, icon: Thermometer },
  { name: 'CO', value: 1.2, unit: 'mg/m³', limit: 4, icon: CloudSun },
  { name: 'SO₂', value: 12.5, unit: 'ppb', limit: 80, icon: Droplets },
];

function getAqiInfo(val) {
  if (val <= 50) return { label: 'Good', color: '#5e8a5e', bg: 'rgba(94,138,94,0.1)' };
  if (val <= 100) return { label: 'Satisfactory', color: '#7da37d', bg: 'rgba(125,163,125,0.1)' };
  if (val <= 200) return { label: 'Moderate', color: '#c4a65a', bg: 'rgba(196,166,90,0.1)' };
  if (val <= 300) return { label: 'Poor', color: '#c48a5a', bg: 'rgba(196,138,90,0.1)' };
  if (val <= 400) return { label: 'Very Poor', color: '#c47a7a', bg: 'rgba(196,122,122,0.1)' };
  return { label: 'Severe', color: '#8a3d3d', bg: 'rgba(138,61,61,0.1)' };
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="dash-tooltip">
      <p className="dash-tooltip__label">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color }}>
          {entry.name}: <strong>{entry.value}</strong>
        </p>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const [selectedCity] = useState('Pune');
  const [currentAqi, setCurrentAqi] = useState(127);
  const [weeklyData, setWeeklyData] = useState(initialWeeklyData);
  const [monthlyTrend, setMonthlyTrend] = useState(initialMonthlyTrend);
  const [categoryDistribution, setCategoryDistribution] = useState(initialCategoryDistribution);
  const [pollutantBreakdown, setPollutantBreakdown] = useState(initialPollutants);
  const [weeklyAvg, setWeeklyAvg] = useState('112');
  const [bestDay, setBestDay] = useState('Sunday');

  useEffect(() => {
    fetch('/api/dashboard/overview')
      .then((res) => (res.ok ? res.json() : null))
      .then((res) => {
        if (res) {
          setCurrentAqi(res.current_aqi);
          setWeeklyAvg(String(res.weekly_avg));
          setBestDay(res.best_day);
          if (res.weekly_trend?.length) setWeeklyData(res.weekly_trend);
          if (res.monthly_trend?.length) setMonthlyTrend(res.monthly_trend);
          if (res.category_distribution?.length) setCategoryDistribution(res.category_distribution);
          if (res.pollutant_breakdown?.length) {
            setPollutantBreakdown(
              res.pollutant_breakdown.map((p) => ({
                ...p,
                icon:
                  p.name.includes('2.5') || p.name.includes('SO')
                    ? Droplets
                    : p.name.includes('10')
                    ? Wind
                    : p.name.includes('O₃')
                    ? Thermometer
                    : CloudSun,
              }))
            );
          }
        }
      })
      .catch((e) => console.log('Dashboard using local cache:', e));
  }, []);

  const aqiInfo = getAqiInfo(currentAqi);

  const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5 },
  };

  return (
    <div className="dashboard">
      <section className="dashboard__header">
        <div className="container">
          <motion.div {...fadeUp}>
            <div className="dashboard__top-bar">
              <div>
                <span className="section__tag">Live Dashboard</span>
                <h1 className="dashboard__title">Air Quality Overview</h1>
                <p className="dashboard__subtitle">
                  Real-time monitoring and historical analysis for {selectedCity}
                </p>
              </div>
              <div className="dashboard__location">
                <MapPin size={16} />
                <span>{selectedCity}, Maharashtra</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <div className="container dashboard__body">
        {/* ── AQI Summary Cards ─────────────────────────── */}
        <div className="dash-summary">
          <motion.div className="dash-aqi-main" {...fadeUp} transition={{ delay: 0.1 }}>
            <div className="dash-aqi-main__inner" style={{ borderColor: aqiInfo.color }}>
              <span className="dash-aqi-main__label">Current AQI</span>
              <div className="dash-aqi-main__value" style={{ color: aqiInfo.color }}>
                {currentAqi}
              </div>
              <span
                className="dash-aqi-main__badge"
                style={{ background: aqiInfo.bg, color: aqiInfo.color }}
              >
                {aqiInfo.label}
              </span>
              <div className="dash-aqi-main__scale">
                <div className="dash-aqi-main__scale-bar">
                  <div
                    className="dash-aqi-main__scale-indicator"
                    style={{ left: `${Math.min((currentAqi / 500) * 100, 100)}%` }}
                  />
                </div>
                <div className="dash-aqi-main__scale-labels">
                  <span>0</span><span>100</span><span>200</span><span>300</span><span>400</span><span>500</span>
                </div>
              </div>
            </div>
          </motion.div>

          {[
            { label: 'Weekly Avg', value: weeklyAvg, change: '+8%', trend: 'up', icon: TrendingUp },
            { label: 'Best Day', value: bestDay, change: 'Optimal AQI', trend: 'down', icon: TrendingDown },
            { label: 'Alerts', value: '3', change: 'This week', trend: 'up', icon: AlertTriangle },
          ].map((card, i) => (
            <motion.div
              key={i}
              className="dash-stat-card"
              {...fadeUp}
              transition={{ delay: 0.2 + i * 0.1 }}
            >
              <div className="dash-stat-card__icon">
                <card.icon size={20} />
              </div>
              <span className="dash-stat-card__label">{card.label}</span>
              <span className="dash-stat-card__value">{card.value}</span>
              <span className={`dash-stat-card__change dash-stat-card__change--${card.trend}`}>
                {card.change}
              </span>
            </motion.div>
          ))}
        </div>

        {/* ── Charts Row 1 ─────────────────────────────── */}
        <div className="dash-charts-row">
          <motion.div
            className="dash-chart-card dash-chart-card--wide"
            {...fadeUp}
            transition={{ delay: 0.3 }}
          >
            <h3 className="dash-chart-card__title">Weekly AQI Trend</h3>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={weeklyData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <defs>
                  <linearGradient id="aqiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--sage-500)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="var(--sage-500)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
                <XAxis dataKey="day" tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
                <YAxis tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="aqi"
                  stroke="var(--sage-500)"
                  strokeWidth={2.5}
                  fill="url(#aqiGrad)"
                  name="AQI"
                  dot={{ fill: 'var(--sage-500)', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          <motion.div
            className="dash-chart-card"
            {...fadeUp}
            transition={{ delay: 0.4 }}
          >
            <h3 className="dash-chart-card__title">AQI Category Distribution</h3>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={categoryDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {categoryDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{ fontSize: '0.75rem' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* ── Pollutant Breakdown ──────────────────────── */}
        <motion.div
          className="dash-pollutants-section"
          {...fadeUp}
          transition={{ delay: 0.5 }}
        >
          <h3 className="dash-section-title">Pollutant Concentrations</h3>
          <div className="dash-pollutant-grid">
            {pollutantBreakdown.map((p, i) => {
              const pct = Math.min((p.value / p.limit) * 100, 100);
              const over = p.value > p.limit;
              return (
                <div key={i} className={`dash-pollutant-card ${over ? 'dash-pollutant-card--over' : ''}`}>
                  <div className="dash-pollutant-card__header">
                    <div className="dash-pollutant-card__icon">
                      <p.icon size={16} />
                    </div>
                    <span className="dash-pollutant-card__name">{p.name}</span>
                  </div>
                  <div className="dash-pollutant-card__value">
                    {p.value} <small>{p.unit}</small>
                  </div>
                  <div className="dash-pollutant-card__bar-wrap">
                    <div
                      className="dash-pollutant-card__bar"
                      style={{
                        width: `${pct}%`,
                        background: over ? 'var(--accent-rose)' : 'var(--sage-500)',
                      }}
                    />
                  </div>
                  <div className="dash-pollutant-card__limit">
                    Limit: {p.limit} {p.unit}
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* ── Charts Row 2 ─────────────────────────────── */}
        <div className="dash-charts-row">
          <motion.div
            className="dash-chart-card dash-chart-card--full"
            {...fadeUp}
            transition={{ delay: 0.6 }}
          >
            <h3 className="dash-chart-card__title">Monthly AQI Trend (12 Months)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={monthlyTrend} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
                <XAxis dataKey="month" tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
                <YAxis tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="aqi" name="AQI" radius={[4, 4, 0, 0]}>
                  {monthlyTrend.map((entry, i) => (
                    <Cell key={i} fill={getAqiInfo(entry.aqi).color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* ── Weekly Pollutant Comparison ──────────────── */}
        <motion.div
          className="dash-chart-card dash-chart-card--full"
          style={{ marginBottom: '2rem' }}
          {...fadeUp}
          transition={{ delay: 0.7 }}
        >
          <h3 className="dash-chart-card__title">Weekly Pollutant Comparison</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={weeklyData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
              <XAxis dataKey="day" tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-tertiary)' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: '0.8rem' }} />
              <Bar dataKey="pm25" name="PM2.5" fill="var(--sage-500)" radius={[3, 3, 0, 0]} />
              <Bar dataKey="pm10" name="PM10" fill="var(--accent-teal)" radius={[3, 3, 0, 0]} />
              <Bar dataKey="no2" name="NO₂" fill="var(--accent-amber)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>
    </div>
  );
}
