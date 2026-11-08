import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Explainability from './pages/Explainability';
import Forecast from './pages/Forecast';
import Simulator from './pages/Simulator';
import Carbon from './pages/Carbon';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app grain-overlay">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/carbon" element={<Carbon />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
