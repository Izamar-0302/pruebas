import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { LeafScanner } from './components/LeafScanner';
import { DiagnosisHistory } from './components/DiagnosisHistory';
import { FeedbackGitHub } from './components/FeedbackGitHub';
import { CoffeeGuide } from './components/CoffeeGuide';
import { DiagnosisResult, CoffeeDiseaseId } from './types';
import { INITIAL_SAMPLE_DIAGNOSES } from './constants/coffeeDiseases';

const LOCAL_STORAGE_KEY = 'agrodetect_coffee_diagnoses_v1';

export default function App() {
  const [activeTab, setActiveTab] = useState<'scanner' | 'history' | 'feedback' | 'guide'>('scanner');
  const [diagnoses, setDiagnoses] = useState<DiagnosisResult[]>([]);
  const [feedbackRefDisease, setFeedbackRefDisease] = useState<CoffeeDiseaseId | null>(null);
  const [darkMode, setDarkMode] = useState<boolean>(false);

  const toggleDarkMode = () => setDarkMode(prev => !prev);

  // Load persistent history or fallback to sample seed diagnoses
  useEffect(() => {
    try {
      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setDiagnoses(parsed);
          return;
        }
      }
    } catch (e) {
      console.error('Error reading saved diagnoses from localStorage:', e);
    }
    // Set default initial seed
    setDiagnoses(INITIAL_SAMPLE_DIAGNOSES);
  }, []);

  // Save to localStorage on change
  const saveDiagnosesToStorage = (updated: DiagnosisResult[]) => {
    setDiagnoses(updated);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
    } catch (e) {
      console.error('Error saving diagnoses to localStorage:', e);
    }
  };

  const handleSaveDiagnosis = (newDiagnosis: DiagnosisResult) => {
    const exists = diagnoses.some(d => d.id === newDiagnosis.id);
    let updated: DiagnosisResult[];
    if (exists) {
      updated = diagnoses.map(d => d.id === newDiagnosis.id ? { ...newDiagnosis, isSaved: true } : d);
    } else {
      updated = [{ ...newDiagnosis, isSaved: true }, ...diagnoses];
    }
    saveDiagnosesToStorage(updated);
  };

  const handleDeleteDiagnosis = (id: string) => {
    const updated = diagnoses.filter(d => d.id !== id);
    saveDiagnosesToStorage(updated);
  };

  const handleClearAllHistory = () => {
    saveDiagnosesToStorage([]);
  };

  const handleOpenFeedbackWithDiagnosis = (diseaseId: CoffeeDiseaseId) => {
    setFeedbackRefDisease(diseaseId);
    setActiveTab('feedback');
  };

  return (
    <div className={`min-h-screen flex flex-col font-sans transition-colors duration-300 selection:bg-[#1E5631] selection:text-white ${
      darkMode ? 'bg-[#160E0A] text-[#F3EFEA]' : 'bg-[#FDFBF7] text-[#2C1A11]'
    }`}>
      
      {/* Navbar Header */}
      <Header 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        historyCount={diagnoses.length}
        feedbackCount={2}
        darkMode={darkMode}
        toggleDarkMode={toggleDarkMode}
      />

      {/* Main Tab Container */}
      <main className="flex-1 pb-16">
        {activeTab === 'scanner' && (
          <LeafScanner 
            onSaveDiagnosis={handleSaveDiagnosis}
            onOpenFeedbackWithDiagnosis={handleOpenFeedbackWithDiagnosis}
            darkMode={darkMode}
            recentDiagnoses={diagnoses}
            onNavigateTab={(tab) => setActiveTab(tab)}
          />
        )}

        {activeTab === 'history' && (
          <DiagnosisHistory 
            diagnoses={diagnoses}
            onDeleteDiagnosis={handleDeleteDiagnosis}
            onClearAll={handleClearAllHistory}
            darkMode={darkMode}
          />
        )}

        {activeTab === 'feedback' && (
          <FeedbackGitHub 
            initialRefDisease={feedbackRefDisease}
            darkMode={darkMode}
          />
        )}

        {activeTab === 'guide' && (
          <CoffeeGuide 
            darkMode={darkMode}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-[#2C1A11] text-[#D1D5DB] border-t border-[#D97706]/30 py-6 px-4 text-center text-xs space-y-2">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-serif font-bold text-white text-sm">
            <span>🌿 AgroDetect</span>
            <span className="text-[#D97706] font-sans text-xs">• Caficultura Honduras</span>
          </div>

          <p className="text-[#9CA3AF]">
            Herramienta de diagnóstico de IA y recomendación agronómica IHCAFE • MobileNetV2 & Gemini
          </p>

          <div className="flex items-center gap-4 text-[11px] text-[#FCD34D]">
            <a href="https://www.ihcafe.hn" target="_blank" rel="noopener noreferrer" className="hover:underline">
              Sitio Oficial IHCAFE
            </a>
            <span>•</span>
            <button onClick={() => setActiveTab('feedback')} className="hover:underline">
              Registrar Feedback GitHub
            </button>
          </div>
        </div>
      </footer>

    </div>
  );
}
