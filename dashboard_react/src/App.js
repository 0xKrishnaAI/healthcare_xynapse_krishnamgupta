import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { AnimatePresence } from 'framer-motion';

// Layout Components
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// Views
import Dashboard from './components/Dashboard';
import Reports from './components/Reports';
import Records from './components/Records';
import SOS from './components/SOS';
import Settings from './components/Settings';
import Help from './components/Help';

const AnimatedRoutes = () => {
    const location = useLocation();

    return (
        <AnimatePresence mode="wait">
            <Routes location={location} key={location.pathname}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/records" element={<Records />} />
                <Route path="/sos" element={<SOS />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/help" element={<Help />} />
            </Routes>
        </AnimatePresence>
    );
};

function App() {
    return (
        <AppProvider>
            <Router>
                <div className="flex h-screen w-full bg-slate-50 overflow-hidden relative font-sans text-gray-800">

                    {/* Background Blobs (Decoration) */}
                    <div className="fixed top-[-10%] left-[-10%] w-[500px] h-[500px] bg-blue-200/30 rounded-full blur-[100px] pointer-events-none animate-blob"></div>
                    <div className="fixed bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-purple-200/30 rounded-full blur-[120px] pointer-events-none animate-blob animation-delay-2000"></div>

                    <Sidebar />

                    <div className="flex-1 flex flex-col md:ml-72 transition-all duration-300 h-full relative z-10">
                        <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-6 scroll-smooth">
                            <Header />

                            {/* Main Content Area */}
                            <main className="max-w-7xl mx-auto pb-10 min-h-[calc(100vh-140px)]">
                                <AnimatedRoutes />
                            </main>
                        </div>
                    </div>

                </div>
            </Router>
        </AppProvider>
    );
}

export default App;
