import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { pageVariants } from '../utils/animations';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMoon, faSun, faLanguage, faShieldAlt, faDatabase } from '@fortawesome/free-solid-svg-icons';
import { useApp } from '../context/AppContext';

const Settings = () => {
    const { state, dispatch } = useApp();
    const [model2D, setModel2D] = useState(false);

    return (
        <motion.div variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">System Settings</h2>

            <div className="grid md:grid-cols-2 gap-6">

                {/* Theme Settings */}
                <div className="glass p-6">
                    <h3 className="text-lg font-bold text-gray-700 mb-4 flex items-center gap-2">
                        <FontAwesomeIcon icon={state.theme === 'dark' ? faMoon : faSun} /> Display
                    </h3>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Theme Mode</span>
                            <select
                                value={state.theme}
                                onChange={(e) => dispatch({ type: 'SET_THEME', payload: e.target.value })}
                                className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-1 text-sm outline-none focus:border-primary"
                            >
                                <option value="medical">Medical (Blue)</option>
                                <option value="light">Light</option>
                                <option value="dark">Dark Mode</option>
                            </select>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Language</span>
                            <div className="flex items-center gap-2">
                                <FontAwesomeIcon icon={faLanguage} className="text-gray-400" />
                                <span className="text-sm font-semibold">English (US)</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* AI Configuration */}
                <div className="glass p-6">
                    <h3 className="text-lg font-bold text-gray-700 mb-4 flex items-center gap-2">
                        <FontAwesomeIcon icon={faDatabase} /> AI Model Config
                    </h3>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Primary Model</span>
                            <span className="text-xs font-mono bg-blue-100 text-primary px-2 py-1 rounded">Simple3DCNN (v2.1)</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600">Use 2D Slice Fallback</span>
                            <div
                                onClick={() => setModel2D(!model2D)}
                                className={`w-10 h-5 rounded-full flex items-center p-1 cursor-pointer transition-colors ${model2D ? 'bg-primary' : 'bg-gray-300'}`}
                            >
                                <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform ${model2D ? 'translate-x-5' : ''}`} />
                            </div>
                        </div>
                        <p className="text-xs text-gray-400 mt-2">
                            Note: Switching models requires a restart of the Docker container.
                        </p>
                    </div>
                </div>
            </div>

            {/* Privacy Card */}
            <div className="glass p-6 mt-6 hover:border-primary/50 transition-colors cursor-pointer group">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-green-50 rounded-xl flex items-center justify-center text-success group-hover:scale-110 transition-transform">
                        <FontAwesomeIcon icon={faShieldAlt} size="lg" />
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-gray-700">Data Privacy & Security</h3>
                        <p className="text-sm text-gray-500">
                            Patient data is encrypted using AES-256. This system is HIPAA compliant.
                            <br />Audit logs are retained for 7 years.
                        </p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default Settings;
