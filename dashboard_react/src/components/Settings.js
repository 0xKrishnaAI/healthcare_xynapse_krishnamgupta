import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { pageVariants } from '../utils/animations';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSun, faBell, faDatabase, faUserShield, faToggleOn, faToggleOff } from '@fortawesome/free-solid-svg-icons';

const Settings = () => {
    // const { dispatch } = useApp(); // Unused
    const [theme, setTheme] = useState('light');
    const [notifications, setNotifications] = useState(true);
    const [language, setLanguage] = useState('en');
    const [twoFactor, setTwoFactor] = useState(false);

    const toggleTheme = () => {
        setTheme(theme === 'light' ? 'dark' : 'light');
        // In a real app, dispatch UPDATE_THEME
    };

    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="max-w-4xl mx-auto"
        >
            <h2 className="text-2xl font-bold text-gray-800 mb-6">System Settings</h2>

            <div className="grid md:grid-cols-2 gap-6">

                {/* Visual Preferences */}
                <div className="glass p-6">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <FontAwesomeIcon icon={faSun} className="text-orange-400" />
                        Appearance
                    </h3>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="font-medium text-gray-700">Dark Mode</p>
                                <p className="text-xs text-gray-500">Switch between light and dark themes</p>
                            </div>
                            <button
                                onClick={toggleTheme}
                                className={`text-2xl transition-colors ${theme === 'dark' ? 'text-primary' : 'text-gray-300'}`}
                            >
                                <FontAwesomeIcon icon={theme === 'dark' ? faToggleOn : faToggleOff} />
                            </button>
                        </div>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="font-medium text-gray-700">Language</p>
                                <p className="text-xs text-gray-500">Select system language</p>
                            </div>
                            <select
                                value={language}
                                onChange={(e) => setLanguage(e.target.value)}
                                className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-sm outline-none focus:ring-2 focus:ring-primary/20"
                            >
                                <option value="en">English (US)</option>
                                <option value="es">Español</option>
                                <option value="fr">Français</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Notifications */}
                <div className="glass p-6">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <FontAwesomeIcon icon={faBell} className="text-primary" />
                        Notifications
                    </h3>
                    <div className="space-y-4">
                        <SettingToggle
                            label="Email Alerts"
                            desc="Receive analysis results via email"
                            checked={notifications}
                            onChange={() => setNotifications(!notifications)}
                        />
                        <SettingToggle
                            label="System Sounds"
                            desc="Play sound on task completion"
                            checked={true}
                            onChange={() => { }}
                        />
                    </div>
                </div>

                {/* Privacy & Security */}
                <div className="glass p-6">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <FontAwesomeIcon icon={faUserShield} className="text-purple-500" />
                        Privacy & Security
                    </h3>
                    <div className="space-y-4">
                        <SettingToggle
                            label="Two-Factor Auth"
                            desc="Secure your account with 2FA"
                            checked={twoFactor}
                            onChange={() => setTwoFactor(!twoFactor)}
                        />
                        <div className="pt-2">
                            <button className="text-primary hover:underline text-sm font-medium">Change Password</button>
                        </div>
                    </div>
                </div>

                {/* Data Management */}
                <div className="glass p-6">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <FontAwesomeIcon icon={faDatabase} className="text-emerald-500" />
                        Data Management
                    </h3>
                    <div className="space-y-3">
                        <button className="w-full text-left px-4 py-3 bg-gray-50 hover:bg-gray-100 rounded-xl transition-colors text-sm font-medium text-gray-700 flex justify-between items-center">
                            Export All Patient Data
                            <FontAwesomeIcon icon={require('@fortawesome/free-solid-svg-icons').faArrowRight} className="text-xs text-gray-400" />
                        </button>
                        <button className="w-full text-left px-4 py-3 bg-red-50 hover:bg-red-100 rounded-xl transition-colors text-sm font-medium text-red-600 flex justify-between items-center">
                            Clear Local Cache
                            <FontAwesomeIcon icon={require('@fortawesome/free-solid-svg-icons').faTrash} className="text-xs text-red-400" />
                        </button>
                    </div>
                </div>

            </div>
        </motion.div>
    );
};

const SettingToggle = ({ label, desc, checked, onChange }) => (
    <div className="flex items-center justify-between">
        <div>
            <p className="font-medium text-gray-700">{label}</p>
            <p className="text-xs text-gray-500">{desc}</p>
        </div>
        <button
            onClick={onChange}
            className={`text-2xl transition-colors ${checked ? 'text-primary' : 'text-gray-300'}`}
        >
            <FontAwesomeIcon icon={checked ? faToggleOn : faToggleOff} />
        </button>
    </div>
);

export default Settings;
