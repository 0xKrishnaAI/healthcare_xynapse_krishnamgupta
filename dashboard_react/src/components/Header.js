import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBell, faSearch, faUserCircle, faCog, faSignOutAlt, faTimes } from '@fortawesome/free-solid-svg-icons';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../context/AppContext';

const Header = () => {
    const location = useLocation();
    const { state, dispatch } = useApp();
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);

    // Get page title based on path
    const getPageTitle = (path) => {
        switch (path) {
            case '/': return 'Dashboard';
            case '/reports': return 'Reports';
            case '/records': return 'Patient Records';
            case '/sos': return 'Emergency SOS';
            case '/settings': return 'Settings';
            case '/help': return 'Help & Support';
            default: return 'NeuroDx';
        }
    };

    const handleClearNotifications = () => {
        dispatch({ type: 'CLEAR_NOTIFICATIONS' });
        setShowNotifications(false);
    };

    return (
        <header className="sticky top-0 z-40 bg-white/50 backdrop-blur-xl border-b border-white/40 mb-6 rounded-b-2xl md:rounded-b-none mx-[-16px] md:mx-0 px-4 md:px-8 py-4 transition-all duration-300">
            <div className="flex items-center justify-between">

                {/* Mobile Menu Toggle & Title */}
                <div className="flex items-center gap-4">
                    <button className="md:hidden text-gray-600 focus:outline-none">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
                    </button>
                    <h2 className="text-xl font-bold text-gray-800 tracking-tight">
                        {getPageTitle(location.pathname)}
                    </h2>
                </div>

                {/* Right Actions */}
                <div className="flex items-center gap-4 md:gap-6">

                    {/* Search (Tablet/Desktop) */}
                    <div className="hidden md:flex items-center bg-white/60 px-3 py-2 rounded-xl border border-white/40 focus-within:ring-2 ring-primary/20 transition-all shadow-sm w-64">
                        <FontAwesomeIcon icon={faSearch} className="text-gray-400 mr-2" />
                        <input
                            type="text"
                            placeholder="Search patients, reports..."
                            className="bg-transparent border-none outline-none text-sm w-full placeholder-gray-400 text-gray-700"
                        />
                    </div>

                    {/* Notifications */}
                    <div className="relative">
                        <button
                            className="relative p-2 text-gray-500 hover:text-primary hover:bg-white rounded-full transition-all duration-200 outline-none"
                            onClick={() => setShowNotifications(!showNotifications)}
                        >
                            <FontAwesomeIcon icon={faBell} className="text-lg" />
                            {state.notifications.length > 0 && (
                                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 border-2 border-white rounded-full animate-pulse"></span>
                            )}
                        </button>

                        <AnimatePresence>
                            {showNotifications && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setShowNotifications(false)}></div>
                                    <motion.div
                                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                                        transition={{ duration: 0.2 }}
                                        className="absolute right-0 mt-3 w-80 bg-white/90 backdrop-blur-xl border border-white/60 rounded-2xl shadow-xl z-50 overflow-hidden"
                                    >
                                        <div className="p-4 border-b border-gray-100 flex justify-between items-center">
                                            <h3 className="font-bold text-gray-800">Notifications</h3>
                                            <button
                                                onClick={handleClearNotifications}
                                                className="text-xs text-primary hover:text-primary-dark font-medium"
                                            >
                                                Mark all read
                                            </button>
                                        </div>
                                        <div className="max-h-[300px] overflow-y-auto">
                                            {state.notifications.length === 0 ? (
                                                <div className="p-8 text-center text-gray-500 text-sm">
                                                    No new notifications
                                                </div>
                                            ) : (
                                                <ul className="divide-y divide-gray-50">
                                                    {state.notifications.map((notif, idx) => (
                                                        <li key={idx} className="p-4 hover:bg-blue-50/50 transition-colors flex gap-3">
                                                            <div className={`w-2 h-2 mt-1.5 rounded-full shrink-0 ${notif.type === 'danger' ? 'bg-red-500' : 'bg-green-500'}`}></div>
                                                            <div>
                                                                <p className="text-sm text-gray-700">{notif.text}</p>
                                                                <p className="text-xs text-gray-400 mt-1">Just now</p>
                                                            </div>
                                                        </li>
                                                    ))}
                                                </ul>
                                            )}
                                        </div>
                                    </motion.div>
                                </>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Profile */}
                    <div className="relative">
                        <button
                            className="flex items-center gap-2 hover:bg-white/60 p-1.5 pr-3 rounded-full transition-all border border-transparent hover:border-white/40 hover:shadow-sm"
                            onClick={() => setShowProfile(true)}
                        >
                            <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix" alt="Profile" className="w-8 h-8 rounded-full border border-white shadow-sm" />
                            <span className="text-sm font-semibold text-gray-700 hidden md:inline">Dr. Smith</span>
                        </button>
                    </div>

                </div>
            </div>

            {/* Profile Modal */}
            <AnimatePresence>
                {showProfile && (
                    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
                            onClick={() => setShowProfile(false)}
                        />
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.95, opacity: 0 }}
                            className="bg-white rounded-3xl shadow-2xl w-full max-w-md relative z-10 overflow-hidden"
                        >
                            <div className="h-24 bg-gradient-to-r from-primary to-blue-600"></div>
                            <button
                                onClick={() => setShowProfile(false)}
                                className="absolute top-4 right-4 text-white/80 hover:text-white bg-black/20 hover:bg-black/30 w-8 h-8 rounded-full flex items-center justify-center transition-colors"
                            >
                                <FontAwesomeIcon icon={faTimes} />
                            </button>

                            <div className="px-6 pb-6 text-center -mt-12">
                                <img
                                    src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix"
                                    alt="Profile"
                                    className="w-24 h-24 rounded-full border-4 border-white shadow-lg mx-auto bg-white"
                                />
                                <h3 className="text-xl font-bold text-gray-800 mt-3">Dr. Alex Smith</h3>
                                <p className="text-gray-500 text-sm">Senior Neurologist • NeuroDx Admin</p>

                                <div className="mt-6 flex flex-col gap-2">
                                    <button className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-gray-50 text-gray-700 transition-colors font-medium border border-gray-100">
                                        <FontAwesomeIcon icon={faUserCircle} className="text-primary" /> Edit Profile
                                    </button>
                                    <button className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-gray-50 text-gray-700 transition-colors font-medium border border-gray-100">
                                        <FontAwesomeIcon icon={faCog} className="text-gray-500" /> Account Settings
                                    </button>
                                    <button className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-red-50 text-red-600 transition-colors font-medium border border-red-100 mt-2">
                                        <FontAwesomeIcon icon={faSignOutAlt} /> Sign Out
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </header>
    );
};

export default Header;
