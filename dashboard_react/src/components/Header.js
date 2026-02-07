import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faBell, faSearch, faUserCircle } from '@fortawesome/free-solid-svg-icons';
import { useApp } from '../context/AppContext';
import { motion, AnimatePresence } from 'framer-motion';

const Header = () => {
    const { state, dispatch } = useApp();
    const [showNotifs, setShowNotifs] = useState(false);

    return (
        <header className="sticky top-0 z-40 w-full mb-6">
            <div className="glass px-6 py-3 flex items-center justify-between">

                {/* Left: Mobile Toggle & Search */}
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
                        className="md:hidden btn-icon text-gray-600"
                    >
                        <FontAwesomeIcon icon={faBars} size="lg" />
                    </button>

                    <div className="hidden md:flex items-center bg-gray-100/50 rounded-full px-4 py-2 border border-transparent focus-within:border-primary/50 focus-within:bg-white transition-all">
                        <FontAwesomeIcon icon={faSearch} className="text-gray-400 mr-2" />
                        <input
                            type="text"
                            placeholder="Search Patient ID..."
                            className="bg-transparent border-none outline-none text-sm w-48 text-gray-700 placeholder-gray-400"
                        />
                    </div>
                </div>

                {/* Right: Actions */}
                <div className="flex items-center gap-3">

                    {/* Status Badge */}
                    <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-green-50 text-success text-xs font-bold rounded-full border border-green-100">
                        <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
                        SYSTEM ONLINE
                    </div>

                    {/* Notifications */}
                    <div className="relative">
                        <button
                            onClick={() => setShowNotifs(!showNotifs)}
                            className="relative btn-icon"
                        >
                            <FontAwesomeIcon icon={faBell} className="text-gray-600" />
                            {state.notifications.length > 0 && (
                                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-danger rounded-full border-2 border-white"></span>
                            )}
                        </button>

                        {/* Dropdown */}
                        <AnimatePresence>
                            {showNotifs && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                                    className="absolute right-0 mt-3 w-80 bg-white/90 backdrop-blur-xl border border-gray-100 shadow-2xl rounded-xl p-2 z-50 origin-top-right"
                                >
                                    <div className="px-3 py-2 border-b border-gray-100 flex justify-between items-center">
                                        <h3 className="text-sm font-bold text-gray-700">Notifications</h3>
                                        <button onClick={() => dispatch({ type: 'CLEAR_NOTIFICATIONS' })} className="text-xs text-primary hover:underline">Clear all</button>
                                    </div>
                                    <ul className="max-h-64 overflow-y-auto">
                                        {state.notifications.length === 0 ? (
                                            <li className="p-4 text-center text-xs text-gray-400">No new notifications.</li>
                                        ) : (
                                            state.notifications.map(n => (
                                                <li key={n.id} className="p-3 hover:bg-gray-50 rounded-lg transition-colors cursor-pointer border-b border-gray-50 last:border-0">
                                                    <p className="text-xs font-medium text-gray-700">{n.text}</p>
                                                    <p className="text-[10px] text-gray-400 mt-1">{n.time}</p>
                                                </li>
                                            ))
                                        )}
                                    </ul>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Profile */}
                    <button className="flex items-center gap-2 btn-icon">
                        <FontAwesomeIcon icon={faUserCircle} className="text-primary text-xl" />
                    </button>
                </div>
            </div>
        </header>
    );
};

export default Header;
