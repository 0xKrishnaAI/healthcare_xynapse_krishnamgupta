import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faHome, faHistory, faLifeRing, faCog, faExclamationTriangle,
    faChevronRight, faSignOutAlt, faChartLine
} from '@fortawesome/free-solid-svg-icons';
import { motion } from 'framer-motion';

const Sidebar = () => {
    const navigate = useNavigate();

    const links = [
        { path: '/', label: 'Dashboard', icon: faHome },
        { path: '/reports', label: 'Reports', icon: faChartLine },
        { path: '/records', label: 'Patient Records', icon: faHistory },
        { path: '/sos', label: 'Emergency SOS', icon: faExclamationTriangle, color: 'text-red-500' },
        { path: '/settings', label: 'Settings', icon: faCog },
        { path: '/help', label: 'Help & Support', icon: faLifeRing },
    ];

    const handleLogout = () => {
        if (window.confirm('Are you sure you want to logout?')) {
            // Mock logout
            navigate('/login');
        }
    };

    return (
        <aside className="fixed left-0 top-0 h-full w-72 bg-white/80 backdrop-blur-xl border-r border-white/40 shadow-2xl z-50 hidden md:flex flex-col justify-between transition-all duration-300">

            {/* Logo Area */}
            <div className="p-8 pb-4">
                <div className="flex items-center gap-3 mb-8">
                    <div className="w-10 h-10 bg-gradient-to-tr from-primary to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                        <FontAwesomeIcon icon={require('@fortawesome/free-solid-svg-icons').faBrain} className="text-white text-lg" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-gray-800 tracking-tight">NeuroDx</h1>
                        <p className="text-[10px] text-gray-500 font-bold tracking-widest uppercase">BrainScan AI</p>
                    </div>
                </div>

                {/* Navigation */}
                <nav className="space-y-4">
                    {links.map((link) => {
                        return (
                            <NavLink
                                key={link.path}
                                to={link.path}
                                className="relative block group outline-none"
                            >
                                {({ isActive }) => (
                                    <div className={`
                                        relative flex items-center gap-4 px-4 py-4 rounded-xl transition-all duration-300
                                        ${isActive
                                            ? 'bg-primary text-white shadow-lg shadow-blue-500/30 translate-x-2'
                                            : 'text-gray-600 hover:bg-gray-50 hover:translate-x-1'
                                        }
                                    `}>
                                        {/* Icon */}
                                        <div className={`
                                            w-6 flex justify-center transition-colors duration-300
                                            ${isActive ? 'text-white' : link.color || 'text-gray-400 group-hover:text-primary'}
                                        `}>
                                            <FontAwesomeIcon icon={link.icon} className={isActive ? 'animate-pulse' : ''} />
                                        </div>

                                        {/* Label */}
                                        <span className={`font-medium tracking-wide ${isActive ? 'font-semibold' : ''} `}>
                                            {link.label}
                                        </span>

                                        {/* Active Indicator (Right Chevron) */}
                                        {isActive && (
                                            <motion.div
                                                layoutId="activeIndicator"
                                                className="absolute right-4 text-white/50 text-xs"
                                            >
                                                <FontAwesomeIcon icon={faChevronRight} />
                                            </motion.div>
                                        )}

                                        {/* Hover Tooltip (if collapsed - implementing simplistic logic for future resize) */}
                                    </div>
                                )}
                            </NavLink>
                        );
                    })}
                </nav>
            </div>

            {/* Footer / User Profile */}
            <div className="p-6 border-t border-gray-100 bg-gray-50/50">
                <div className="flex items-center gap-3 p-3 rounded-xl hover:bg-white hover:shadow-md transition-all cursor-pointer group">
                    <img
                        src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix"
                        alt="User"
                        className="w-10 h-10 rounded-full border-2 border-white shadow-sm"
                    />
                    <div className="flex-1 overflow-hidden">
                        <p className="text-sm font-bold text-gray-800 truncate">Dr. Alex Smith</p>
                        <p className="text-xs text-gray-500 truncate">Neurologist</p>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="text-gray-400 hover:text-red-500 transition-colors p-2"
                        title="Logout"
                    >
                        <FontAwesomeIcon icon={faSignOutAlt} />
                    </button>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
