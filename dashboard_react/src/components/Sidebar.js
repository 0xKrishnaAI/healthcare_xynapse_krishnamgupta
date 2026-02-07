import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faChartLine,
    faFileMedical,
    faFolderOpen,
    faTriangleExclamation,
    faGear,
    faBrain,
    faCircleQuestion
} from '@fortawesome/free-solid-svg-icons';
import { useApp } from '../context/AppContext';
import { motion, AnimatePresence } from 'framer-motion';

const Sidebar = () => {
    const { state } = useApp();
    const location = useLocation();

    const navItems = [
        { path: '/', icon: faChartLine, label: 'Dashboard' },
        { path: '/reports', icon: faFileMedical, label: 'Reports' },
        { path: '/records', icon: faFolderOpen, label: 'Records' },
        { path: '/sos', icon: faTriangleExclamation, label: 'SOS Protocol', isAlert: true },
        { path: '/settings', icon: faGear, label: 'Settings' },
        { path: '/help', icon: faCircleQuestion, label: 'Help' },
    ];

    // Mobile/Desktop Conditional Rendering logic handled via CSS classes usually,
    // but Framer Motion handles the slide-in.

    return (
        <motion.aside
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ type: "spring", stiffness: 100 }}
            className={`
                fixed top-0 left-0 h-full w-72 p-4 z-50
                hidden md:flex flex-col
                glass
                ${!state.isSidebarOpen ? '-translate-x-full' : ''} 
                transition-transform duration-300
            `}
        >
            {/* Logo Area */}
            <div className="flex items-center gap-3 px-4 py-6 mb-4">
                <div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center text-primary">
                    <FontAwesomeIcon icon={faBrain} className="text-xl animate-pulse-slow" />
                </div>
                <div>
                    <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-dark to-primary">NeuroDx</h1>
                    <p className="text-xs text-gray-400 font-medium">BrainScan AI v2.0</p>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-2">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) => `
                            flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group
                            ${isActive
                                ? 'bg-primary text-white shadow-lg shadow-blue-500/20 translate-x-1'
                                : 'text-gray-500 hover:bg-white/50 hover:text-primary hover:translate-x-1'}
                            ${item.isAlert ? 'mt-8 text-danger hover:bg-red-50 hover:text-danger' : ''}
                        `}
                    >
                        <FontAwesomeIcon icon={item.icon} className={`w-5 ${item.isAlert ? 'animate-bounce' : ''}`} />
                        <span className="font-medium">{item.label}</span>
                        {/* Active Indicator Dot */}
                        {location.pathname === item.path && !item.isAlert && (
                            <motion.div
                                layoutId="active-nav"
                                className="ml-auto w-1.5 h-1.5 rounded-full bg-white"
                            />
                        )}
                    </NavLink>
                ))}
            </nav>

            {/* User Profile Footer */}
            <div className="mt-auto pt-4 border-t border-gray-200/50">
                <div className="flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-white/50 cursor-pointer transition-colors">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-primary to-purple-500 p-[2px]">
                        <div className="w-full h-full rounded-full bg-white flex items-center justify-center overflow-hidden">
                            <img src={`https://ui-avatars.com/api/?name=${state.user.name}&background=random`} alt="User" />
                        </div>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold truncate text-gray-700">{state.user.name}</p>
                        <p className="text-xs text-gray-500 truncate">{state.user.role}</p>
                    </div>
                    <FontAwesomeIcon icon={faGear} className="text-gray-400 w-3" />
                </div>
            </div>
        </motion.aside>
    );
};

export default Sidebar;
