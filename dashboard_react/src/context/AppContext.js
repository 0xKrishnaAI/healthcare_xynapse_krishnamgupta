import React, { createContext, useContext, useReducer, useEffect } from 'react';

// Initial State
const initialState = {
    theme: localStorage.getItem('theme') || 'medical', // medical, light, dark
    user: { name: 'Dr. A. Smith', role: 'Chief Neurologist', avatar: '/assets/user.svg' },
    notifications: [
        { id: 1, text: 'New analysis ready: Patient 002_S_0295', type: 'success', time: '10m ago' },
        { id: 2, text: 'System Maintenance scheduled for 2 AM', type: 'info', time: '1h ago' }
    ],
    isSidebarOpen: true,
    // Dashboard Specific State
    isProcessing: false,
    processStep: 0, // 0-100
    processStatus: '',
    result: null, // { diagnosis: 'MCI', confidence: 0.85, ... }
};

// Reducer
const appReducer = (state, action) => {
    switch (action.type) {
        case 'SET_THEME':
            localStorage.setItem('theme', action.payload);
            return { ...state, theme: action.payload };
        case 'TOGGLE_SIDEBAR':
            return { ...state, isSidebarOpen: !state.isSidebarOpen };
        case 'SET_PROCESSING':
            return { ...state, isProcessing: action.payload };
        case 'UPDATE_PROGRESS':
            return {
                ...state,
                processStep: action.payload.progress,
                processStatus: action.payload.status
            };
        case 'SET_RESULT':
            return { ...state, result: action.payload, isProcessing: false };
        case 'RESET_DASHBOARD':
            return { ...state, processStep: 0, processStatus: '', result: null };
        case 'CLEAR_NOTIFICATIONS':
            return { ...state, notifications: [] };
        default:
            return state;
    }
};

// Context
const AppContext = createContext();

export const AppProvider = ({ children }) => {
    const [state, dispatch] = useReducer(appReducer, initialState);

    // Effect: Apply Theme to Body
    useEffect(() => {
        document.documentElement.className = state.theme;
        if (state.theme === 'dark') document.documentElement.classList.add('dark');
        else document.documentElement.classList.remove('dark');
    }, [state.theme]);

    return (
        <AppContext.Provider value={{ state, dispatch }}>
            {children}
        </AppContext.Provider>
    );
};

export const useApp = () => useContext(AppContext);
