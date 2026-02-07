import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { pageVariants } from '../utils/animations';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBell, faPhoneAlt, faUserMd, faAmbulance } from '@fortawesome/free-solid-svg-icons';

const SOS = () => {
    const [isTriggered, setIsTriggered] = useState(false);

    const handleSOS = () => {
        setIsTriggered(true);
        // In real app, create API call
        setTimeout(() => alert("Emergency Services Notified (Simulation)"), 1000);
    };

    return (
        <motion.div
            variants={pageVariants}
            initial="initial" animate="animate" exit="exit"
            className="flex flex-col items-center justify-center min-h-[70vh] text-center"
        >
            <div className={`p-10 rounded-full mb-8 relative transition-all duration-500 ${isTriggered ? 'bg-danger/20' : 'bg-red-50'}`}>
                {/* Pulse Rings */}
                <div className={`absolute inset-0 rounded-full border-4 border-danger opacity-20 ${isTriggered ? 'animate-ping' : ''}`}></div>
                <div className={`absolute -inset-4 rounded-full border border-danger opacity-10 ${isTriggered ? 'animate-ping animation-delay-500' : ''}`}></div>

                <div className="relative z-10">
                    <button
                        onClick={handleSOS}
                        className={`w-64 h-64 rounded-full flex flex-col items-center justify-center gap-4 text-white shadow-2xl transition-transform active:scale-95
                            ${isTriggered ? 'bg-danger animate-pulse' : 'bg-gradient-to-br from-red-500 to-red-600 hover:scale-105'}
                        `}
                    >
                        <FontAwesomeIcon icon={faBell} className="text-6xl" />
                        <span className="text-2xl font-bold">EMERGENCY<br />SOS</span>
                    </button>
                </div>
            </div>

            <h2 className="text-2xl font-bold text-gray-800 mb-2">Emergency Protocol</h2>
            <p className="text-gray-500 max-w-md mb-8">
                Pressing this button will immediately notify the on-call neurologist and trigger an ambulance dispatch to the registered patient address.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
                <ContactCard name="Dr. A. Smith" role="Neurologist" phone="+1 555-0123" icon={faUserMd} />
                <ContactCard name="City General" role="Ambulance" phone="911" icon={faAmbulance} />
            </div>
        </motion.div>
    );
};

const ContactCard = ({ name, role, phone, icon }) => (
    <div className="glass p-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-red-50 rounded-full flex items-center justify-center text-danger">
                <FontAwesomeIcon icon={icon} size="lg" />
            </div>
            <div className="text-left">
                <h4 className="font-bold text-gray-800">{name}</h4>
                <p className="text-xs text-gray-500">{role}</p>
            </div>
        </div>
        <a href={`tel:${phone}`} className="btn-icon bg-green-50 text-success hover:bg-green-100">
            <FontAwesomeIcon icon={faPhoneAlt} />
        </a>
    </div>
);

export default SOS;
