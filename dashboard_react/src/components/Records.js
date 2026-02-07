import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { pageVariants, staggerContainer, cardVariants } from '../utils/animations';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFileAlt, faChevronDown, faChevronUp, faBrain } from '@fortawesome/free-solid-svg-icons';

const Records = () => {
    // Mock Data
    const records = [
        { id: 'REC-2023-001', patient: 'Anonymous-002', date: 'Oct 24, 2023', diagnosis: 'CN', confidence: 0.98, type: 'MRI (T1)' },
        { id: 'REC-2023-002', patient: 'Anonymous-128', date: 'Oct 22, 2023', diagnosis: 'AD', confidence: 0.94, type: 'MRI (T1)' },
        { id: 'REC-2023-003', patient: 'Anonymous-011', date: 'Oct 15, 2023', diagnosis: 'MCI', confidence: 0.76, type: 'MRI (T1)' },
    ];

    return (
        <motion.div variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Patient Records</h2>

            <motion.div
                variants={staggerContainer}
                initial="hidden"
                animate="show"
                className="grid md:grid-cols-2 gap-6"
            >
                {records.map((record) => (
                    <RecordCard key={record.id} record={record} />
                ))}
            </motion.div>
        </motion.div>
    );
};

const RecordCard = ({ record }) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <motion.div variants={cardVariants} layout className="glass p-4 cursor-pointer" onClick={() => setIsOpen(!isOpen)}>
            <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-white
                    ${record.diagnosis === 'AD' ? 'bg-danger' : record.diagnosis === 'MCI' ? 'bg-warning' : 'bg-success'}
                `}>
                    <FontAwesomeIcon icon={faBrain} />
                </div>
                <div className="flex-1">
                    <h4 className="font-bold text-gray-700">{record.id}</h4>
                    <p className="text-xs text-gray-500">{record.date} • {record.type}</p>
                </div>
                <FontAwesomeIcon icon={isOpen ? faChevronUp : faChevronDown} className="text-gray-400" />
            </div>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="pt-4 mt-4 border-t border-gray-100 text-sm text-gray-600 space-y-2">
                            <div className="flex justify-between">
                                <span>Subject ID:</span>
                                <span className="font-semibold">{record.patient}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Diagnosis:</span>
                                <span className={`font-bold ${record.diagnosis === 'AD' ? 'text-danger' : record.diagnosis === 'CN' ? 'text-success' : 'text-warning'}`}>
                                    {record.diagnosis}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>AI Confidence:</span>
                                <span className="font-bold">{(record.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="mt-4 bg-gray-50 p-2 rounded text-xs font-mono">
                                JSON_METADATA: &#123; "scan_res": "1mm", "tesla": "3T" &#125;
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}

export default Records;
