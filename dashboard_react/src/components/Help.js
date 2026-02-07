import React from 'react';
import { motion } from 'framer-motion';
import { pageVariants, staggerContainer, cardVariants } from '../utils/animations';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faMagic, faLaptopMedical } from '@fortawesome/free-solid-svg-icons';

const Help = () => {
    const guides = [
        { icon: faCloudUploadAlt, title: 'Uploading Scans', text: 'Drag and drop .nii or .nii.gz MRI files into the dashboard upload zone. Ensure files are anonymized.' },
        { icon: faMagic, title: 'AI Analysis', text: 'The system automatically performs N4 Bias Correction and Skull Stripping before Classification.' },
        { icon: faLaptopMedical, title: 'Interpreting Results', text: 'Results show probability scores for CN (Normal), MCI (Mild Impairment), and AD (Alzheimer\'s).' }
    ];

    return (
        <motion.div variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">User Guide & Documentation</h2>

            <motion.div variants={staggerContainer} initial="hidden" animate="show" className="grid md:grid-cols-3 gap-6 mb-8">
                {guides.map((guide, i) => (
                    <motion.div key={i} variants={cardVariants} className="glass p-6 text-center">
                        <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-4 text-primary text-2xl">
                            <FontAwesomeIcon icon={guide.icon} />
                        </div>
                        <h3 className="font-bold text-gray-800 mb-2">{guide.title}</h3>
                        <p className="text-sm text-gray-500">{guide.text}</p>
                    </motion.div>
                ))}
            </motion.div>

            <div className="glass p-6 border-l-4 border-primary">
                <h3 className="font-bold text-gray-700">Medical Disclaimer</h3>
                <p className="text-sm text-gray-500 italic mt-2">
                    NeuroDx - BrainScan AI is a decision support tool and is NOT intended to be a replacement for professional medical diagnosis.
                    All AI-generated results should be verified by a certified radiologist or neurologist.
                </p>
                <p className="text-xs text-gray-400 mt-4">
                    v2.0.0 (Build 2023-10-24) | Powered by Simple3DCNN
                </p>
            </div>
        </motion.div>
    );
};

export default Help;
