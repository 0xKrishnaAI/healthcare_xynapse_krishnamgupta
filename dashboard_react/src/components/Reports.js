import React from 'react';
import { motion } from 'framer-motion';
import { pageVariants } from '../utils/animations';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload, faFilter } from '@fortawesome/free-solid-svg-icons';
import { getMockReports } from '../utils/api';

const Reports = () => {
    const reports = getMockReports();

    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
        >
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Diagnostic Reports</h2>
                <div className="flex gap-3">
                    <button className="px-4 py-2 bg-white text-gray-600 rounded-xl shadow-sm hover:bg-gray-50 text-sm font-semibold flex items-center gap-2">
                        <FontAwesomeIcon icon={faFilter} /> Filter
                    </button>
                    <button className="px-4 py-2 bg-primary text-white rounded-xl shadow-md hover:bg-primary-dark text-sm font-semibold flex items-center gap-2">
                        <FontAwesomeIcon icon={faDownload} /> Export CSV
                    </button>
                </div>
            </div>

            <div className="glass overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead className="bg-gray-50/50 border-b border-gray-100">
                        <tr>
                            <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Patient ID</th>
                            <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Date</th>
                            <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Prediction</th>
                            <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Confidence</th>
                            <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Status</th>
                            <th className="p-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                        {reports.map((report) => (
                            <motion.tr
                                key={report.id}
                                whileHover={{ backgroundColor: "rgba(249, 250, 251, 0.5)" }}
                                className="transition-colors"
                            >
                                <td className="p-4 font-semibold text-gray-700">{report.id}</td>
                                <td className="p-4 text-gray-500">{report.date}</td>
                                <td className="p-4">
                                    <span className={`px-2 py-1 rounded-md text-xs font-bold
                                        ${report.pred === 'CN' ? 'bg-green-100 text-green-700' :
                                            report.pred === 'MCI' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}
                                    `}>
                                        {report.pred}
                                    </span>
                                </td>
                                <td className="p-4 text-gray-700 font-medium">
                                    <div className="flex items-center gap-2">
                                        <div className="w-16 bg-gray-200 h-1.5 rounded-full overflow-hidden">
                                            <div className="bg-primary h-full" style={{ width: `${report.conf}%` }}></div>
                                        </div>
                                        {report.conf}%
                                    </div>
                                </td>
                                <td className="p-4 text-sm text-gray-500">{report.status}</td>
                                <td className="p-4">
                                    <button className="text-primary hover:text-primary-dark font-medium text-sm">View</button>
                                </td>
                            </motion.tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </motion.div>
    );
};

export default Reports;
