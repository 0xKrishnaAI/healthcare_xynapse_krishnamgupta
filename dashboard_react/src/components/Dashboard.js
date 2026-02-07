import { motion, AnimatePresence } from 'framer-motion';
import confetti from 'canvas-confetti';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faCheckCircle, faPlayCircle, faFilePdf } from '@fortawesome/free-solid-svg-icons';

import { useApp } from '../context/AppContext';
// import BrainViewer from './BrainViewer'; // Deprecated in favor of Spline
import { simulatePreprocessing, simulatePipeline, fetchInferenceResult } from '../utils/api';
import { pageVariants, cardVariants, fadeInUp } from '../utils/animations';

const Dashboard = () => {
    const { state, dispatch } = useApp();
    const fileInputRef = useRef(null);

    // --- Actions ---
    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // 1. Start Preprocessing
        dispatch({ type: 'SET_PROCESSING', payload: true });
        dispatch({ type: 'RESET_DASHBOARD' });

        try {
            // Simulate Upload
            await simulatePreprocessing(file);

            // Simulate Pipeline Steps
            const steps = await simulatePipeline(file);

            // Loop through steps to show progress bar moving
            for (const step of steps) {
                dispatch({
                    type: 'UPDATE_PROGRESS',
                    payload: { progress: step.progress, status: step.text }
                });
                // Add small delay between steps for visual effect
                await new Promise(r => setTimeout(r, 600));
            }

            // Fetch Final Result
            const result = await fetchInferenceResult();
            dispatch({ type: 'SET_RESULT', payload: result });

            // Trigger Confetti if completed
            if (result) {
                confetti({
                    particleCount: 150,
                    spread: 80,
                    origin: { y: 0.6 },
                    colors: [result.color === 'danger' ? '#dc3545' : '#28a745', '#007bff']
                });

                // Add Notification
                dispatch({
                    type: 'ADD_NOTIFICATION',
                    payload: { text: `Analysis Complete: ${result.label} detected.`, type: result.color }
                });
            }

        } catch (error) {
            console.error("Analysis Failed", error);
        }
    };

    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="grid lg:grid-cols-3 md:grid-cols-2 sm:grid-cols-1 gap-6"
        >
            {/* --- Left Column: Visualization (Span 2) --- */}
            <motion.div variants={cardVariants} className="lg:col-span-2 flex flex-col gap-6">

                {/* 3D Brain Viewer (Spline Embed) */}
                <div className="glass h-[500px] relative overflow-hidden flex flex-col p-0">
                    <div className="absolute top-4 left-4 z-10 pointer-events-none">
                        <h3 className="text-lg font-bold text-gray-700">3D Neural Analysis</h3>
                        <p className="text-xs text-gray-500">Interactive Spline Model</p>
                    </div>

                    <div className="absolute top-4 right-4 z-10 flex gap-2 pointer-events-none">
                        <span className={`px-2 py-1 rounded-lg text-xs font-bold ${state.isProcessing ? 'bg-blue-100 text-primary' : 'bg-gray-100 text-gray-500'}`}>
                            {state.isProcessing ? 'PROCESSING' : 'IDLE'}
                        </span>
                    </div>

                    <iframe
                        src='https://my.spline.design/particleuibrain-rdjn2jg6NlGU7CyoplyPPwoP/'
                        frameBorder='0'
                        width='100%'
                        height='100%'
                        title="Spline 3D Brain"
                        className="w-full h-full"
                    ></iframe>
                </div>

                {/* Slice Previews */}
                <div className="grid grid-cols-3 gap-4">
                    {['Axial', 'Sagittal', 'Coronal'].map((view, i) => (
                        <div key={view} className="glass p-2 relative group cursor-pointer hover:scale-105 transition-transform duration-300">
                            <div className="aspect-square bg-gray-200 rounded-lg overflow-hidden relative">
                                {/* Placeholder Gradient for slices */}
                                <div className={`w-full h-full bg-gradient-to-br from-gray-800 to-black opacity-80 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white/20`}>
                                    <FontAwesomeIcon icon={faPlayCircle} size="2x" />
                                </div>
                            </div>
                            <span className="absolute bottom-3 left-3 text-xs font-bold text-white drop-shadow-md">{view} View</span>
                        </div>
                    ))}
                </div>
            </motion.div>


            {/* --- Right Column: Controls & Results --- */}
            <div className="flex flex-col gap-6">

                {/* Upload Zone */}
                <motion.div variants={cardVariants} className="glass p-6 text-center">
                    <h3 className="text-lg font-bold text-gray-800 mb-4">Input MRI Scan</h3>

                    {!state.isProcessing && !state.result ? (
                        <div
                            className="border-2 border-dashed border-gray-300 rounded-2xl p-8 hover:bg-primary/5 hover:border-primary cursor-pointer transition-colors group"
                            onClick={() => fileInputRef.current.click()}
                        >
                            <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                                <FontAwesomeIcon icon={faCloudUploadAlt} className="text-2xl text-primary" />
                            </div>
                            <p className="font-semibold text-gray-700">Click to Upload</p>
                            <p className="text-xs text-gray-400 mt-1">Supports NIfTI (.nii, .nii.gz)</p>
                            <input
                                type="file"
                                ref={fileInputRef}
                                className="hidden"
                                onChange={handleFileUpload}
                                accept=".nii,.nii.gz"
                            />
                        </div>
                    ) : (
                        <div className="py-4">
                            {/* Progress Simulation */}
                            <div className="mb-4 flex justify-between items-end">
                                <span className="text-xs font-bold text-primary">{state.processStatus || 'Initializing...'}</span>
                                <span className="text-sm font-bold text-gray-700">{Math.round(state.processStep)}%</span>
                            </div>
                            <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${state.processStep}%` }}
                                    className="h-full bg-gradient-to-r from-primary to-purple-500"
                                />
                            </div>

                            {/* Steps List */}
                            <div className="mt-6 space-y-3 text-left">
                                <StepItem label="Task 1: Preprocessing" done={state.processStep > 30} processing={state.processStep <= 30} />
                                <StepItem label="Task 2: Binary Class." done={state.processStep > 60} processing={state.processStep > 30 && state.processStep <= 60} />
                                <StepItem label="Task 3: Multi-Class Logic" done={state.processStep > 90} processing={state.processStep > 60} />
                            </div>
                        </div>
                    )}
                </motion.div>

                {/* Results Panel */}
                <AnimatePresence>
                    {state.result && (
                        <motion.div
                            variants={fadeInUp}
                            initial="hidden"
                            animate="show"
                            className="glass p-6 relative overflow-hidden"
                        >
                            <div className={`absolute top-0 left-0 w-1 h-full bg-${state.result.color}`}></div>

                            <div className="flex justify-between items-start mb-6">
                                <div>
                                    <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Diagnosis</p>
                                    <h2 className={`text-2xl font-bold text-${state.result.color === 'danger' ? 'red-500' : state.result.color === 'warning' ? 'yellow-500' : 'green-500'}`}>
                                        {state.result.label}
                                    </h2>
                                    <p className="text-xs text-gray-500">{state.result.full}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs font-bold text-gray-400 uppercase">Confidence</p>
                                    <p className="text-2xl font-bold text-gray-800">{(state.result.confidence * 100).toFixed(1)}%</p>
                                </div>
                            </div>

                            {/* Metrics */}
                            <div className="space-y-4">
                                <MetricBar label="Brain Atrophy" value={state.result.atrophy} color="bg-orange-400" />
                                <MetricBar label="Amyloid Load" value={state.result.amyloid} color="bg-red-400" />
                            </div>

                            <div className="mt-6 pt-4 border-t border-gray-100 flex gap-3">
                                <button className="flex-1 btn-primary text-sm">
                                    View Report
                                </button>
                                <button className="p-2 bg-gray-100 rounded-lg text-gray-600 hover:bg-gray-200">
                                    <FontAwesomeIcon icon={faFilePdf} />
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </div>
        </motion.div>
    );
};

// Sub-components for cleaner code
const StepItem = ({ label, done, processing }) => (
    <div className={`flex items-center gap-3 text-sm ${done ? 'text-success' : processing ? 'text-primary font-bold' : 'text-gray-400'}`}>
        <div className={`w-5 h-5 rounded-full flex items-center justify-center border ${done ? 'bg-success border-success text-white' : processing ? 'border-primary border-2' : 'border-gray-300'}`}>
            {done && <FontAwesomeIcon icon={faCheckCircle} className="text-xs" />}
            {processing && <div className="w-2 h-2 bg-primary rounded-full animate-ping" />}
        </div>
        <span>{label}</span>
    </div>
);

const MetricBar = ({ label, value, color }) => (
    <div>
        <div className="flex justify-between text-xs mb-1">
            <span className="font-semibold text-gray-600">{label}</span>
            <span className="font-bold text-gray-800">{value}%</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
            <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ duration: 1, delay: 0.5 }}
                className={`h-full ${color}`}
            />
        </div>
    </div>
);

export default Dashboard;
