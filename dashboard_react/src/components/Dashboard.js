import React, { useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import confetti from 'canvas-confetti';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faCheckCircle, faPlayCircle, faFilePdf, faBrain, faChartLine, faMicroscope } from '@fortawesome/free-solid-svg-icons';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial, Points, PointMaterial } from '@react-three/drei';
import * as random from 'maath/random/dist/maath-random.esm';
import {
    PieChart, Pie, Cell, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar,
    BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid
} from 'recharts';

import { useApp } from '../context/AppContext';
import { simulatePreprocessing, simulatePipeline, fetchInferenceResult } from '../utils/api';
import { pageVariants, cardVariants, fadeInUp } from '../utils/animations';

// --- Chart Components ---

const ConfidenceRadial = ({ result }) => {
    const value = result ? Math.round(result.confidence * 100) : 0;
    const data = [
        { name: 'Confidence', value: value, fill: result?.color === 'danger' ? '#dc3545' : '#28a745' },
        { name: 'Remaining', value: 100 - value, fill: '#e9ecef' }
    ];

    return (
        <div className="flex flex-col items-center justify-center h-full relative">
            <ResponsiveContainer width="100%" height={120}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={55}
                        startAngle={180}
                        endAngle={0}
                        paddingAngle={0}
                        dataKey="value"
                        stroke="none"
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                    </Pie>
                </PieChart>
            </ResponsiveContainer>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1 text-center">
                <p className="text-2xl font-black text-gray-800">{value}%</p>
                <p className="text-[10px] font-bold text-gray-400 uppercase leading-none">Confidence</p>
            </div>
        </div>
    );
};

const BiomarkerRadar = ({ result }) => {
    const data = [
        { subject: 'Atrophy', A: result ? result.atrophy : 65, fullMark: 100 },
        { subject: 'Amyloid', A: result ? result.amyloid : 45, fullMark: 100 },
        { subject: 'Tau', A: result ? result.atrophy * 0.9 : 55, fullMark: 100 },
    ];

    return (
        <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="65%" data={data}>
                <PolarGrid stroke="#e0e0e0" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#6c757d', fontWeight: 'bold' }} />
                <Radar
                    name="Patient"
                    dataKey="A"
                    stroke="#007bff"
                    strokeWidth={2}
                    fill="#007bff"
                    fillOpacity={0.4}
                />
            </RadarChart>
        </ResponsiveContainer>
    );
};

const PopulationBar = ({ result }) => {
    const data = [
        { name: 'Avg', value: 45 },
        { name: 'You', value: result ? (result.atrophy + result.amyloid) / 2 : 60 },
    ];

    return (
        <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 20, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#6c757d' }} axisLine={false} tickLine={false} />
                <YAxis hide domain={[0, 100]} />
                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {data.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === 1 ? (result?.color === 'danger' ? '#dc3545' : '#007bff') : '#adb5bd'} />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
};



// --- Advanced 3D Components ---

const NeuralGalaxy = ({ diagnosis }) => {
    const ref = useRef();

    // Determine visual properties based on diagnosis
    const config = useMemo(() => {
        switch (diagnosis) {
            case 'AD': return { color: '#ff0055', radius: 1.4, count: 5000 }; // Atrophy & Danger
            case 'MCI': return { color: '#ffaa00', radius: 1.5, count: 6500 }; // Warning
            case 'CN':
            default: return { color: '#00f2ff', radius: 1.6, count: 9000 }; // Healthy
        }
    }, [diagnosis]);

    const [sphere] = useState(() => random.inSphere(new Float32Array(config.count * 3), { radius: config.radius }));

    // Re-generate sphere if diagnosis changes (key changes in parent) could be expensive, 
    // but for now we'll rely on React key to remount or just accept the initial state if not remounting.
    // To properly update density dynamically we'd need useEffect. 
    // For simplicity, we will stick to the initial generation or force remount via key.

    useFrame((state, delta) => {
        ref.current.rotation.x -= delta / 15;
        ref.current.rotation.y -= delta / 20;
    });

    return (
        <group rotation={[0, 0, Math.PI / 4]}>
            <Points ref={ref} positions={sphere} stride={3} frustumCulled={false}>
                <PointMaterial
                    transparent
                    color={config.color}
                    size={0.012}
                    sizeAttenuation={true}
                    depthWrite={false}
                    opacity={0.8}
                    blending={2}
                />
            </Points>
        </group>
    );
};

const HolographicHull = ({ diagnosis }) => {
    const meshRef = useRef();

    const config = useMemo(() => {
        switch (diagnosis) {
            case 'AD': return { distort: 0.6, speed: 2.5, opacity: 0.1 }; // Unstable
            case 'MCI': return { distort: 0.45, speed: 2, opacity: 0.12 };
            case 'CN':
            default: return { distort: 0.3, speed: 1.5, opacity: 0.15 }; // Stable
        }
    }, [diagnosis]);

    useFrame((state) => {
        const t = state.clock.getElapsedTime();
        meshRef.current.rotation.y = t * 0.1;
        meshRef.current.rotation.z = t * 0.05;
    });

    return (
        <Sphere ref={meshRef} visible args={[1, 64, 64]} scale={1.8}>
            <MeshDistortMaterial
                color="#0044aa"
                attach="material"
                distort={config.distort}
                speed={config.speed}
                roughness={0.1}
                metalness={0.8}
                transparent
                opacity={config.opacity}
                wireframe={false}
                side={2}
            />
        </Sphere>
    );
};

const SynapseSparks = ({ diagnosis }) => {
    const ref = useRef();
    const count = diagnosis === 'AD' ? 200 : diagnosis === 'MCI' ? 350 : 500; // Less activity in AD

    const [sparks] = useState(() => random.inSphere(new Float32Array(count * 3), { radius: 1.7 }));

    useFrame((state, delta) => {
        ref.current.rotation.x += delta / 10;
        ref.current.rotation.y += delta / 10;
    });

    return (
        <group rotation={[0, 0, Math.PI / 4]}>
            <Points ref={ref} positions={sparks} stride={3} frustumCulled={false}>
                <PointMaterial
                    transparent
                    color="#ffffff"
                    size={0.03}
                    sizeAttenuation={true}
                    depthWrite={false}
                    opacity={0.6}
                />
            </Points>
        </group>
    );
};

const DiseaseInfo = ({ result }) => {
    if (!result) return null;

    const info = {
        'AD': {
            title: "Alzheimer's Disease",
            symptoms: ["Memory loss", "Confusion with time/place", "Trouble completing tasks"],
            desc: "Patterns indicate significant atrophy in the hippocampus and cortical thinning.",
            refImage: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Brain_MRI_112010_rgb.jpg/600px-Brain_MRI_112010_rgb.jpg" // Placeholder for AD
        },
        'MCI': {
            title: "Mild Cognitive Impairment",
            symptoms: ["Forgetfulness", "Impulsivity", "Depression"],
            desc: "Early signs of cognitive decline detected, potentially reversible.",
            refImage: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Brain_MRI_112010_rgb.jpg/600px-Brain_MRI_112010_rgb.jpg" // Placeholder for MCI
        },
        'CN': {
            title: "Cognitively Normal",
            symptoms: ["No significant issues detected"],
            desc: "Brain volume and density appear within normal healthy ranges.",
            refImage: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Brain_MRI_112010_rgb.jpg/600px-Brain_MRI_112010_rgb.jpg"
        }
    };

    const details = info[result.label] || info['CN'];

    return (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="p-4 bg-white/5 rounded-xl border border-white/10 mt-4 backdrop-blur-md">
            <div className="flex gap-4">
                <div className="w-1/3">
                    <p className="text-[10px] uppercase font-bold text-gray-400 mb-1">Reference: Typical {result.label}</p>
                    <img src={details.refImage} alt="Reference" className="w-full rounded-lg border border-white/20 opacity-80" />
                </div>
                <div className="w-2/3">
                    <h4 className={`text-lg font-bold mb-1 ${result.color === 'danger' ? 'text-red-400' : 'text-blue-400'}`}>{details.title}</h4>
                    <p className="text-xs text-gray-300 mb-2">{details.desc}</p>
                    <div className="space-y-1">
                        <p className="text-[10px] uppercase font-bold text-gray-500">Common Symptoms:</p>
                        <ul className="text-xs text-gray-400 list-disc list-inside">
                            {details.symptoms.map(s => <li key={s}>{s}</li>)}
                        </ul>
                    </div>
                </div>
            </div>
            <div className="mt-3 pt-3 border-t border-white/10">
                <p className="text-[9px] text-gray-500 leading-tight">
                    <span className="font-bold text-red-400/80">DISCLAIMER:</span> This visualization is a generative representation based on analysis data, NOT a direct anatomical reconstruction. Do not use for definitive medical diagnosis. Consult a specialist.
                </p>
            </div>
        </motion.div>
    );
};

const Dashboard = () => {
    const { state, dispatch } = useApp();
    const fileInputRef = useRef(null);
    const [dragActive, setDragActive] = useState(false);

    // --- Actions ---
    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) processFile(file);
    };

    const processFile = async (file) => {
        // 1. Start Preprocessing
        dispatch({ type: 'SET_PROCESSING', payload: true });
        dispatch({ type: 'RESET_DASHBOARD' });

        try {
            // Task 1: Preprocessing & Upload
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

            // Task 2/3: Inference
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

                {/* 3D Brain Viewer (Procedural Three.js) */}

                <div className="glass h-[500px] relative overflow-hidden flex flex-col p-0 shadow-2xl border-white/50 bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
                    <div className="absolute top-6 left-6 z-10 pointer-events-none">
                        <div className="flex items-center gap-2 mb-1">
                            <div className="bg-white/10 backdrop-blur-md p-2 rounded-lg shadow-sm border border-white/20">
                                <FontAwesomeIcon icon={faBrain} className="text-blue-400 text-xl" />
                            </div>
                            <div>
                                <h3 className="text-xl font-bold text-white">3D Neural Analysis</h3>
                                <p className="text-xs text-blue-200 font-medium tracking-wide">GENERATIVE MODEL • CORTICAL SURFACE</p>
                            </div>
                        </div>
                    </div>

                    <div className="absolute top-6 right-6 z-10 flex gap-2 pointer-events-none">
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className={`px-3 py-1.5 rounded-full text-xs font-bold border ${state.isProcessing ? 'bg-blue-500/20 text-blue-200 border-blue-400/30 animate-pulse' : 'bg-white/10 text-gray-300 border-white/10'}`}
                        >
                            {state.isProcessing ? 'PROCESSING...' : 'READY'}
                        </motion.div>
                    </div>

                    <div className="w-full h-full cursor-move">
                        <Canvas camera={{ position: [0, 0, 4] }}>
                            <ambientLight intensity={0.5} />
                            <directionalLight position={[10, 10, 5]} intensity={1} color="#ffffff" />
                            <pointLight position={[-10, -10, -5]} intensity={0.5} color="#007bff" />

                            <NeuralGalaxy diagnosis={state.result?.label} key={`galaxy-${state.result?.label}`} />
                            <SynapseSparks diagnosis={state.result?.label} key={`sparks-${state.result?.label}`} />
                            <HolographicHull diagnosis={state.result?.label} />

                            <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                        </Canvas>
                    </div>

                    {/* Overlay Gradient at bottom */}
                    <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-slate-900 to-transparent pointer-events-none"></div>
                </div>

                {/* Disease Info Panel (New) */}
                <DiseaseInfo result={state.result} />

                {/* Live Analysis Graphs */}
                <div className="grid grid-cols-3 gap-4 h-40">
                    {/* 1. Confidence Gauge */}
                    <motion.div whileHover={{ y: -5 }} className="glass p-2 flex flex-col items-center justify-between bg-white/60">
                        <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider w-full text-center border-b border-gray-100 pb-2">AI Certainty</h4>
                        <div className="flex-1 w-full relative">
                            {!state.result && state.isProcessing && (
                                <div className="absolute inset-0 flex items-center justify-center bg-white/50 backdrop-blur-sm z-10">
                                    <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                                </div>
                            )}
                            <ConfidenceRadial result={state.result} />
                        </div>
                    </motion.div>

                    {/* 2. Biomarker Radar */}
                    <motion.div whileHover={{ y: -5 }} className="glass p-2 flex flex-col items-center justify-between bg-white/60">
                        <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider w-full text-center border-b border-gray-100 pb-2">Biomarkers</h4>
                        <div className="flex-1 w-full relative">
                            <BiomarkerRadar result={state.result} />
                        </div>
                    </motion.div>

                    {/* 3. Population Comparison */}
                    <motion.div whileHover={{ y: -5 }} className="glass p-2 flex flex-col items-center justify-between bg-white/60">
                        <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider w-full text-center border-b border-gray-100 pb-2">Risk vs Avg</h4>
                        <div className="flex-1 w-full relative">
                            <PopulationBar result={state.result} />
                        </div>
                    </motion.div>
                </div>
            </motion.div >


            {/* --- Right Column: Controls & Results --- */}
            < div className="flex flex-col gap-6" >

                {/* Upload Zone */}
                < motion.div variants={cardVariants} className="glass p-6 text-center relative overflow-hidden" >
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center justify-center gap-2">
                        <FontAwesomeIcon icon={faCloudUploadAlt} className="text-primary" />
                        Input MRI Scan
                    </h3>

                    {
                        !state.isProcessing && !state.result ? (
                            <div
                                className={`border-2 border-dashed rounded-2xl p-8 cursor-pointer transition-all duration-300 group ${dragActive ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-gray-300 hover:border-primary hover:bg-gray-50'}`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                                onClick={() => fileInputRef.current.click()}
                            >
                                <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform shadow-sm">
                                    <FontAwesomeIcon icon={faCloudUploadAlt} className="text-3xl text-primary" />
                                </div>
                                <p className="font-bold text-gray-700 text-lg">Click to Upload</p>
                                <p className="text-sm text-gray-500 mt-1">or drag and drop .nii file here</p>
                                <p className="text-xs text-gray-400 mt-4 bg-gray-100 inline-block px-3 py-1 rounded-full">Supports NIfTI (.nii, .nii.gz) &lt;100MB</p>
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
                                <div className="mb-2 flex justify-between items-end">
                                    <span className="text-xs font-bold text-primary tracking-wide uppercase">{state.processStatus || 'Initializing...'}</span>
                                    <span className="text-sm font-bold text-gray-700">{Math.round(state.processStep)}%</span>
                                </div>
                                <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden shadow-inner mb-8">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${state.processStep}%` }}
                                        className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500"
                                    />
                                </div>

                                {/* Steps List */}
                                <div className="mt-6 space-y-4 text-left">
                                    <StepItem
                                        icon={faMicroscope}
                                        title="Task 1: Preprocessing"
                                        desc="N4 Bias, Skull Stripping, MNI Reg."
                                        done={state.processStep > 30}
                                        processing={state.processStep <= 30}
                                    />
                                    <StepItem
                                        icon={faBrain}
                                        title="Task 2: Binary Class."
                                        desc="CN vs AD (MedicalNet ResNet-10)"
                                        done={state.processStep > 60}
                                        processing={state.processStep > 30 && state.processStep <= 60}
                                    />
                                    <StepItem
                                        icon={faChartLine}
                                        title="Task 3: Multi-Class"
                                        desc="CN vs MCI vs AD Staging"
                                        done={state.processStep > 90}
                                        processing={state.processStep > 60}
                                    />
                                </div>
                            </div>
                        )
                    }
                </motion.div >

                {/* Results Panel */}
                < AnimatePresence >
                    {
                        state.result && (
                            <motion.div
                                variants={fadeInUp}
                                initial="hidden"
                                animate="show"
                                className="glass p-0 relative overflow-hidden shadow-xl"
                            >
                                {/* Result Header */}
                                <div className={`p-6 bg-gradient-to-br ${state.result.color === 'danger' ? 'from-red-50 to-red-100' : state.result.color === 'warning' ? 'from-amber-50 to-amber-100' : 'from-emerald-50 to-emerald-100'}`}>
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <div className="inline-flex items-center gap-2 px-2 py-1 bg-white/60 rounded-md text-xs font-bold uppercase tracking-wider text-gray-600 mb-2 border border-black/5">
                                                AI Diagnosis
                                            </div>
                                            <h2 className={`text-4xl font-extrabold ${state.result.color === 'danger' ? 'text-red-600' : state.result.color === 'warning' ? 'text-amber-600' : 'text-emerald-600'}`}>
                                                {state.result.label}
                                            </h2>
                                            <p className="text-sm font-medium text-gray-600 mt-1">{state.result.full}</p>
                                        </div>
                                        <div className="text-right bg-white/50 p-3 rounded-xl backdrop-blur-sm border border-white/40">
                                            <p className="text-[10px] font-bold text-gray-500 uppercase">Confidence</p>
                                            <p className="text-3xl font-black text-gray-800">{(state.result.confidence * 100).toFixed(1)}%</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Result Metrics */}
                                <div className="p-6 space-y-5 bg-white/40">
                                    <MetricBar label="Hippocampal Atrophy" value={state.result.atrophy} color="bg-orange-500" />
                                    <MetricBar label="Amyloid Plaque Load" value={state.result.amyloid} color="bg-red-500" />
                                    <MetricBar label="Ventricle Enlargement" value={state.result.atrophy * 0.8} color="bg-blue-500" />
                                </div>

                                {/* Actions */}
                                <div className="p-4 bg-gray-50/50 flex gap-3 border-t border-gray-100">
                                    <button className="flex-1 btn-primary text-sm shadow-lg shadow-blue-500/20">
                                        View Full Report
                                    </button>
                                    <button className="px-4 py-2 bg-white border border-gray-200 rounded-xl text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors shadow-sm">
                                        <FontAwesomeIcon icon={faFilePdf} />
                                    </button>
                                </div>
                            </motion.div>
                        )
                    }
                </AnimatePresence >

            </div >
        </motion.div >
    );
};

// Sub-components for cleaner code
const StepItem = ({ icon, title, desc, done, processing }) => (
    <div className={`relative pl-4 border-l-2 transitiion-colors duration-300 ${done ? 'border-emerald-500' : processing ? 'border-primary' : 'border-gray-200'}`}>
        <div className="flex items-start gap-3">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-all duration-500 ${done ? 'bg-emerald-100 text-emerald-600' : processing ? 'bg-blue-100 text-primary scale-110' : 'bg-gray-100 text-gray-400'}`}>
                {processing ? <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" /> : <FontAwesomeIcon icon={icon} className="text-sm" />}
            </div>
            <div>
                <h4 className={`text-sm font-bold transition-colors ${done ? 'text-gray-800' : processing ? 'text-primary' : 'text-gray-400'}`}>{title}</h4>
                <p className="text-xs text-gray-500 mt-0.5">{desc}</p>
            </div>
            {done && <FontAwesomeIcon icon={faCheckCircle} className="ml-auto text-emerald-500 text-lg animate-bounce-short" />}
        </div>
    </div>
);

const MetricBar = ({ label, value, color }) => (
    <div>
        <div className="flex justify-between text-xs mb-1.5 align-middle">
            <span className="font-semibold text-gray-600 flex items-center gap-2">
                {label}
                <div className="group relative">
                    <span className="w-3 h-3 rounded-full bg-gray-200 text-[8px] flex items-center justify-center text-gray-500 cursor-help">?</span>
                    <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block w-32 bg-gray-800 text-white text-[10px] p-1.5 rounded z-20 text-center">
                        Clinical indicator derived from segmentation masks
                    </span>
                </div>
            </span>
            <span className="font-bold text-gray-800">{value.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200/50 rounded-full h-2.5 overflow-hidden">
            <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ duration: 1.2, delay: 0.2, type: "spring" }}
                className={`h-full ${color} rounded-full shadow-[0_0_10px_rgba(0,0,0,0.1)]`}
            />
        </div>
    </div>
);

export default Dashboard;
