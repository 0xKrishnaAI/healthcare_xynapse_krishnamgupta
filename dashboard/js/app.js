// NeuroDx Dashboard Logic
// Handles UI interactions, Tabs, Upload Simulation, and Data Injection

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUpload();
    initSimulatedReports();
});

// --- Tab Navigation ---
function initTabs() {
    const tabs = document.querySelectorAll('.nav-item:not(.alert-item)');
    const views = document.querySelectorAll('.view-section');
    const pageTitle = document.getElementById('page-title');

    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();

            // Remove active class
            tabs.forEach(t => t.classList.remove('active'));
            views.forEach(v => v.classList.remove('active'));

            // Add active class
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            const view = document.getElementById(`${tabId}-view`);

            if (view) view.classList.add('active');

            // Update Title
            pageTitle.textContent = tab.querySelector('span').textContent;
        });
    });
}

// --- Upload Simulation ---
function initUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const processingState = document.getElementById('processing-state');
    const uploadProgress = document.getElementById('upload-progress');
    const statusText = document.getElementById('status-text');
    const resultsPanel = document.getElementById('results-panel');
    const diagnosisRing = document.getElementById('diagnosis-ring');
    const diagnosisLabel = document.getElementById('diagnosis-label');
    const confidenceScore = document.getElementById('confidence-score');
    const atrophyBar = document.getElementById('atrophy-bar');
    const amyloidBar = document.getElementById('amyloid-bar');

    // Drag & Drop visual feedback
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.border = '2px dashed var(--primary-color)';
        dropZone.style.background = 'rgba(0, 123, 255, 0.05)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.border = '2px dashed #cbd5e0';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;

        // Reset UI
        resultsPanel.classList.add('hidden');
        dropZone.querySelector('.upload-icon').classList.add('hidden');
        dropZone.querySelector('h3').classList.add('hidden');
        dropZone.querySelector('p').classList.add('hidden');
        dropZone.querySelector('button').classList.add('hidden');

        // Start Processing
        processingState.classList.remove('hidden');
        animateProgress();
    }

    function animateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 100) progress = 100;

            uploadProgress.style.width = `${progress}%`;

            // Update status text based on progress stages
            if (progress < 30) statusText.textContent = "Preprocessing (Task 1): N4 Bias Correction...";
            else if (progress < 60) statusText.textContent = "Preprocessing (Task 1): Skull Stripping & MNI Registration...";
            else if (progress < 85) statusText.textContent = "AI Analysis (Task 3): Multi-Class Classification...";
            else statusText.textContent = "Finalizing Report...";

            if (progress === 100) {
                clearInterval(interval);
                setTimeout(showResults, 500);
            }
        }, 150);
    }

    function showResults() {
        // Hide processing, show results
        processingState.classList.add('hidden');
        dropZone.style.display = 'none'; // Hide upload area completely or reset
        resultsPanel.classList.remove('hidden');
        resultsPanel.style.animation = 'fadeIn 0.5s ease';

        // Simulate Random AI Result (for demo)
        const scenarios = [
            { label: 'CN', full: 'Cognitively Normal', class: 'cn', score: 98, atrophy: 12, amyloid: 5 },
            { label: 'MCI', full: 'Mild Impairment', class: 'mci', score: 76, atrophy: 35, amyloid: 42 },
            { label: 'AD', full: 'Alzheimer\'s Disease', class: 'ad', score: 92, atrophy: 68, amyloid: 85 }
        ];

        const result = scenarios[Math.floor(Math.random() * scenarios.length)];

        // Update DOM
        diagnosisRing.className = `diagnosis-ring ${result.class}`;
        diagnosisLabel.textContent = result.label;
        diagnosisLabel.title = result.full;

        // Animate numbers
        animateValue(confidenceScore, 0, result.score, 1000, '%');

        // Animate Bars
        setTimeout(() => {
            atrophyBar.style.width = `${result.atrophy}%`;
            amyloidBar.style.width = `${result.amyloid}%`;
        }, 200);

        // Update Brain Viewer Highlight (if integrated)
        if (window.updateBrainColor) {
            window.updateBrainColor(result.class);
        }
    }
}

// Helper: Animate numbers
function animateValue(obj, start, end, duration, suffix = '') {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start) + suffix;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// --- Simulated Reports Table ---
function initSimulatedReports() {
    const tableBody = document.querySelector('#reports-table tbody');
    if (!tableBody) return;

    const data = [
        { id: '002_S_0295', date: '2023-10-15', type: 'T1-MRI', diag: 'CN (Normal)', conf: '98%', status: 'Accepted' },
        { id: '002_S_0413', date: '2023-10-12', type: 'T1-MRI', diag: 'AD (Alzheimer\'s)', conf: '92%', status: 'Accepted' },
        { id: '002_S_1234', date: '2023-10-10', type: 'T1-MRI', diag: 'MCI', conf: '55%', status: 'Rejected (<60%)' },
    ];

    data.forEach(row => {
        const tr = document.createElement('tr');
        const statusClass = row.status.includes('Rejected') ? 'rejected' : 'accepted';

        tr.innerHTML = `
            <td>${row.id}</td>
            <td>${row.date}</td>
            <td>${row.type}</td>
            <td>${row.diag}</td>
            <td>${row.conf}</td>
            <td><span class="status-pill ${statusClass}">${row.status}</span></td>
        `;
        tableBody.appendChild(tr);
    });
}
