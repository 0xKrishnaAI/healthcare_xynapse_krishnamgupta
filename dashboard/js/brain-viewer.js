// Brain Viewer - Three.js Implementation
// Renders an interactive 3D brain model (using primitive shapes for demo)

// Wait for Three.js to load
window.addEventListener('load', initBrainViewer);

let scene, camera, renderer, brainMesh, particles;
let autoRotate = true;

function initBrainViewer() {
    const container = document.getElementById('brain-container');
    const loader = document.getElementById('brain-loader');

    if (!container || !THREE) return;

    // 1. Scene Setup
    scene = new THREE.Scene();
    // scene.background = new THREE.Color(0xf0f4f8); // Match app bg or transparent

    // 2. Camera
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.z = 5;
    camera.position.y = 1;

    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    const blueLight = new THREE.PointLight(0x007bff, 1, 10);
    blueLight.position.set(-2, 2, 2);
    scene.add(blueLight);

    // 5. Brain Model (Procedural Generation using Geometries)
    // Since we don't have an OBJ file, we'll construct a stylized brain using spheres
    const brainGroup = new THREE.Group();

    const hemisphereGeo = new THREE.SphereGeometry(1.2, 32, 32);
    hemisphereGeo.scale(0.8, 1, 1.2); // stretch to look like a brain lobe

    // Material - Glassy/Holographic look
    const brainMat = new THREE.MeshPhongMaterial({
        color: 0xe2e8f0,
        shininess: 80,
        transparent: true,
        opacity: 0.9,
        flatShading: false
    });

    // Left Lobe
    const leftLobe = new THREE.Mesh(hemisphereGeo, brainMat);
    leftLobe.position.x = -0.65;
    leftLobe.rotation.z = 0.2;
    brainGroup.add(leftLobe);

    // Right Lobe
    const rightLobe = new THREE.Mesh(hemisphereGeo, brainMat);
    rightLobe.position.x = 0.65;
    rightLobe.rotation.z = -0.2;
    brainGroup.add(rightLobe);

    // Cerebellum (Small back part)
    const cerebellumGeo = new THREE.SphereGeometry(0.6, 16, 16);
    cerebellumGeo.scale(1, 0.6, 0.6);
    const cerebellum = new THREE.Mesh(cerebellumGeo, brainMat);
    cerebellum.position.set(0, -0.8, -0.5);
    brainGroup.add(cerebellum);

    // Add Particles/Neurons around
    const particlesGeo = new THREE.BufferGeometry();
    const particleCount = 200;
    const posArray = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * 5;
    }
    particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    const particlesMat = new THREE.PointsMaterial({
        size: 0.05,
        color: 0x007bff,
        transparent: true,
        opacity: 0.6
    });
    particles = new THREE.Points(particlesGeo, particlesMat);
    brainGroup.add(particles);

    brainMesh = brainGroup;
    scene.add(brainGroup);

    // Hide Loader
    if (loader) loader.style.display = 'none';

    // Animation Loop
    animate();

    // Resize Handler
    window.addEventListener('resize', onWindowResize);
}

function animate() {
    requestAnimationFrame(animate);

    if (autoRotate && brainMesh) {
        brainMesh.rotation.y += 0.005;
        particles.rotation.y -= 0.002;
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('brain-container');
    if (!container || !camera || !renderer) return;

    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// --- Interaction Functions ---
window.rotateBrain = (direction) => {
    autoRotate = false;
    if (brainMesh) {
        if (direction === 'left') brainMesh.rotation.y += 0.5;
        if (direction === 'right') brainMesh.rotation.y -= 0.5;
    }
    // Resume auto-rotate after 2s
    setTimeout(() => autoRotate = true, 2000);
}

window.toggleAutoRotate = () => {
    autoRotate = !autoRotate;
}

window.updateBrainColor = (status) => {
    if (!brainMesh) return;

    let color = 0xe2e8f0; // Default
    if (status === 'cn') color = 0x28a745;
    if (status === 'mci') color = 0xffc107;
    if (status === 'ad') color = 0xdc3545;

    brainMesh.children.forEach(child => {
        if (child.isMesh) {
            child.material.color.setHex(color);
        }
    });

    if (particles) {
        particles.material.color.setHex(color);
    }
}
