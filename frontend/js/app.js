// Init Feather Icons
feather.replace();

// Elements mapping
const dropdownSourceSelect = document.getElementById('dropdown-source-select');
const fileUploaderInput = document.getElementById('file-uploader-input');
const videoUploaderInput = document.getElementById('video-uploader-input');
const btnReuploadFile = document.getElementById('btn-reupload-file');

const viewportPlaceholder = document.getElementById('viewport-placeholder');
const webcamRawFeed = document.getElementById('webcam-raw-feed');
const videoFileFeed = document.getElementById('video-file-feed');
const rawUploadedImage = document.getElementById('raw-uploaded-image');

const overlayDrawingCanvas = document.getElementById('overlay-drawing-canvas');
const viewportLoader = document.getElementById('viewport-loader');
const lblLoaderText = document.getElementById('lbl-loader-text');

// Custom Viewport controls mapping
const viewportControlsOverlay = document.getElementById('viewport-controls-overlay');
const btnToggleStreamPlay = document.getElementById('btn-toggle-stream-play');
const videoProgressContainer = document.getElementById('video-progress-container');
const videoProgressSlider = document.getElementById('video-progress-slider');
const lblVideoTime = document.getElementById('lbl-video-time');
const badgeStreamLive = document.getElementById('badge-stream-live');

// Control panel inputs
const sliderConfThresh = document.getElementById('slider-conf-thresh');
const lblSliderThreshVal = document.getElementById('lbl-slider-thresh-val');
const btnNetMnet = document.getElementById('btn-net-mnet');
const btnNetResnet = document.getElementById('btn-net-resnet');
const lblDetectorBackbone = document.getElementById('lbl-detector-backbone');

// Checkboxes
const chkBbox = document.getElementById('chk-bbox');
const chkMask = document.getElementById('chk-mask');
const chkLandm = document.getElementById('chk-landm');
const chkConf = document.getElementById('chk-conf');
const chkFaceid = document.getElementById('chk-faceid');
const chkFacequality = document.getElementById('chk-facequality');

// View Mode presets buttons
const btnModeNormal = document.getElementById('btn-mode-normal');
const btnModeDetect = document.getElementById('btn-mode-detect');
const btnModeSegment = document.getElementById('btn-mode-segment');

// Action Utilities buttons
const btnSaveFrame = document.getElementById('btn-save-frame');
const btnSaveCrops = document.getElementById('btn-save-crops');
const btnSnapshot = document.getElementById('btn-snapshot');
const btnExportResult = document.getElementById('btn-export-result');

// Telemetry labels
const lblTelemetryFaces = document.getElementById('lbl-telemetry-faces');
const lblTelemetryLatency = document.getElementById('lbl-telemetry-latency');
const lblTelemetryDensity = document.getElementById('lbl-telemetry-density');
const lblTelemetryFps = document.getElementById('lbl-telemetry-fps');

// Header stats
const lblHeaderFps = document.getElementById('lbl-header-fps');
const lblHeaderLoad = document.getElementById('lbl-header-load');
const headerDigitalClock = document.getElementById('header-digital-clock');
const alertFeedEl = document.getElementById('alert-feed-el');
const btnClearAlerts = document.getElementById('btn-clear-alerts');
const lblViewportSource = document.getElementById('lbl-viewport-source');
const lblHudResolution = document.getElementById('lbl-hud-resolution');
const mediaWrapperEl = document.getElementById('media-wrapper-el');
const galleryContainerEl = document.getElementById('gallery-container-el');
const lblGalleryCount = document.getElementById('lbl-gallery-count');

// Main client-side states
let facesData = [];
let currentViewMode = 'segment'; // Default mode
let selectedNetwork = 'mobile0.25';
let selectedFaceId = null;

let isWebcamStreaming = false;
let isVideoStreaming = false;
let isLoopProcessing = false;
let webcamMediaStream = null;

let lastFrameTimestamp = performance.now();
let streamFps = 0.0;
let latencyHistory = [];
let faceCountHistory = [];

// ==============================================================================
// 1. Audio Alerts Generator (Web Audio API)
// ==============================================================================
let audioCtx = null;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function playBeep(frequency = 440, duration = 0.15, type = 'sine') {
    try {
        initAudio();
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.value = frequency;
        gain.gain.setValueAtTime(0.05, audioCtx.currentTime); // Low volume
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.error("Audio Context playback failed: ", e);
    }
}

// ==============================================================================
// 2. ApexCharts Init and Config
// ==============================================================================
let donutChart = null;
let barChart = null;
let lineChart = null;

function initCharts() {
    // A. Quality Distribution Donut Chart
    const donutOptions = {
        series: [0, 0, 0, 0, 0],
        chart: {
            type: 'donut',
            height: 180,
            background: 'transparent',
            foreColor: '#9ca3af'
        },
        labels: ['Excellent', 'Good', 'Acceptable', 'Poor', 'Unusable'],
        colors: ['#05ffb0', '#00d2ff', '#3b82f6', '#ffb800', '#ff3b6b'],
        dataLabels: { enabled: false },
        legend: {
            show: true,
            position: 'bottom',
            fontSize: '10px',
            labels: { colors: '#9ca3af' }
        },
        stroke: { show: false },
        plotOptions: {
            pie: {
                donut: {
                    size: '70%',
                    background: 'transparent',
                }
            }
        }
    };
    donutChart = new ApexCharts(document.querySelector("#chart-quality-donut"), donutOptions);
    donutChart.render();

    // B. Head Pose Distribution Bar Chart
    const barOptions = {
        series: [{
            data: [0, 0, 0, 0, 0]
        }],
        chart: {
            type: 'bar',
            height: 150,
            toolbar: { show: false },
            background: 'transparent',
            foreColor: '#9ca3af'
        },
        colors: ['#00d2ff'],
        plotOptions: {
            bar: {
                borderRadius: 4,
                horizontal: true,
            }
        },
        dataLabels: { enabled: false },
        xaxis: {
            categories: ['Thẳng', 'Trái', 'Phải', 'Ngửa', 'Cúi'],
            labels: { style: { colors: '#9ca3af', fontSize: '9px' } }
        },
        yaxis: {
            labels: { style: { colors: '#9ca3af', fontSize: '10px' } }
        },
        grid: { borderColor: 'rgba(255,255,255,0.05)' }
    };
    barChart = new ApexCharts(document.querySelector("#chart-pose-bar"), barOptions);
    barChart.render();

    // C. Historical Targets Line Chart
    const lineOptions = {
        series: [{
            name: 'Mục tiêu',
            data: []
        }],
        chart: {
            type: 'line',
            height: 140,
            toolbar: { show: false },
            animations: { enabled: false },
            background: 'transparent',
            foreColor: '#9ca3af'
        },
        stroke: {
            curve: 'smooth',
            width: 2,
            colors: ['#00d2ff']
        },
        grid: { borderColor: 'rgba(255,255,255,0.05)' },
        xaxis: {
            labels: { show: false }
        },
        yaxis: {
            labels: { style: { colors: '#9ca3af', fontSize: '9px' } }
        },
        tooltip: { enabled: false }
    };
    lineChart = new ApexCharts(document.querySelector("#chart-history-line"), lineOptions);
    lineChart.render();
}

function updateCharts(faces) {
    // 1. Donut quality update
    let qCounts = { Excellent: 0, Good: 0, Acceptable: 0, Poor: 0, Unusable: 0 };
    faces.forEach(f => {
        if (qCounts[f.rating] !== undefined) qCounts[f.rating]++;
    });
    donutChart.updateSeries([qCounts.Excellent, qCounts.Good, qCounts.Acceptable, qCounts.Poor, qCounts.Unusable]);

    // 2. Bar Pose update
    let pCounts = { Normal: 0, 'Head Left': 0, 'Head Right': 0, 'Head Up': 0, 'Head Down': 0 };
    faces.forEach(f => {
        if (pCounts[f.pose] !== undefined) pCounts[f.pose]++;
    });
    barChart.updateSeries([{
        data: [pCounts.Normal, pCounts['Head Left'], pCounts['Head Right'], pCounts['Head Up'], pCounts['Head Down']]
    }]);

    // 3. Line history update
    faceCountHistory.push(faces.length);
    if (faceCountHistory.length > 20) {
        faceCountHistory.shift();
    }
    lineChart.updateSeries([{
        data: faceCountHistory
    }]);
}

// ==============================================================================
// 3. System Digital Clock & Core UI Event Bindings
// ==============================================================================
function updateClock() {
    const now = new Date();
    headerDigitalClock.textContent = now.toTimeString().split(' ')[0];
}
setInterval(updateClock, 1000);
updateClock();

btnClearAlerts.addEventListener('click', () => {
    alertFeedEl.innerHTML = `
        <div class="alert-feed-item normal">
            <span class="alert-time">${new Date().toTimeString().split(' ')[0]}</span>
            <span>Hệ thống giám sát: Đã dọn dẹp nhật ký.</span>
        </div>
    `;
});

function pushAlertLog(message, level = 'normal') {
    const timeStr = new Date().toTimeString().split(' ')[0];
    const alertItem = document.createElement('div');
    alertItem.className = `alert-feed-item ${level}`;
    alertItem.innerHTML = `
        <span class="alert-time">${timeStr}</span>
        <span>${message}</span>
    `;
    alertFeedEl.prepend(alertItem);

    if (alertFeedEl.children.length > 20) {
        alertFeedEl.removeChild(alertFeedEl.lastChild);
    }
    alertFeedEl.scrollTop = 0;
}

// Slider and settings binds
sliderConfThresh.addEventListener('input', (e) => {
    lblSliderThreshVal.textContent = parseFloat(e.target.value).toFixed(2);
    drawAllAnnotations();
});

btnNetMnet.addEventListener('click', () => {
    btnNetMnet.classList.add('active');
    btnNetResnet.classList.remove('active');
    selectedNetwork = 'mobile0.25';
    lblDetectorBackbone.textContent = "RetinaFace (MNet)";
    pushAlertLog("Cấu hình: Đổi sang mô hình MobileNet0.25 (Ưu tiên tốc độ).", "normal");
    playBeep(660, 0.1, 'sine');
});

btnNetResnet.addEventListener('click', () => {
    btnNetResnet.classList.add('active');
    btnNetMnet.classList.remove('active');
    selectedNetwork = 'resnet50';
    lblDetectorBackbone.textContent = "RetinaFace (ResNet50)";
    pushAlertLog("Cấu hình: Đổi sang mô hình ResNet50 (Ưu tiên chính xác).", "normal");
    playBeep(660, 0.1, 'sine');
});

// View mode selectors
btnModeNormal.addEventListener('click', () => {
    btnModeNormal.classList.add('active');
    btnModeDetect.classList.remove('active');
    btnModeSegment.classList.remove('active');
    currentViewMode = 'normal';
    pushAlertLog("Chế độ xem: Normal (Chỉ hiển thị camera sạch).", "normal");
    drawAllAnnotations();
});

btnModeDetect.addEventListener('click', () => {
    btnModeDetect.classList.add('active');
    btnModeNormal.classList.remove('active');
    btnModeSegment.classList.remove('active');
    currentViewMode = 'detect';
    pushAlertLog("Chế độ xem: Detection (Hiển thị Bbox và landmarks).", "normal");
    drawAllAnnotations();
});

btnModeSegment.addEventListener('click', () => {
    btnModeSegment.classList.add('active');
    btnModeNormal.classList.remove('active');
    btnModeDetect.classList.remove('active');
    currentViewMode = 'segment';
    pushAlertLog("Chế độ xem: Segmentation (Bật mặt nạ phân vùng U-Net).", "normal");
    drawAllAnnotations();
});

// Options checkboxes redraws
[chkBbox, chkMask, chkLandm, chkConf, chkFaceid, chkFacequality].forEach(chk => {
    chk.addEventListener('change', () => {
        btnModeNormal.classList.remove('active');
        btnModeDetect.classList.remove('active');
        btnModeSegment.classList.remove('active');
        currentViewMode = 'custom';
        drawAllAnnotations();
    });
});

// Align canvas layout dimensions
function matchCanvasToMedia(mediaElement) {
    if (!mediaElement || mediaElement.style.display === 'none') return;
    const w = mediaElement.tagName === 'VIDEO' ? mediaElement.videoWidth : mediaElement.naturalWidth;
    const h = mediaElement.tagName === 'VIDEO' ? mediaElement.videoHeight : mediaElement.naturalHeight;

    if (overlayDrawingCanvas.width !== w || overlayDrawingCanvas.height !== h) {
        overlayDrawingCanvas.width = w;
        overlayDrawingCanvas.height = h;
    }
}

// ==============================================================================
// 4. Drawing Canvas Core (Vector HUD Annotation Engine)
// ==============================================================================
function drawAllAnnotations() {
    const ctx = overlayDrawingCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayDrawingCanvas.width, overlayDrawingCanvas.height);

    if (currentViewMode === 'normal') {
        return;
    }

    const drawBbox = (currentViewMode === 'detect' || currentViewMode === 'segment' || (currentViewMode === 'custom' && chkBbox.checked));
    const drawMask = (currentViewMode === 'segment' || (currentViewMode === 'custom' && chkMask.checked));
    const drawLandm = (currentViewMode === 'detect' || currentViewMode === 'segment' || (currentViewMode === 'custom' && chkLandm.checked));
    const drawConf = (currentViewMode === 'detect' || (currentViewMode === 'custom' && chkConf.checked));
    const drawFaceId = (currentViewMode === 'custom' && chkFaceid.checked);
    const drawFaceQuality = (currentViewMode === 'detect' || currentViewMode === 'segment' || (currentViewMode === 'custom' && chkFacequality.checked));

    facesData.forEach(face => {
        const [x1, y1, x2, y2] = face.box;
        const rw = x2 - x1;
        const rh = y2 - y1;

        // A. Draw Mask Overlay
        if (drawMask && face.face_png_alpha) {
            const maskImg = new Image();
            maskImg.src = face.face_png_alpha;
            maskImg.onload = function () {
                const pad = Math.min(15, Math.floor(Math.max(rw, rh) * 0.1));
                const px1 = Math.max(0, x1 - pad);
                const py1 = Math.max(0, y1 - pad);
                const px2 = Math.min(overlayDrawingCanvas.width, x2 + pad);
                const py2 = Math.min(overlayDrawingCanvas.height, y2 + pad);
                const pw = px2 - px1;
                const ph = py2 - py1;

                // Cyan mask translucent fill
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = maskImg.width;
                maskCanvas.height = maskImg.height;
                const mCtx = maskCanvas.getContext('2d');
                mCtx.drawImage(maskImg, 0, 0);
                mCtx.globalCompositeOperation = 'source-in';
                mCtx.fillStyle = 'rgba(0, 210, 255, 0.25)';
                mCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

                ctx.drawImage(maskCanvas, px1, py1, pw, ph);

                // Cyan border contours
                const outlineCanvas = document.createElement('canvas');
                outlineCanvas.width = maskImg.width;
                outlineCanvas.height = maskImg.height;
                const oCtx = outlineCanvas.getContext('2d');

                const cyanMask = document.createElement('canvas');
                cyanMask.width = maskImg.width;
                cyanMask.height = maskImg.height;
                const cCtx = cyanMask.getContext('2d');
                cCtx.drawImage(maskImg, 0, 0);
                cCtx.globalCompositeOperation = 'source-in';
                cCtx.fillStyle = '#00d2ff';
                cCtx.fillRect(0, 0, cyanMask.width, cyanMask.height);

                oCtx.drawImage(cyanMask, -1, 0);
                oCtx.drawImage(cyanMask, 1, 0);
                oCtx.drawImage(cyanMask, 0, -1);
                oCtx.drawImage(cyanMask, 0, 1);

                oCtx.globalCompositeOperation = 'destination-out';
                oCtx.drawImage(maskImg, 0, 0);

                ctx.drawImage(outlineCanvas, px1, py1, pw, ph);
            };
        }

        // B. Bounding Boxes
        if (drawBbox) {
            const isSelected = (face.id === selectedFaceId);
            ctx.strokeStyle = isSelected ? '#ff3b6b' : '#00d2ff';
            ctx.lineWidth = isSelected ? 4.5 : 2.5;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(x1, y1, rw, rh, 6);
            } else {
                ctx.rect(x1, y1, rw, rh);
            }
            ctx.stroke();
        }

        // C. Landmarks (Neon emerald with glowing white edge)
        if (drawLandm && face.landmarks) {
            for (let i = 0; i < 5; i++) {
                const lx = face.landmarks[2 * i];
                const ly = face.landmarks[2 * i + 1];

                ctx.fillStyle = '#05ffb0';
                ctx.beginPath();
                ctx.arc(lx, ly, 4, 0, 2 * Math.PI);
                ctx.fill();

                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

        // D. HUD Overlay text boxes
        if (drawConf || drawFaceId || drawFaceQuality) {
            const txtSize = Math.max(10, Math.round(overlayDrawingCanvas.width / 75));
            ctx.font = `700 ${txtSize}px 'Share Tech Mono', monospace`;

            if (drawFaceQuality && face.quality_score !== undefined) {
                const lines = [
                    `TARGET #${face.id.toString().padStart(2, '0')}`,
                    `Conf: ${(face.confidence * 100).toFixed(0)}%`,
                    `Vis: ${face.visibility}%`,
                    `Qual: ${face.quality_score}%`,
                    `Pose: ${face.pose}`,
                    `Stat: ${face.status}`
                ];

                let maxW = 0;
                lines.forEach(l => {
                    const w = ctx.measureText(l).width;
                    if (w > maxW) maxW = w;
                });

                const lineH = txtSize + 4;
                const boxW = maxW + 12;
                const boxH = lines.length * lineH + 10;

                ctx.fillStyle = 'rgba(10, 15, 25, 0.85)';
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(x1, y1 - boxH - 4, boxW, boxH, 6);
                } else {
                    ctx.rect(x1, y1 - boxH - 4, boxW, boxH);
                }
                ctx.fill();

                const isWarning = (face.status !== 'Normal');
                ctx.strokeStyle = isWarning ? '#ffb800' : '#05ffb0';
                ctx.lineWidth = 1.2;
                ctx.stroke();

                lines.forEach((l, idx) => {
                    if (idx === 0) {
                        ctx.fillStyle = '#00d2ff';
                    } else if (l.startsWith('Stat:')) {
                        ctx.fillStyle = isWarning ? '#ffb800' : '#05ffb0';
                    } else {
                        ctx.fillStyle = '#ffffff';
                    }
                    ctx.fillText(l, x1 + 6, y1 - boxH - 4 + 8 + (idx + 1) * lineH - 2);
                });
            } else if (drawConf || drawFaceId) {
                let label = '';
                if (drawFaceId) label += `TARGET #${face.id.toString().padStart(2, '0')} `;
                if (drawConf) label += `${(face.confidence * 100).toFixed(0)}%`;

                const txtW = ctx.measureText(label).width;
                const txtH = txtSize;

                ctx.fillStyle = 'rgba(10, 15, 25, 0.75)';
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(x1, y1 - txtH - 6, txtW + 8, txtH + 6, 4);
                } else {
                    ctx.rect(x1, y1 - txtH - 6, txtW + 8, txtH + 6);
                }
                ctx.fill();

                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, x1 + 4, y1 - 4);
            }
        }
    });
}

function getFaceQuality(face) {
    if (face.rating === 'Excellent') return { label: 'Excellent', level: 'success' };
    if (face.rating === 'Good') return { label: 'Good', level: 'success' };
    if (face.rating === 'Acceptable') return { label: 'Acceptable', level: 'success' };
    if (face.rating === 'Poor') return { label: 'Poor', level: 'warning' };
    return { label: 'Unusable', level: 'danger' };
}

function highlightFace(faceId) {
    selectedFaceId = faceId;
    drawAllAnnotations();
    playBeep(980, 0.1, 'triangle');
    pushAlertLog(`Hành động: Chọn kiểm tra mục tiêu #${faceId.toString().padStart(2, '0')} trên camera.`, "normal");

    setTimeout(() => {
        if (selectedFaceId === faceId) {
            selectedFaceId = null;
            drawAllAnnotations();
        }
    }, 1500);
}

// ==============================================================================
// 5. API Response Handler & Alert Checks
// ==============================================================================
function handleDetectionResults(result, latencyMs) {
    facesData = result.faces || [];
    const count = result.face_count || 0;

    lblTelemetryFaces.textContent = count;
    lblTelemetryLatency.textContent = `${latencyMs} ms`;
    lblHeaderLoad.textContent = `${latencyMs}ms`;

    // Crowd Alarm checks
    let densityPercentage = Math.min(100, Math.round((count / 12) * 100));
    let densityLevel = 'LOW';
    let densityColor = 'var(--success)';

    if (count >= 8) {
        densityLevel = 'HIGH';
        densityColor = 'var(--danger)';
        pushAlertLog(`CẢNH BÁO ĐÁM ĐÔNG: Mật độ khuôn mặt cao (${count} đối tượng).`, "danger");
        playBeep(440, 0.35, 'sawtooth');
    } else if (count >= 4) {
        densityLevel = 'MEDIUM';
        densityColor = 'var(--warning)';
        pushAlertLog(`Cảnh báo: Mật độ đám đông tăng (${count} đối tượng).`, "warning");
        playBeep(880, 0.15, 'triangle');
    }
    lblTelemetryDensity.textContent = `${densityLevel} (${densityPercentage}%)`;
    lblTelemetryDensity.style.color = densityColor;

    // Checks for face warnings
    facesData.forEach(face => {
        const idStr = face.id.toString().padStart(2, '0');
        if (face.status === 'Face Too Small') {
            pushAlertLog(`Mục tiêu #${idStr}: Kích thước mặt quá nhỏ so với tiêu chuẩn.`, "warning");
        } else if (face.status === 'Face Occluded') {
            pushAlertLog(`CẢNH BÁO MỤC TIÊU #${idStr}: Vùng mặt bị che khuất / đeo mặt nạ!`, "danger");
            playBeep(520, 0.25, 'triangle');
        } else if (face.status === 'Low Confidence') {
            pushAlertLog(`Mục tiêu #${idStr}: Độ tin cậy nhận diện thấp.`, "warning");
        }
    });

    // Enabled capture buttons
    const disableButtons = facesData.length === 0;
    btnSaveFrame.disabled = disableButtons;
    btnSaveCrops.disabled = disableButtons;
    btnSnapshot.disabled = disableButtons;
    btnExportResult.disabled = disableButtons;

    // Render gallery thumbnails
    lblGalleryCount.textContent = `TỔNG SỐ: ${count} MỤC TIÊU`;
    galleryContainerEl.innerHTML = '';

    if (facesData.length === 0) {
        galleryContainerEl.innerHTML = `
            <div style="color: var(--text-muted); font-size: 0.75rem; width: 100%; text-align: center; font-style: italic;">Không có mục tiêu nào trong khung nhìn</div>
        `;
    } else {
        facesData.forEach(face => {
            const qual = getFaceQuality(face);
            const card = document.createElement('div');
            card.className = 'face-gallery-card';
            card.title = `Nhấn để đánh dấu Mục tiêu #${face.id}`;
            card.addEventListener('click', () => {
                highlightFace(face.id);
            });
            card.innerHTML = `
                <div class="face-gallery-thumbnail-box">
                    <img src="${face.face_png_alpha || ''}" alt="Face">
                </div>
                <span class="face-gallery-id">TARGET #${face.id.toString().padStart(2, '0')}</span>
                <div class="face-gallery-status-badge ${qual.level}">${qual.label}</div>
                <div class="face-gallery-metrics">
                    <div class="metric-progress-row" title="Face Quality: ${face.quality_score}%">
                        <span class="metric-label">QLY</span>
                        <div class="metric-track">
                            <div class="metric-bar qly-bar" style="width: ${face.quality_score}%; background-color: var(--${qual.level === 'danger' ? 'danger' : qual.level === 'warning' ? 'warning' : 'success'})"></div>
                        </div>
                    </div>
                    <div class="metric-progress-row" title="Face Visibility: ${face.visibility}%">
                        <span class="metric-label">VIS</span>
                        <div class="metric-track">
                            <div class="metric-bar vis-bar" style="width: ${face.visibility}%; background-color: var(--primary)"></div>
                        </div>
                    </div>
                </div>
            `;
            galleryContainerEl.appendChild(card);
        });
    }

    // Canvas annotations redraw
    drawAllAnnotations();

    // Chart analytics redraw
    updateCharts(facesData);
}

// ==============================================================================
// 6. Webcam & Video Loops Engine
// ==============================================================================
function getActiveMediaElement() {
    const source = dropdownSourceSelect.value;
    if (source === 'webcam') return webcamRawFeed;
    if (source === 'video') return videoFileFeed;
    if (source === 'upload') return rawUploadedImage;
    return null;
}

dropdownSourceSelect.addEventListener('change', (e) => {
    const source = e.target.value;
    stopWebcamStream();
    stopVideoFileStream();
    resetViewport();
    initAudio();

    if (source === 'upload') {
        lblViewportSource.textContent = "NGUỒN: TẢI ẢNH LÊN";
        fileUploaderInput.click();
        viewportControlsOverlay.style.display = 'none';
        btnReuploadFile.style.display = 'block';
    } else if (source === 'video') {
        lblViewportSource.textContent = "NGUỒN: VIDEO FILE FEED";
        videoUploaderInput.click();
        viewportControlsOverlay.style.display = 'flex';
        videoProgressContainer.style.display = 'flex';
        badgeStreamLive.style.display = 'none';
        btnToggleStreamPlay.innerHTML = `<i data-feather="pause"></i>`;
        feather.replace();
        btnReuploadFile.style.display = 'block';
    } else if (source === 'webcam') {
        lblViewportSource.textContent = "NGUỒN: WEBCAM TRỰC TIẾP";
        startWebcamStream();
        viewportControlsOverlay.style.display = 'flex';
        videoProgressContainer.style.display = 'none';
        badgeStreamLive.style.display = 'flex';
        btnToggleStreamPlay.innerHTML = `<i data-feather="pause"></i>`;
        feather.replace();
        btnReuploadFile.style.display = 'none';
    } else {
        lblViewportSource.textContent = "NGUỒN: ĐANG CHỜ";
        viewportControlsOverlay.style.display = 'none';
        btnReuploadFile.style.display = 'none';
    }
});

function resetViewport() {
    if (viewportControlsOverlay) {
        viewportControlsOverlay.style.display = 'none';
    }
    if (btnReuploadFile) {
        btnReuploadFile.style.display = 'none';
    }
    rawUploadedImage.style.display = 'none';
    webcamRawFeed.style.display = 'none';
    videoFileFeed.style.display = 'none';
    mediaWrapperEl.style.display = 'none';
    viewportPlaceholder.style.display = 'flex';

    lblTelemetryFaces.textContent = '0';
    lblTelemetryLatency.textContent = '0 ms';
    lblTelemetryDensity.textContent = 'LOW (0%)';
    lblTelemetryDensity.style.color = 'var(--success)';
    lblTelemetryFps.textContent = '0.0';

    lblHeaderFps.textContent = '0.0';
    lblHeaderLoad.textContent = 'Đang chờ';
    lblHudResolution.textContent = '640x480';

    facesData = [];
    overlayDrawingCanvas.getContext('2d').clearRect(0, 0, overlayDrawingCanvas.width, overlayDrawingCanvas.height);

    galleryContainerEl.innerHTML = `
        <div style="color: var(--text-muted); font-size: 0.75rem; width: 100%; text-align: center; font-style: italic;">Không có mục tiêu nào trong khung nhìn</div>
    `;
    lblGalleryCount.textContent = 'TỔNG SỐ: 0 MỤC TIÊU';

    btnSaveFrame.disabled = true;
    btnSaveCrops.disabled = true;
    btnSnapshot.disabled = true;
    btnExportResult.disabled = true;

    // Reset charts
    if (donutChart) {
        donutChart.updateSeries([0, 0, 0, 0, 0]);
        barChart.updateSeries([{ data: [0, 0, 0, 0, 0] }]);
        lineChart.updateSeries([{ data: [] }]);
        faceCountHistory = [];
    }
}

// Upload Static Frame
fileUploaderInput.addEventListener('change', async (e) => {
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];

    pushAlertLog(`Hệ thống: Bắt đầu tải ảnh phân tích: ${file.name}.`, "normal");

    viewportLoader.style.display = 'flex';
    lblLoaderText.textContent = "Đang gửi ảnh phân tích...";

    const formData = new FormData();
    formData.append('image', file);
    formData.append('network', selectedNetwork);
    formData.append('threshold', sliderConfThresh.value);
    formData.append('draw_mask', 'true');
    formData.append('upscale', 'true');

    const tStart = performance.now();

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (response.ok) {
            rawUploadedImage.src = URL.createObjectURL(file);
            rawUploadedImage.style.display = 'block';
            mediaWrapperEl.style.display = 'flex';
            viewportPlaceholder.style.display = 'none';

            const tEnd = performance.now();
            const latency = Math.round(tEnd - tStart);

            rawUploadedImage.onload = function () {
                lblHudResolution.textContent = `${this.naturalWidth}x${this.naturalHeight}`;
                matchCanvasToMedia(rawUploadedImage);
                handleDetectionResults(result, latency);
            };
            pushAlertLog("Hệ thống: Nhận diện ảnh tĩnh thành công.", "success");
        } else {
            pushAlertLog(`Lỗi xử lý: ${result.detail || "Không rõ nguyên nhân."}`, "danger");
        }
    } catch (err) {
        pushAlertLog("Lỗi kết nối: Không thể kết nối với API AI.", "danger");
        console.error(err);
    } finally {
        viewportLoader.style.display = 'none';
    }
});

// Webcam Surveillance Stream
async function startWebcamStream() {
    viewportLoader.style.display = 'flex';
    lblLoaderText.textContent = "Khởi động luồng Webcam...";

    try {
        webcamMediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        });

        webcamRawFeed.srcObject = webcamMediaStream;
        webcamRawFeed.style.display = 'block';
        rawUploadedImage.style.display = 'none';
        videoFileFeed.style.display = 'none';
        mediaWrapperEl.style.display = 'flex';
        viewportPlaceholder.style.display = 'none';

        isWebcamStreaming = true;

        webcamRawFeed.onloadedmetadata = function () {
            lblHudResolution.textContent = `${this.videoWidth}x${this.videoHeight}`;
            matchCanvasToMedia(webcamRawFeed);
        };

        pushAlertLog("Hệ thống: Kết nối Webcam USB an ninh thành công.", "normal");
        requestAnimationFrame(processWebcamSurveillanceLoop);
    } catch (err) {
        pushAlertLog("Lỗi hệ thống: Từ chối quyền truy cập Webcam.", "danger");
        dropdownSourceSelect.value = 'idle';
        lblViewportSource.textContent = "NGUỒN: ĐANG CHỜ";
        resetViewport();
        console.error(err);
    } finally {
        viewportLoader.style.display = 'none';
    }
}

function stopWebcamStream() {
    isWebcamStreaming = false;
    isLoopProcessing = false;
    if (webcamMediaStream) {
        webcamMediaStream.getTracks().forEach(t => t.stop());
        webcamMediaStream = null;
    }
    webcamRawFeed.srcObject = null;
    webcamRawFeed.style.display = 'none';
}

async function processWebcamSurveillanceLoop() {
    if (!isWebcamStreaming) return;
    if (webcamRawFeed.videoWidth === 0 || webcamRawFeed.videoHeight === 0) {
        requestAnimationFrame(processWebcamSurveillanceLoop);
        return;
    }
    if (webcamRawFeed.paused) {
        requestAnimationFrame(processWebcamSurveillanceLoop);
        return;
    }
    if (isLoopProcessing) {
        requestAnimationFrame(processWebcamSurveillanceLoop);
        return;
    }

    isLoopProcessing = true;
    lblHudResolution.textContent = `${webcamRawFeed.videoWidth}x${webcamRawFeed.videoHeight}`;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = webcamRawFeed.videoWidth;
    offCanvas.height = webcamRawFeed.videoHeight;
    offCanvas.getContext('2d').drawImage(webcamRawFeed, 0, 0);

    offCanvas.toBlob(async (blob) => {
        if (!blob) {
            isLoopProcessing = false;
            if (isWebcamStreaming) requestAnimationFrame(processWebcamSurveillanceLoop);
            return;
        }

        const formData = new FormData();
        formData.append('image', blob, 'webcam.jpg');
        formData.append('network', selectedNetwork);
        formData.append('threshold', sliderConfThresh.value);
        formData.append('draw_mask', 'true');
        formData.append('upscale', 'false');

        const tStart = performance.now();

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (response.ok && isWebcamStreaming) {
                const result = await response.json();
                const tEnd = performance.now();
                const latency = Math.round(tEnd - tStart);

                const currentStamp = performance.now();
                const d = currentStamp - lastFrameTimestamp;
                lastFrameTimestamp = currentStamp;
                streamFps = (1000 / d).toFixed(1);
                lblTelemetryFps.textContent = streamFps;
                lblHeaderFps.textContent = streamFps;

                matchCanvasToMedia(webcamRawFeed);
                handleDetectionResults(result, latency);
            }
        } catch (err) {
            console.error("Webcam loop frame fetch error:", err);
        } finally {
            isLoopProcessing = false;
            if (isWebcamStreaming) {
                setTimeout(() => {
                    requestAnimationFrame(processWebcamSurveillanceLoop);
                }, 80); // Restricting FPS to lower local CPU load
            }
        }
    }, 'image/jpeg', 0.85);
}

// Local Video Stream Loop
videoUploaderInput.addEventListener('change', (e) => {
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];

    pushAlertLog(`Hệ thống: Nạp tệp video nội bộ: ${file.name}.`, "normal");
    videoFileFeed.src = URL.createObjectURL(file);

    webcamRawFeed.style.display = 'none';
    rawUploadedImage.style.display = 'none';
    videoFileFeed.style.display = 'block';
    mediaWrapperEl.style.display = 'flex';
    viewportPlaceholder.style.display = 'none';

    videoFileFeed.onloadedmetadata = function () {
        lblHudResolution.textContent = `${this.videoWidth}x${this.videoHeight}`;
        matchCanvasToMedia(videoFileFeed);
    };

    isVideoStreaming = true;
    requestAnimationFrame(processVideoFileSurveillanceLoop);
});

function stopVideoFileStream() {
    isVideoStreaming = false;
    isLoopProcessing = false;
    videoFileFeed.src = "";
    videoFileFeed.style.display = 'none';
}

async function processVideoFileSurveillanceLoop() {
    if (!isVideoStreaming) return;
    if (videoFileFeed.paused || videoFileFeed.ended) {
        requestAnimationFrame(processVideoFileSurveillanceLoop);
        return;
    }
    if (videoFileFeed.videoWidth === 0 || videoFileFeed.videoHeight === 0) {
        requestAnimationFrame(processVideoFileSurveillanceLoop);
        return;
    }
    if (isLoopProcessing) {
        requestAnimationFrame(processVideoFileSurveillanceLoop);
        return;
    }

    isLoopProcessing = true;
    lblHudResolution.textContent = `${videoFileFeed.videoWidth}x${videoFileFeed.videoHeight}`;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = videoFileFeed.videoWidth;
    offCanvas.height = videoFileFeed.videoHeight;
    offCanvas.getContext('2d').drawImage(videoFileFeed, 0, 0);

    offCanvas.toBlob(async (blob) => {
        if (!blob) {
            isLoopProcessing = false;
            if (isVideoStreaming) requestAnimationFrame(processVideoFileSurveillanceLoop);
            return;
        }

        const formData = new FormData();
        formData.append('image', blob, 'video_frame.jpg');
        formData.append('network', selectedNetwork);
        formData.append('threshold', sliderConfThresh.value);
        formData.append('draw_mask', 'true');
        formData.append('upscale', 'false');

        const tStart = performance.now();

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (response.ok && isVideoStreaming) {
                const result = await response.json();
                const tEnd = performance.now();
                const latency = Math.round(tEnd - tStart);

                const currentStamp = performance.now();
                const d = currentStamp - lastFrameTimestamp;
                lastFrameTimestamp = currentStamp;
                streamFps = (1000 / d).toFixed(1);
                lblTelemetryFps.textContent = streamFps;
                lblHeaderFps.textContent = streamFps;

                matchCanvasToMedia(videoFileFeed);
                handleDetectionResults(result, latency);
            }
        } catch (err) {
            console.error("Video loop frame fetch error:", err);
        } finally {
            isLoopProcessing = false;
            if (isVideoStreaming) {
                setTimeout(() => {
                    requestAnimationFrame(processVideoFileSurveillanceLoop);
                }, 100);
            }
        }
    }, 'image/jpeg', 0.85);
}

// Single seeked processing
videoFileFeed.addEventListener('seeked', async () => {
    if (videoFileFeed.videoWidth === 0 || videoFileFeed.videoHeight === 0) return;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = videoFileFeed.videoWidth;
    offCanvas.height = videoFileFeed.videoHeight;
    offCanvas.getContext('2d').drawImage(videoFileFeed, 0, 0);

    const tStart = performance.now();

    offCanvas.toBlob(async (blob) => {
        if (!blob) return;
        const formData = new FormData();
        formData.append('image', blob, 'seek_frame.jpg');
        formData.append('network', selectedNetwork);
        formData.append('threshold', sliderConfThresh.value);
        formData.append('draw_mask', 'true');
        formData.append('upscale', 'false');

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                const result = await response.json();
                const tEnd = performance.now();
                const latency = Math.round(tEnd - tStart);
                matchCanvasToMedia(videoFileFeed);
                handleDetectionResults(result, latency);
            }
        } catch (err) {
            console.error("Single seek frame error:", err);
        }
    }, 'image/jpeg', 0.85);
});

// ==============================================================================
// 7. File Export & Evidence Capture Utilities
// ==============================================================================
function downloadURI(uri, name) {
    const link = document.createElement('a');
    link.href = uri;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Bbox-drawn view frame export
btnSaveFrame.addEventListener('click', () => {
    const media = getActiveMediaElement();
    if (!media) return;

    const w = media.tagName === 'VIDEO' ? media.videoWidth : media.naturalWidth;
    const h = media.tagName === 'VIDEO' ? media.videoHeight : media.naturalHeight;

    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = w;
    frameCanvas.height = h;
    const fCtx = frameCanvas.getContext('2d');

    fCtx.drawImage(media, 0, 0);
    fCtx.drawImage(overlayDrawingCanvas, 0, 0);

    const timestamp = new Date().toISOString().replace(/T/, '_').replace(/\..+/, '').replace(/:/g, '');
    downloadURI(frameCanvas.toDataURL('image/jpeg', 0.95), `CCTV_frame_${timestamp}.jpg`);
    pushAlertLog(`Công cụ: Đã tải xuống khung ảnh chụp frame_${timestamp}.jpg`, "normal");
});

// Save all face crops
btnSaveCrops.addEventListener('click', () => {
    if (facesData.length === 0) return;
    pushAlertLog(`Công cụ: Đang lưu ${facesData.length} ảnh chân dung trong suốt...`, "normal");

    facesData.forEach((face, idx) => {
        setTimeout(() => {
            downloadURI(face.face_png_alpha, `face_target_${face.id.toString().padStart(2, '0')}.png`);
        }, idx * 250); // Offset delay to bypass browser bulk block
    });
});

// Snapshot: saves frame, crop, mask and JSON logs for target face
btnSnapshot.addEventListener('click', () => {
    const media = getActiveMediaElement();
    if (!media || facesData.length === 0) return;

    const targetFace = facesData.find(f => f.id === selectedFaceId) || facesData[0];
    const targetId = targetFace.id;

    const w = media.tagName === 'VIDEO' ? media.videoWidth : media.naturalWidth;
    const h = media.tagName === 'VIDEO' ? media.videoHeight : media.naturalHeight;

    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];
    const timeStr = now.toTimeString().split(' ')[0];
    const timestamp = `${dateStr.replace(/-/g, '')}_${timeStr.replace(/:/g, '')}`;

    // 1. Full image
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = w;
    snapCanvas.height = h;
    snapCanvas.getContext('2d').drawImage(media, 0, 0);
    downloadURI(snapCanvas.toDataURL('image/jpeg', 0.92), `CCTV_BằngChứng_KhungHình_ID${targetId}_${timestamp}.jpg`);

    // 2. Face crop
    if (targetFace.face_png_alpha) {
        downloadURI(targetFace.face_png_alpha, `CCTV_BằngChứng_MặtCắt_ID${targetId}_${timestamp}.png`);

        // 3. Binary mask
        const cropImg = new Image();
        cropImg.src = targetFace.face_png_alpha;
        cropImg.onload = function () {
            const maskCropCanvas = document.createElement('canvas');
            maskCropCanvas.width = cropImg.width;
            maskCropCanvas.height = cropImg.height;
            const mcCtx = maskCropCanvas.getContext('2d');

            mcCtx.fillStyle = '#000000';
            mcCtx.fillRect(0, 0, maskCropCanvas.width, maskCropCanvas.height);

            const silhouetteCanvas = document.createElement('canvas');
            silhouetteCanvas.width = cropImg.width;
            silhouetteCanvas.height = cropImg.height;
            const sCtx = silhouetteCanvas.getContext('2d');
            sCtx.drawImage(cropImg, 0, 0);
            sCtx.globalCompositeOperation = 'source-in';
            sCtx.fillStyle = '#ffffff';
            sCtx.fillRect(0, 0, silhouetteCanvas.width, silhouetteCanvas.height);

            mcCtx.drawImage(silhouetteCanvas, 0, 0);
            downloadURI(maskCropCanvas.toDataURL('image/png'), `CCTV_BằngChứng_MặtNạ_ID${targetId}_${timestamp}.png`);
        };
    }

    // 4. JSON metadata log
    const jsonMeta = {
        "time": `${dateStr} ${timeStr}`,
        "face_id": targetId,
        "detection_confidence": parseFloat(targetFace.confidence.toFixed(2)),
        "visibility_percent": targetFace.visibility,
        "quality_score": targetFace.quality_score,
        "status_alert": targetFace.status,
        "pose_estimation": targetFace.pose,
        "rating_decision": targetFace.rating
    };
    const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(jsonMeta, null, 4));
    downloadURI(jsonStr, `CCTV_BằngChứng_Log_ID${targetId}_${timestamp}.json`);

    pushAlertLog(`Hệ thống: Đã lưu hồ sơ bằng chứng đầy đủ cho Mục tiêu #${targetId.toString().padStart(2, '0')}.`, "success");
});

// Bulk data export (JSON coordinates + compiled bulk mask)
btnExportResult.addEventListener('click', () => {
    const media = getActiveMediaElement();
    if (!media || facesData.length === 0) return;

    const w = media.tagName === 'VIDEO' ? media.videoWidth : media.naturalWidth;
    const h = media.tagName === 'VIDEO' ? media.videoHeight : media.naturalHeight;

    pushAlertLog("Công cụ: Đang biên dịch gói kết quả dữ liệu xuất khẩu...", "normal");

    // 1. Export JSON Data
    const exportData = {
        scan_timestamp: new Date().toISOString(),
        face_count: facesData.length,
        targets: facesData.map(f => ({
            id: f.id,
            bbox: f.box,
            confidence: parseFloat(f.confidence.toFixed(4)),
            landmarks: f.landmarks,
            quality: f.quality_score,
            pose: f.pose,
            alert_status: f.status
        }))
    };
    const jsonBlob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const jsonUrl = URL.createObjectURL(jsonBlob);
    downloadURI(jsonUrl, `CCTV_dữ_liệu_${Date.now()}.json`);
    URL.revokeObjectURL(jsonUrl);

    // 2. Compile and Export Binary Mask
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = w;
    maskCanvas.height = h;
    const mCtx = maskCanvas.getContext('2d');
    mCtx.fillStyle = '#000000';
    mCtx.fillRect(0, 0, w, h);

    let loadedCount = 0;
    facesData.forEach(face => {
        const cropImg = new Image();
        cropImg.src = face.face_png_alpha;
        cropImg.onload = function () {
            const tempC = document.createElement('canvas');
            tempC.width = cropImg.width;
            tempC.height = cropImg.height;
            const tempCtx = tempC.getContext('2d');
            tempCtx.drawImage(cropImg, 0, 0);

            tempCtx.globalCompositeOperation = 'source-in';
            tempCtx.fillStyle = '#ffffff';
            tempCtx.fillRect(0, 0, tempC.width, tempC.height);

            const [x1, y1, x2, y2] = face.box;
            const rw = x2 - x1;
            const rh = y2 - y1;
            const pad = Math.min(15, Math.floor(Math.max(rw, rh) * 0.1));
            const px1 = Math.max(0, x1 - pad);
            const py1 = Math.max(0, y1 - pad);
            const px2 = Math.min(w, x2 + pad);
            const py2 = Math.min(h, y2 + pad);

            mCtx.drawImage(tempC, px1, py1, px2 - px1, py2 - py1);

            loadedCount++;
            if (loadedCount === facesData.length) {
                downloadURI(maskCanvas.toDataURL('image/png'), `CCTV_toàn_bộ_mặt_nạ_${Date.now()}.png`);
                pushAlertLog("Công cụ: Gói dữ liệu xuất khẩu thành công (JSON + Mask).", "success");
            }
        };
    });
});

// ==============================================================================
// 8. Aspect-Ratio Click / Hover Mapping on Canvas & Custom Viewport Controls
// ==============================================================================

// Resolve mouse coordinate inside canvas under object-fit: contain
function getCanvasCoords(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    const imageRatio = canvas.width / canvas.height;
    const elementRatio = rect.width / rect.height;
    
    let renderedWidth, renderedHeight, renderedLeft, renderedTop;
    
    if (elementRatio > imageRatio) {
        // Height-constrained (vertical letterbox)
        renderedHeight = rect.height;
        renderedWidth = rect.height * imageRatio;
        renderedLeft = rect.left + (rect.width - renderedWidth) / 2;
        renderedTop = rect.top;
    } else {
        // Width-constrained (horizontal letterbox)
        renderedWidth = rect.width;
        renderedHeight = rect.width / imageRatio;
        renderedTop = rect.top + (rect.height - renderedHeight) / 2;
        renderedLeft = rect.left;
    }
    
    const relativeX = event.clientX - renderedLeft;
    const relativeY = event.clientY - renderedTop;
    
    if (relativeX >= 0 && relativeX <= renderedWidth && relativeY >= 0 && relativeY <= renderedHeight) {
        return {
            x: relativeX * (canvas.width / renderedWidth),
            y: relativeY * (canvas.height / renderedHeight)
        };
    }
    return null;
}

// Bounding Box Hover feedback
overlayDrawingCanvas.addEventListener('mousemove', (e) => {
    if (facesData.length === 0) return;
    const coords = getCanvasCoords(e, overlayDrawingCanvas);
    let hoverOnBox = false;
    
    if (coords) {
        for (const face of facesData) {
            const [x1, y1, x2, y2] = face.box;
            if (coords.x >= x1 && coords.x <= x2 && coords.y >= y1 && coords.y <= y2) {
                hoverOnBox = true;
                break;
            }
        }
    }
    overlayDrawingCanvas.style.cursor = hoverOnBox ? 'pointer' : 'default';
});

// Bounding Box Click targeting focus
overlayDrawingCanvas.addEventListener('click', (e) => {
    if (facesData.length === 0) return;
    const coords = getCanvasCoords(e, overlayDrawingCanvas);
    if (!coords) return;
    
    for (const face of facesData) {
        const [x1, y1, x2, y2] = face.box;
        if (coords.x >= x1 && coords.x <= x2 && coords.y >= y1 && coords.y <= y2) {
            // Focus target face
            highlightFace(face.id);
            
            // Smoothly scroll corresponding gallery card into view
            const targetCards = galleryContainerEl.querySelectorAll('.face-gallery-card');
            for (const card of targetCards) {
                const cardIdText = card.querySelector('.face-gallery-id')?.textContent;
                if (cardIdText && cardIdText.includes(`#${face.id.toString().padStart(2, '0')}`)) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                    break;
                }
            }
            break;
        }
    }
});

// Custom Video controls play/pause dynamic icons
function setPlayPauseIcon(isPaused) {
    btnToggleStreamPlay.innerHTML = isPaused ? `<i data-feather="play"></i>` : `<i data-feather="pause"></i>`;
    feather.replace();
}

// Stream play/pause trigger logic
btnToggleStreamPlay.addEventListener('click', () => {
    const source = dropdownSourceSelect.value;
    if (source === 'webcam') {
        if (webcamRawFeed.paused) {
            webcamRawFeed.play();
            setPlayPauseIcon(false);
            pushAlertLog("Hệ thống: Tiếp tục nhận diện Webcam USB trực tiếp.", "normal");
        } else {
            webcamRawFeed.pause();
            setPlayPauseIcon(true);
            pushAlertLog("Hệ thống: Tạm dừng đóng băng nhận diện Webcam USB.", "warning");
        }
    } else if (source === 'video') {
        if (videoFileFeed.paused) {
            videoFileFeed.play();
            setPlayPauseIcon(false);
            pushAlertLog("Hệ thống: Tiếp tục phát video file feed.", "normal");
        } else {
            videoFileFeed.pause();
            setPlayPauseIcon(true);
            pushAlertLog("Hệ thống: Tạm dừng phát video file feed.", "warning");
        }
    }
});

// Video progression event updates
videoFileFeed.addEventListener('timeupdate', () => {
    if (videoFileFeed.duration) {
        const pct = (videoFileFeed.currentTime / videoFileFeed.duration) * 100;
        videoProgressSlider.value = pct;
        
        // Update label
        const formatTime = (secs) => {
            const m = Math.floor(secs / 60).toString().padStart(2, '0');
            const s = Math.floor(secs % 60).toString().padStart(2, '0');
            return `${m}:${s}`;
        };
        lblVideoTime.textContent = `${formatTime(videoFileFeed.currentTime)} / ${formatTime(videoFileFeed.duration)}`;
    }
});

// Video dragging seek logic
videoProgressSlider.addEventListener('input', () => {
    if (videoFileFeed.duration) {
        const seekTime = (videoProgressSlider.value / 100) * videoFileFeed.duration;
        videoFileFeed.currentTime = seekTime;
    }
});

// Re-upload trigger logic
btnReuploadFile.addEventListener('click', () => {
    const source = dropdownSourceSelect.value;
    if (source === 'upload') {
        fileUploaderInput.click();
    } else if (source === 'video') {
        videoUploaderInput.click();
    }
});

// Initialize charts on window load
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
});