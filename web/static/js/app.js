/**
 * BioSonification - Frontend Application
 * Handles user interaction, file upload, and API calls
 */

// DOM Elements
const elements = {
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // Input
    fastaInput: document.getElementById('fasta-input'),
    fastaFile: document.getElementById('fasta-file'),
    fileUploadArea: document.getElementById('file-upload-area'),
    fileName: document.getElementById('file-name'),
    
    // Buttons
    generateBtn: document.getElementById('generate-btn'),
    retryBtn: document.getElementById('retry-btn'),
    generateAnotherBtn: document.getElementById('generate-another-btn'),
    downloadMidiBtn: document.getElementById('download-midi-btn'),
    
    // Sections
    inputSection: document.querySelector('.input-section'),
    loadingSection: document.getElementById('loading-section'),
    errorSection: document.getElementById('error-section'),
    resultsSection: document.getElementById('results-section'),
    errorMessage: document.getElementById('error-message'),
    
    // Audio
    audioPlayer: document.getElementById('audio-player'),
    audioUnavailable: document.getElementById('audio-unavailable'),
    
    // Parameters
    paramTempo: document.getElementById('param-tempo'),
    paramKey: document.getElementById('param-key'),
    paramPitchRange: document.getElementById('param-pitch-range'),
    paramScaleType: document.getElementById('param-scale-type'),
    paramRhythm: document.getElementById('param-rhythm'),
    paramDynamicRange: document.getElementById('param-dynamic-range'),
    
    // Sequence info
    seqHeader: document.getElementById('seq-header'),
    seqLength: document.getElementById('seq-length'),
};

// State
let selectedFile = null;

// ============================================
// Tab Switching
// ============================================

elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        // Update buttons
        elements.tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content
        elements.tabContents.forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabId}-tab`).classList.add('active');
    });
});

// ============================================
// File Upload
// ============================================

// File input change
elements.fastaFile.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        elements.fileName.textContent = selectedFile.name;
        elements.fileUploadArea.classList.add('has-file');
    }
});

// Drag and drop
elements.fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.fileUploadArea.classList.add('dragover');
});

elements.fileUploadArea.addEventListener('dragleave', () => {
    elements.fileUploadArea.classList.remove('dragover');
});

elements.fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        elements.fastaFile.files = e.dataTransfer.files;
        elements.fileName.textContent = selectedFile.name;
        elements.fileUploadArea.classList.add('has-file');
    }
});

// Click on upload area to trigger file input
elements.fileUploadArea.addEventListener('click', (e) => {
    if (e.target !== elements.fastaFile) {
        elements.fastaFile.click();
    }
});

// ============================================
// Generation
// ============================================

elements.generateBtn.addEventListener('click', handleGenerate);
elements.retryBtn.addEventListener('click', resetToInput);
elements.generateAnotherBtn.addEventListener('click', resetToInput);

async function handleGenerate() {
    // Get FASTA data
    let fastaText = elements.fastaInput.value.trim();
    
    // If no text, check if file is selected
    if (!fastaText && !selectedFile) {
        showError('Please paste a DNA sequence or upload a FASTA file');
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        let response;
        
        if (selectedFile) {
            // Upload file
            const formData = new FormData();
            formData.append('fasta_file', selectedFile);
            
            response = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });
        } else {
            // Send text
            response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ fasta: fastaText })
            });
        }
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Generation failed');
        }
        
        // Show results
        showResults(data);
        
    } catch (error) {
        showError(error.message);
    }
}

// ============================================
// Display Functions
// ============================================

function showLoading() {
    elements.inputSection.classList.add('hidden');
    elements.loadingSection.classList.remove('hidden');
    elements.errorSection.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
}

function showError(message) {
    elements.inputSection.classList.add('hidden');
    elements.loadingSection.classList.add('hidden');
    elements.errorSection.classList.remove('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.errorMessage.textContent = message;
}

function showResults(data) {
    elements.inputSection.classList.add('hidden');
    elements.loadingSection.classList.add('hidden');
    elements.errorSection.classList.add('hidden');
    elements.resultsSection.classList.remove('hidden');
    
    // Audio player
    if (data.audio_available) {
        elements.audioPlayer.src = `/api/download/${data.session_id}/wav`;
        elements.audioPlayer.classList.remove('hidden');
        elements.audioUnavailable.classList.add('hidden');
    } else {
        elements.audioPlayer.classList.add('hidden');
        elements.audioUnavailable.classList.remove('hidden');
    }
    
    // Download link
    elements.downloadMidiBtn.href = `/api/download/${data.session_id}/midi`;
    
    // Parameters
    const params = data.musical_params;
    elements.paramTempo.textContent = Math.round(params.tempo);
    elements.paramKey.textContent = params.key;
    elements.paramPitchRange.textContent = params.sequence_type;
    elements.paramScaleType.textContent = params.harmony_bars;
    elements.paramRhythm.textContent = params.melody_notes;
    elements.paramDynamicRange.textContent = params.device;
    
    // Sequence info
    elements.seqHeader.textContent = data.header || 'User Sequence';
    elements.seqLength.textContent = data.sequence_length.toLocaleString();
}

function resetToInput() {
    elements.inputSection.classList.remove('hidden');
    elements.loadingSection.classList.add('hidden');
    elements.errorSection.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
    
    // Reset file selection
    selectedFile = null;
    elements.fastaFile.value = '';
    elements.fileName.textContent = '';
    elements.fileUploadArea.classList.remove('has-file');
    
    // Stop audio
    elements.audioPlayer.pause();
    elements.audioPlayer.src = '';
}

// ============================================
// Utility Functions
// ============================================

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// ============================================
// Keyboard Shortcuts
// ============================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!elements.inputSection.classList.contains('hidden')) {
            handleGenerate();
        }
    }
});

// ============================================
// Initialization
// ============================================

// Check server status on load
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        if (!status.ready) {
            console.warn('Generator not ready:', status.error);
            // Could show a warning banner here
        }
        
        if (!status.audio_enabled) {
            console.info('Audio playback disabled - no synthesizer found');
        }
    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

// Run status check
checkStatus();

// ============================================
// Demo Data (for testing)
// ============================================

function loadDemoData() {
    const demoFasta = `>BRCA1 gene fragment (demo)
ATCGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC`;
    
    elements.fastaInput.value = demoFasta;
}

// Uncomment to load demo data on page load:
// loadDemoData();
