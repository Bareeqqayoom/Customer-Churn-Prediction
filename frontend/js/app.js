/**
 * ChurnIQ - Client-side Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // API Endpoints - dynamically bind to current host or fallback to 8001
    const isFileProtocol = window.location.protocol === 'file:' || !window.location.origin || window.location.origin === 'null';
    const API_BASE = isFileProtocol ? 'http://127.0.0.1:8001' : window.location.origin;
    const ENDPOINTS = {
        health: `${API_BASE}/api/health`,
        metadata: `${API_BASE}/api/metadata`,
        predict: `${API_BASE}/api/predict`,
        predictBatch: `${API_BASE}/api/predict-batch`
    };

    // State
    let appMetadata = null;
    let batchResultsData = null;

    // DOM Elements - Navigation & Status
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('backend-status-text');
    const tabSingle = document.getElementById('tab-single');
    const tabBatch = document.getElementById('tab-batch');
    const viewSingle = document.getElementById('view-single');
    const viewBatch = document.getElementById('view-batch');

    // DOM Elements - Single Prediction Form
    const churnForm = document.getElementById('churn-form');
    const tenureInput = document.getElementById('tenure');
    const monthlyInput = document.getElementById('MonthlyCharges');
    const totalInput = document.getElementById('TotalCharges');
    const tenureBadge = document.getElementById('tenure-badge');
    const predictBtn = document.getElementById('predict-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnIcon = document.getElementById('btn-icon');
    const btnText = document.getElementById('btn-text');

    // DOM Elements - Results Card
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsDisplay = document.getElementById('results-display');
    const gaugeCircle = document.getElementById('gauge-circle');
    const resPercentage = document.getElementById('res-percentage');
    const resRiskBadge = document.getElementById('res-risk-badge');
    const resClassBadge = document.getElementById('res-class-badge');
    const resSummary = document.getElementById('res-summary');
    const resConfidence = document.getElementById('res-confidence');
    const resConfidenceFill = document.getElementById('res-confidence-fill');
    const resRecommendations = document.getElementById('res-recommendations');
    const resRiskDrivers = document.getElementById('res-risk-drivers');
    const resProtectiveFactors = document.getElementById('res-protective-factors');

    // DOM Elements - Batch Prediction
    const csvDropzone = document.getElementById('csv-dropzone');
    const csvFileInput = document.getElementById('csv-file-input');
    const selectedFileName = document.getElementById('selected-file-name');
    const processBatchBtn = document.getElementById('process-batch-btn');
    const batchSpinner = document.getElementById('batch-spinner');
    const batchResultsSection = document.getElementById('batch-results-section');
    const downloadSampleBtn = document.getElementById('download-sample-btn');
    const exportCsvBtn = document.getElementById('export-csv-btn');
    const batchFilter = document.getElementById('batch-filter');
    const batchTableBody = document.getElementById('batch-table-body');
    const kpiTotal = document.getElementById('kpi-total');
    const kpiChurn = document.getElementById('kpi-churn');
    const kpiChurnRate = document.getElementById('kpi-churn-rate');
    const kpiHighRisk = document.getElementById('kpi-high-risk');
    const kpiLowRisk = document.getElementById('kpi-low-risk');

    // Circumference of SVG gauge circle (r=68) -> 2 * PI * 68 = 427.25
    const GAUGE_CIRCUMFERENCE = 427.25;

    /* --------------------------------------------------------------------------
       1. Initialization & Backend Connection
       -------------------------------------------------------------------------- */
    async function init() {
        setupEventListeners();
        await checkBackendHealth();
        await loadMetadata();
    }

    async function checkBackendHealth() {
        try {
            const res = await fetch(ENDPOINTS.health);
            if (res.ok) {
                const data = await res.json();
                statusDot.className = 'status-dot online';
                statusText.textContent = `Model Ready (${data.features_expected} Features)`;
            } else {
                throw new Error('Health check response not OK');
            }
        } catch (err) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Backend Offline';
            console.error('Backend health check error:', err);
        }
    }

    async function loadMetadata() {
        try {
            const res = await fetch(ENDPOINTS.metadata);
            if (res.ok) {
                appMetadata = await res.json();
            }
        } catch (err) {
            console.warn('Could not fetch metadata, using defaults:', err);
        }
    }

    /* --------------------------------------------------------------------------
       2. Event Listeners Setup
       -------------------------------------------------------------------------- */
    function setupEventListeners() {
        // Tab switching
        tabSingle.addEventListener('click', () => switchTab('single'));
        tabBatch.addEventListener('click', () => switchTab('batch'));

        // Persona preset clicks
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const presetKey = btn.dataset.preset;
                loadPersonaPreset(presetKey);
            });
        });

        // Tenure & Monthly charges dynamic recalculation
        tenureInput.addEventListener('input', handleChargesCalculation);
        monthlyInput.addEventListener('input', handleChargesCalculation);

        // Single prediction submit
        churnForm.addEventListener('submit', handleSinglePrediction);

        // Batch CSV file handling
        setupDropzone();
        downloadSampleBtn.addEventListener('click', generateSampleCSV);
        processBatchBtn.addEventListener('click', handleBatchPrediction);
        exportCsvBtn.addEventListener('click', exportBatchResultsToCSV);
        batchFilter.addEventListener('change', renderBatchTable);
    }

    /* --------------------------------------------------------------------------
       3. Tab Navigation
       -------------------------------------------------------------------------- */
    function switchTab(tab) {
        if (tab === 'single') {
            tabSingle.classList.add('active');
            tabSingle.setAttribute('aria-selected', 'true');
            tabBatch.classList.remove('active');
            tabBatch.setAttribute('aria-selected', 'false');
            viewSingle.classList.remove('hidden');
            viewBatch.classList.add('hidden');
        } else {
            tabBatch.classList.add('active');
            tabBatch.setAttribute('aria-selected', 'true');
            tabSingle.classList.remove('active');
            tabSingle.setAttribute('aria-selected', 'false');
            viewBatch.classList.remove('hidden');
            viewSingle.classList.add('hidden');
        }
    }

    /* --------------------------------------------------------------------------
       4. Auto-Calculation and Preset Logic
       -------------------------------------------------------------------------- */
    function handleChargesCalculation() {
        const tenure = parseInt(tenureInput.value, 10) || 0;
        const monthly = parseFloat(monthlyInput.value) || 0.0;
        
        tenureBadge.textContent = tenure === 1 ? '1 mo' : `${tenure} mos`;
        
        const calculatedTotal = (tenure * monthly).toFixed(2);
        totalInput.value = calculatedTotal;
    }

    function loadPersonaPreset(key) {
        if (!appMetadata || !appMetadata.presets || !appMetadata.presets[key]) {
            // Fallback hardcoded presets
            const fallbackPresets = {
                high_risk: {
                    gender: 'Female', SeniorCitizen: 0, Partner: 'No', Dependents: 'No',
                    tenure: 2, PhoneService: 'Yes', MultipleLines: 'Yes',
                    InternetService: 'Fiber optic', OnlineSecurity: 'No', OnlineBackup: 'No',
                    DeviceProtection: 'No', TechSupport: 'No', StreamingTV: 'Yes',
                    StreamingMovies: 'Yes', Contract: 'Month-to-month', PaperlessBilling: 'Yes',
                    PaymentMethod: 'Electronic check', MonthlyCharges: 98.50, TotalCharges: 197.00
                },
                medium_risk: {
                    gender: 'Female', SeniorCitizen: 1, Partner: 'No', Dependents: 'No',
                    tenure: 18, PhoneService: 'Yes', MultipleLines: 'No',
                    InternetService: 'Fiber optic', OnlineSecurity: 'Yes', OnlineBackup: 'No',
                    DeviceProtection: 'No', TechSupport: 'No', StreamingTV: 'No',
                    StreamingMovies: 'Yes', Contract: 'One year', PaperlessBilling: 'Yes',
                    PaymentMethod: 'Bank transfer (automatic)', MonthlyCharges: 79.20, TotalCharges: 1425.60
                },
                low_risk: {
                    gender: 'Male', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'Yes',
                    tenure: 60, PhoneService: 'Yes', MultipleLines: 'Yes',
                    InternetService: 'DSL', OnlineSecurity: 'Yes', OnlineBackup: 'Yes',
                    DeviceProtection: 'Yes', TechSupport: 'Yes', StreamingTV: 'Yes',
                    StreamingMovies: 'Yes', Contract: 'Two year', PaperlessBilling: 'No',
                    PaymentMethod: 'Credit card (automatic)', MonthlyCharges: 85.00, TotalCharges: 5100.00
                }
            };
            populateForm(fallbackPresets[key]);
        } else {
            populateForm(appMetadata.presets[key].data);
        }

        // Trigger prediction immediately for instant feedback
        churnForm.dispatchEvent(new Event('submit'));
    }

    function populateForm(data) {
        if (!data) return;
        Object.entries(data).forEach(([key, val]) => {
            const input = document.getElementById(key);
            if (input) {
                input.value = val;
            }
        });
        const tenure = parseInt(tenureInput.value, 10) || 0;
        tenureBadge.textContent = tenure === 1 ? '1 mo' : `${tenure} mos`;
    }

    /* --------------------------------------------------------------------------
       5. Single Prediction Processing
       -------------------------------------------------------------------------- */
    async function handleSinglePrediction(e) {
        e.preventDefault();

        // Extract form data
        const formData = new FormData(churnForm);
        const payload = {
            gender: formData.get('gender'),
            SeniorCitizen: parseInt(formData.get('SeniorCitizen'), 10),
            Partner: formData.get('Partner'),
            Dependents: formData.get('Dependents'),
            tenure: parseInt(formData.get('tenure'), 10),
            PhoneService: formData.get('PhoneService'),
            MultipleLines: formData.get('MultipleLines'),
            InternetService: formData.get('InternetService'),
            OnlineSecurity: formData.get('OnlineSecurity'),
            OnlineBackup: formData.get('OnlineBackup'),
            DeviceProtection: formData.get('DeviceProtection'),
            TechSupport: formData.get('TechSupport'),
            StreamingTV: formData.get('StreamingTV'),
            StreamingMovies: formData.get('StreamingMovies'),
            Contract: formData.get('Contract'),
            PaperlessBilling: formData.get('PaperlessBilling'),
            PaymentMethod: formData.get('PaymentMethod'),
            MonthlyCharges: parseFloat(formData.get('MonthlyCharges')),
            TotalCharges: parseFloat(formData.get('TotalCharges')) || null
        };

        // UI Loading State
        predictBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnIcon.classList.add('hidden');
        btnText.textContent = 'Calculating Inference...';

        try {
            const res = await fetch(ENDPOINTS.predict, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Prediction failed');
            }

            const result = await res.json();
            renderPredictionResults(result);
        } catch (err) {
            alert(`Prediction Error: ${err.message}`);
            console.error('Prediction API Error:', err);
        } finally {
            predictBtn.disabled = false;
            btnSpinner.classList.add('hidden');
            btnIcon.classList.remove('hidden');
            btnText.textContent = 'Run Churn Prediction';
        }
    }

    function renderPredictionResults(result) {
        resultsPlaceholder.classList.add('hidden');
        resultsDisplay.classList.remove('hidden');

        const prob = result.churn_probability;
        const pct = result.churn_percentage;
        const risk = result.risk_level;

        // Animate circular gauge
        const strokeOffset = GAUGE_CIRCUMFERENCE * (1 - prob);
        gaugeCircle.style.strokeDashoffset = strokeOffset;

        // Theme colors by risk
        let themeColor = 'var(--danger)';
        resRiskBadge.className = 'risk-badge';

        if (risk === 'High') {
            themeColor = 'var(--danger)';
            resRiskBadge.classList.add('risk-high');
            resRiskBadge.textContent = 'HIGH RISK';
            resClassBadge.textContent = 'Predicted: Churn';
            resClassBadge.style.color = 'var(--danger)';
        } else if (risk === 'Medium') {
            themeColor = 'var(--warning)';
            resRiskBadge.classList.add('risk-medium');
            resRiskBadge.textContent = 'MEDIUM RISK';
            resClassBadge.textContent = 'Predicted: Retained (Monitor)';
            resClassBadge.style.color = 'var(--warning)';
        } else {
            themeColor = 'var(--success)';
            resRiskBadge.classList.add('risk-low');
            resRiskBadge.textContent = 'LOW RISK';
            resClassBadge.textContent = 'Predicted: Retained';
            resClassBadge.style.color = 'var(--success)';
        }

        gaugeCircle.style.stroke = themeColor;
        resPercentage.textContent = `${pct.toFixed(1)}%`;
        resSummary.textContent = result.summary;

        // Confidence bar
        resConfidence.textContent = `${result.confidence.toFixed(1)}%`;
        resConfidenceFill.style.width = `${result.confidence}%`;

        // Render Retention Recommendations
        resRecommendations.innerHTML = '';
        if (result.retention_recommendations && result.retention_recommendations.length > 0) {
            result.retention_recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>${rec}</span>`;
                resRecommendations.appendChild(li);
            });
        }

        // Render Risk Drivers
        resRiskDrivers.innerHTML = '';
        if (result.top_risk_drivers && result.top_risk_drivers.length > 0) {
            const maxScore = Math.max(...result.top_risk_drivers.map(d => Math.abs(d.impact_score)), 0.001);
            result.top_risk_drivers.forEach(driver => {
                const widthPct = Math.min(100, Math.round((Math.abs(driver.impact_score) / maxScore) * 100));
                const item = document.createElement('div');
                item.className = 'factor-bar-item';
                item.innerHTML = `
                    <div class="factor-bar-header">
                        <span class="factor-name">${driver.label}</span>
                        <span class="factor-score text-danger">+${driver.impact_score.toFixed(3)}</span>
                    </div>
                    <div class="factor-track">
                        <div class="factor-fill factor-fill-danger" style="width: ${widthPct}%"></div>
                    </div>
                `;
                resRiskDrivers.appendChild(item);
            });
        } else {
            resRiskDrivers.innerHTML = '<span style="font-size:12px; color:var(--text-dim);">No significant churn risk factors identified.</span>';
        }

        // Render Protective Factors
        resProtectiveFactors.innerHTML = '';
        if (result.top_protective_factors && result.top_protective_factors.length > 0) {
            const maxScore = Math.max(...result.top_protective_factors.map(d => Math.abs(d.impact_score)), 0.001);
            result.top_protective_factors.forEach(factor => {
                const widthPct = Math.min(100, Math.round((Math.abs(factor.impact_score) / maxScore) * 100));
                const item = document.createElement('div');
                item.className = 'factor-bar-item';
                item.innerHTML = `
                    <div class="factor-bar-header">
                        <span class="factor-name">${factor.label}</span>
                        <span class="factor-score text-success">${factor.impact_score.toFixed(3)}</span>
                    </div>
                    <div class="factor-track">
                        <div class="factor-fill factor-fill-success" style="width: ${widthPct}%"></div>
                    </div>
                `;
                resProtectiveFactors.appendChild(item);
            });
        } else {
            resProtectiveFactors.innerHTML = '<span style="font-size:12px; color:var(--text-dim);">No protective factors identified.</span>';
        }
    }

    /* --------------------------------------------------------------------------
       6. Batch CSV File Handling & Prediction
       -------------------------------------------------------------------------- */
    let selectedFile = null;

    function setupDropzone() {
        csvDropzone.addEventListener('click', () => csvFileInput.click());
        
        csvDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            csvDropzone.classList.add('dragover');
        });

        csvDropzone.addEventListener('dragleave', () => {
            csvDropzone.classList.remove('dragover');
        });

        csvDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            csvDropzone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        csvFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a valid .csv file.');
            return;
        }
        selectedFile = file;
        selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        processBatchBtn.disabled = false;
    }

    async function handleBatchPrediction() {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        processBatchBtn.disabled = true;
        batchSpinner.classList.remove('hidden');

        try {
            const res = await fetch(ENDPOINTS.predictBatch, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Batch processing failed');
            }

            batchResultsData = await res.json();
            renderBatchResults(batchResultsData);
        } catch (err) {
            alert(`Batch Prediction Error: ${err.message}`);
            console.error('Batch error:', err);
        } finally {
            processBatchBtn.disabled = false;
            batchSpinner.classList.add('hidden');
        }
    }

    function renderBatchResults(data) {
        batchResultsSection.classList.remove('hidden');

        kpiTotal.textContent = data.total_customers.toLocaleString();
        kpiChurn.textContent = data.churn_count.toLocaleString();
        kpiChurnRate.textContent = `${data.churn_rate_percentage}% churn rate`;
        kpiHighRisk.textContent = data.high_risk_count.toLocaleString();
        kpiLowRisk.textContent = data.low_risk_count.toLocaleString();

        renderBatchTable();
    }

    function renderBatchTable() {
        if (!batchResultsData || !batchResultsData.results) return;

        const filter = batchFilter.value;
        const rows = batchResultsData.results.filter(item => {
            if (filter === 'high') return item.risk_level === 'High';
            if (filter === 'churn') return item.prediction === 1;
            if (filter === 'no-churn') return item.prediction === 0;
            return true;
        });

        batchTableBody.innerHTML = '';
        if (rows.length === 0) {
            batchTableBody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding:24px; color:var(--text-dim);">No records match the selected filter.</td></tr>';
            return;
        }

        rows.slice(0, 100).forEach(r => {
            const tr = document.createElement('tr');
            let riskBadgeClass = 'risk-low';
            if (r.risk_level === 'High') riskBadgeClass = 'risk-high';
            else if (r.risk_level === 'Medium') riskBadgeClass = 'risk-medium';

            tr.innerHTML = `
                <td>${r.row_index}</td>
                <td><strong style="font-family:var(--font-mono); font-size:12.5px;">${r.customer_id}</strong></td>
                <td><strong>${r.churn_percentage}%</strong></td>
                <td><span class="risk-badge ${riskBadgeClass}" style="font-size:11px; padding:2px 8px;">${r.risk_level}</span></td>
                <td>${r.prediction_label === 'Churn' ? '<span class="text-danger"><i class="fa-solid fa-triangle-exclamation"></i> Churn</span>' : '<span class="text-success"><i class="fa-solid fa-check"></i> Retained</span>'}</td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    function generateSampleCSV() {
        const sampleHeaders = "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n";
        const sampleRows = [
            "CUST-001,Female,0,No,No,1,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.35,75.35",
            "CUST-002,Male,0,Yes,Yes,65,Yes,Yes,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Two year,No,Credit card (automatic),84.50,5492.50",
            "CUST-003,Female,1,No,No,12,Yes,No,Fiber optic,Yes,No,No,No,Yes,No,One year,Yes,Bank transfer (automatic),80.10,961.20",
            "CUST-004,Male,0,No,No,3,Yes,No,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Month-to-month,No,Mailed check,19.85,59.55",
            "CUST-005,Female,0,Yes,No,36,Yes,Yes,Fiber optic,No,Yes,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,104.20,3751.20"
        ].join('\n');

        const blob = new Blob([sampleHeaders + sampleRows], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'churn_customers_sample.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function exportBatchResultsToCSV() {
        if (!batchResultsData || !batchResultsData.results) return;

        let csv = "Row,CustomerID,ChurnProbability,ChurnPercentage,RiskTier,Prediction\n";
        batchResultsData.results.forEach(r => {
            csv += `${r.row_index},"${r.customer_id}",${r.churn_probability},${r.churn_percentage}%,"${r.risk_level}","${r.prediction_label}"\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'churn_predictions_export.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Start App
    init();
});
