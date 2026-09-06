/* ================================================================
   AI Risk Manager — Premium Dashboard SPA
   Pure vanilla JS — zero external dependencies
   ================================================================ */

(function () {
    'use strict';

    // ─── State ──────────────────────────────────────────────────
    const state = {
        currentSection: 'hero',
        metricsData: null,
        robustnessData: null,
        sensitivityData: null,
        modelInfo: null,
        scoringHistory: [],
        chartsRendered: false,
    };

    // ─── DOM Ready ──────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        initNavigation();
        initScrollAnimations();
        initHeroCounters();
        initScorer();
        initPresets();
        checkAPI();
        loadDashboardData();
    });

    // ─── Navigation ─────────────────────────────────────────────
    function initNavigation() {
        const nav = document.getElementById('mainNav');
        const navLinks = document.querySelectorAll('.nav-link');

        // Smooth scroll on nav click
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = link.getAttribute('href').substring(1);
                const section = document.getElementById(target);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });

        // Smooth scroll for hero CTAs
        document.querySelectorAll('.btn-primary-hero, .btn-ghost-hero').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const href = link.getAttribute('href');
                if (href && href.startsWith('#')) {
                    const section = document.getElementById(href.substring(1));
                    if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });

        // Track scroll for nav styling and active link
        const sections = document.querySelectorAll('.section');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && entry.intersectionRatio > 0.2) {
                    setActiveNav(entry.target.id);
                }
            });
        }, { threshold: [0.2, 0.5] });

        sections.forEach(s => observer.observe(s));

        // Nav background on scroll
        window.addEventListener('scroll', () => {
            if (window.scrollY > 20) {
                nav.classList.add('scrolled');
            } else {
                nav.classList.remove('scrolled');
            }
        }, { passive: true });
    }

    function setActiveNav(sectionId) {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`.nav-link[data-section="${sectionId}"]`);
        if (active) active.classList.add('active');
        state.currentSection = sectionId;
    }

    // ─── Scroll Animations ──────────────────────────────────────
    function initScrollAnimations() {
        const animElements = document.querySelectorAll('.animate-on-scroll');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

        animElements.forEach(el => observer.observe(el));
    }

    // ─── Hero Counter Animations ─────────────────────────────────
    function initHeroCounters() {
        const counters = document.querySelectorAll('.counter-value');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(c => observer.observe(c));
    }

    function animateCounter(el) {
        const target = parseFloat(el.dataset.target);
        const suffix = el.dataset.suffix || '';
        const prefix = el.dataset.prefix || '';
        const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
        const duration = 2200;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;

            el.textContent = prefix + current.toFixed(decimals) + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }

    // ─── API Health Check ────────────────────────────────────────
    async function checkAPI() {
        const badge = document.getElementById('apiStatus');
        try {
            const res = await fetch('/health');
            if (res.ok) {
                badge.classList.remove('offline');
                badge.classList.add('online');
                badge.querySelector('.status-text').textContent = 'API Online';
            } else throw new Error();
        } catch {
            badge.classList.remove('online');
            badge.classList.add('offline');
            badge.querySelector('.status-text').textContent = 'API Offline';
        }
    }

    // ─── Load Dashboard Data ─────────────────────────────────────
    async function loadDashboardData() {
        try {
            const [metrics, robustness, sensitivity, modelInfo] = await Promise.all([
                fetch('/api/metrics').then(r => r.json()).catch(() => null),
                fetch('/api/robustness').then(r => r.json()).catch(() => null),
                fetch('/api/sensitivity').then(r => r.json()).catch(() => null),
                fetch('/api/model-info').then(r => r.json()).catch(() => null),
            ]);

            state.metricsData = metrics;
            state.robustnessData = robustness;
            state.sensitivityData = sensitivity;
            state.modelInfo = modelInfo;

            renderPerformanceCharts();
            renderRobustnessCards();
            renderModelComparison();
        } catch (e) {
            console.error('Failed to load dashboard data:', e);
        }
    }

    // ─── Scorer ──────────────────────────────────────────────────
    function initScorer() {
        const form = document.getElementById('scoreForm');
        const fillBtn = document.getElementById('fillExample');
        const retryBtn = document.getElementById('retryBtn');
        const toggleRaw = document.getElementById('toggleRaw');

        form.addEventListener('submit', handleScore);
        fillBtn.addEventListener('click', () => fillPreset('high_risk'));
        if (retryBtn) retryBtn.addEventListener('click', () => {
            document.getElementById('errorState').hidden = true;
            document.getElementById('emptyState').style.display = '';
        });
        if (toggleRaw) toggleRaw.addEventListener('click', () => {
            const raw = document.getElementById('rawJson');
            raw.hidden = !raw.hidden;
            toggleRaw.lastChild.textContent = raw.hidden ? ' Show raw JSON' : ' Hide raw JSON';
        });
    }

    function initPresets() {
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                fillPreset(preset);
                // Auto-submit after a brief visual delay
                setTimeout(() => {
                    document.getElementById('scoreForm').dispatchEvent(new Event('submit', { cancelable: true }));
                }, 350);
            });
        });
    }

    const PRESETS = {
        high_risk: {
            account_age_days: 0, prior_orders_count: 0, prior_return_rate: 0,
            order_value: 4500, item_category: 'footwear', item_count: 2,
            discount_pct: 65, payment_method: 'cod', delivery_promise_days: 3,
            actual_delivery_days: 8, device_type: 'mobile', time_of_day: 'late_night',
            is_first_order: 1,
        },
        safe_buyer: {
            account_age_days: 730, prior_orders_count: 25, prior_return_rate: 0.04,
            order_value: 1200, item_category: 'electronics', item_count: 1,
            discount_pct: 10, payment_method: 'upi', delivery_promise_days: 5,
            actual_delivery_days: 4, device_type: 'desktop', time_of_day: 'afternoon',
            is_first_order: 0,
        },
        edge_case: {
            account_age_days: 45, prior_orders_count: 3, prior_return_rate: 0.33,
            order_value: 2800, item_category: 'fashion', item_count: 3,
            discount_pct: 40, payment_method: 'card', delivery_promise_days: 4,
            actual_delivery_days: 6, device_type: 'mobile', time_of_day: 'evening',
            is_first_order: 0,
        },
    };

    function fillPreset(name) {
        const p = PRESETS[name];
        if (!p) return;
        Object.entries(p).forEach(([key, val]) => {
            const el = document.getElementById(key);
            if (el) el.value = val;
        });
        // Highlight active preset
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.preset === name);
        });
        // Scroll scorer into view
        const scorer = document.getElementById('scorer');
        if (scorer) scorer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    async function handleScore(e) {
        e.preventDefault();
        const form = e.target;
        const btn = document.getElementById('submitBtn');
        const btnLabel = btn.querySelector('.btn-label');
        const btnLoader = btn.querySelector('.btn-loader');

        // Loading state
        btn.disabled = true;
        btnLabel.hidden = true;
        btnLoader.hidden = false;

        const formData = new FormData(form);
        const payload = {};
        formData.forEach((val, key) => {
            if (['order_value', 'prior_return_rate', 'discount_pct'].includes(key)) {
                payload[key] = parseFloat(val);
            } else if (['account_age_days', 'prior_orders_count', 'item_count',
                'delivery_promise_days', 'actual_delivery_days', 'is_first_order'].includes(key)) {
                const v = parseInt(val);
                if (!isNaN(v)) payload[key] = v;
            } else {
                payload[key] = val;
            }
        });

        // Remove actual_delivery_days if empty
        if (payload.actual_delivery_days === undefined || isNaN(payload.actual_delivery_days)) {
            payload.actual_delivery_days = null;
        }

        try {
            const res = await fetch('/score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errBody = await res.json().catch(() => ({}));
                throw new Error(errBody.detail || `API error ${res.status}`);
            }

            const data = await res.json();
            showResults(data);
        } catch (err) {
            showError(err.message);
        } finally {
            btn.disabled = false;
            btnLabel.hidden = false;
            btnLoader.hidden = true;
        }
    }

    function showResults(data) {
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('errorState').hidden = true;
        const results = document.getElementById('results');
        results.hidden = false;

        // Animate gauge
        animateGauge(data.risk_score, data.threshold);

        // Decision badge
        const badge = document.getElementById('decisionBadge');
        const icon = document.getElementById('decisionIcon');
        const text = document.getElementById('decisionText');

        badge.className = 'decision-badge ' + (data.flagged_for_review ? 'flagged' : 'safe');
        icon.textContent = data.flagged_for_review ? '⚠' : '✓';
        text.textContent = data.flagged_for_review ? 'Flagged for Review' : 'Low Risk — Pass';

        // Explanation
        document.getElementById('explanationText').textContent = data.explanation;

        // SHAP Contributors — waterfall bars
        renderContributors(data.top_contributors);

        // Calibration note
        const calNote = document.querySelector('#calibrationNote span');
        if (calNote) calNote.textContent = data.calibration_note;

        // Raw JSON
        document.getElementById('rawJson').textContent = JSON.stringify(data, null, 2);

        // Add to history
        state.scoringHistory.push({
            score: data.risk_score,
            flagged: data.flagged_for_review,
            time: new Date().toLocaleTimeString(),
        });
        renderScoreHistory();
    }

    function showError(msg) {
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('results').hidden = true;
        const errState = document.getElementById('errorState');
        errState.hidden = false;
        document.getElementById('errorText').textContent = msg;
    }

    // ─── Gauge Animation ─────────────────────────────────────────
    function animateGauge(score, threshold) {
        const totalArc = 251.33;
        const fillOffset = totalArc * (1 - score);
        const gaugeFill = document.getElementById('gaugeFill');
        const gaugeNeedle = document.getElementById('gaugeNeedle');
        const gaugeValue = document.getElementById('gaugeValue');

        // Animate fill
        gaugeFill.style.strokeDashoffset = fillOffset;

        // Animate needle: angle from -180 (left) to 0 (right)
        const angle = -180 + (score * 180);
        gaugeNeedle.style.transform = `rotate(${angle}deg)`;

        // Threshold marker position
        const thAngle = (-180 + threshold * 180) * Math.PI / 180;
        const cx = 100 + 80 * Math.cos(thAngle);
        const cy = 100 + 80 * Math.sin(thAngle);
        const thCircle = document.getElementById('gaugeThreshold');
        const thLabel = document.getElementById('gaugeThresholdLabel');
        thCircle.setAttribute('cx', cx);
        thCircle.setAttribute('cy', cy);
        thLabel.setAttribute('x', cx);
        thLabel.setAttribute('y', cy - 10);
        thLabel.textContent = `T=${threshold}`;

        // Value text
        gaugeValue.textContent = (score * 100).toFixed(1) + '%';

        // Color based on risk
        if (score < 0.3) {
            gaugeValue.style.color = 'var(--risk-low)';
        } else if (score < 0.6) {
            gaugeValue.style.color = 'var(--risk-medium)';
        } else {
            gaugeValue.style.color = 'var(--risk-high)';
        }
    }

    // ─── SHAP Contributors Rendering ─────────────────────────────
    function renderContributors(contributors) {
        const list = document.getElementById('contributorsList');
        list.innerHTML = '';

        const maxAbsShap = Math.max(...contributors.map(c => Math.abs(c.shap_value)), 0.01);

        contributors.forEach((c, i) => {
            const isUp = c.direction === 'increases_risk';
            const barWidth = (Math.abs(c.shap_value) / maxAbsShap) * 100;

            const div = document.createElement('div');
            div.className = 'contributor';
            div.style.animationDelay = `${i * 0.12}s`;

            div.innerHTML = `
                <div class="contributor-direction ${isUp ? 'risk-up' : 'risk-down'}">
                    ${isUp ? '↑' : '↓'}
                </div>
                <div class="contributor-info">
                    <div class="contributor-name">${c.friendly_name}</div>
                    <div class="contributor-feature">${c.feature}</div>
                </div>
                <div class="contributor-bar-wrapper">
                    <div class="contributor-bar-track">
                        <div class="contributor-bar-fill ${isUp ? 'risk-up' : 'risk-down'}" style="width: ${barWidth}%"></div>
                    </div>
                    <div class="contributor-value ${isUp ? 'risk-up' : 'risk-down'}">
                        ${isUp ? '+' : ''}${c.shap_value.toFixed(2)}
                    </div>
                </div>
            `;
            list.appendChild(div);
        });
    }

    // ─── Score History Sparkline ──────────────────────────────────
    function renderScoreHistory() {
        const wrapper = document.getElementById('scoreHistoryWrapper');
        const container = document.getElementById('scoreHistoryChart');
        if (!container || state.scoringHistory.length === 0) return;

        wrapper.hidden = false;
        container.innerHTML = '';
        const canvas = document.createElement('canvas');
        const dpr = window.devicePixelRatio || 1;
        const w = container.clientWidth || 300;
        const h = 60;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        container.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        const history = state.scoringHistory.slice(-20);
        const step = w / Math.max(history.length - 1, 1);

        // Draw threshold line
        const thY = h - (0.14 * h);
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, thY);
        ctx.lineTo(w, thY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw line
        ctx.strokeStyle = '#0C83FF';
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        history.forEach((item, i) => {
            const x = i * step;
            const y = h - (item.score * h);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Draw dots
        history.forEach((item, i) => {
            const x = i * step;
            const y = h - (item.score * h);
            ctx.fillStyle = item.flagged ? '#ef4444' : '#22c55e';
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();
        });

        // Update count
        const countEl = document.getElementById('historyCount');
        if (countEl) countEl.textContent = state.scoringHistory.length;
    }

    // ─── Performance Charts ──────────────────────────────────────

    function renderPerformanceCharts() {
        if (!state.metricsData) return;
        renderConfusionMatrix();
        renderCalibrationChart();
        renderSensitivityChart();
        renderMetricsBars();
    }

    function renderConfusionMatrix() {
        const container = document.getElementById('confusionMatrix');
        if (!container || !state.metricsData) return;

        const cm = state.metricsData.test_confusion_matrix;
        if (!cm) return;

        const total = cm.tn + cm.fp + cm.fn + cm.tp;
        const cells = [
            { label: 'TN', val: cm.tn, pct: (cm.tn / total * 100).toFixed(1), cls: 'cm-tn' },
            { label: 'FP', val: cm.fp, pct: (cm.fp / total * 100).toFixed(1), cls: 'cm-fp' },
            { label: 'FN', val: cm.fn, pct: (cm.fn / total * 100).toFixed(1), cls: 'cm-fn' },
            { label: 'TP', val: cm.tp, pct: (cm.tp / total * 100).toFixed(1), cls: 'cm-tp' },
        ];

        container.innerHTML = `
            <div class="cm-grid">
                <div class="cm-axis cm-axis-y">
                    <span>Actual<br>Negative</span>
                    <span>Actual<br>Positive</span>
                </div>
                <div class="cm-cells">
                    <div class="cm-axis cm-axis-x">
                        <span>Pred. Negative</span>
                        <span>Pred. Positive</span>
                    </div>
                    ${cells.map(c => `
                        <div class="cm-cell ${c.cls}" title="${c.label}: ${c.val} (${c.pct}%)">
                            <span class="cm-val">${c.val}</span>
                            <span class="cm-label">${c.label}</span>
                            <span class="cm-pct">${c.pct}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    function renderCalibrationChart() {
        const container = document.getElementById('calibrationChart');
        if (!container || !state.metricsData) return;

        const before = state.metricsData.mean_abs_calibration_gap_before;
        const after = state.metricsData.mean_abs_calibration_gap_after;
        const reduction = state.metricsData.calibration_gap_reduction;

        if (before === undefined) {
            container.innerHTML = '<div class="chart-loading">No calibration data</div>';
            return;
        }

        container.innerHTML = `
            <div class="cal-comparison">
                <div class="cal-bar-group">
                    <div class="cal-bar-label">Before<br><span class="cal-val">${(before * 100).toFixed(1)}%</span></div>
                    <div class="cal-bar-track">
                        <div class="cal-bar-fill cal-before" style="width: ${Math.min(before * 100 * 3.5, 100)}%"></div>
                    </div>
                </div>
                <div class="cal-bar-group">
                    <div class="cal-bar-label">After<br><span class="cal-val">${(after * 100).toFixed(1)}%</span></div>
                    <div class="cal-bar-track">
                        <div class="cal-bar-fill cal-after" style="width: ${Math.min(after * 100 * 3.5, 100)}%"></div>
                    </div>
                </div>
                <div class="cal-reduction">
                    <span class="cal-reduction-value">${(reduction * 100).toFixed(1)}%</span>
                    <span class="cal-reduction-label">Gap Reduction</span>
                </div>
            </div>
        `;
    }

    function renderSensitivityChart() {
        const container = document.getElementById('sensitivityChart');
        if (!container || !state.sensitivityData || !Array.isArray(state.sensitivityData)) return;

        const canvas = document.createElement('canvas');
        const dpr = window.devicePixelRatio || 1;
        const w = container.clientWidth || 500;
        const h = 210;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        container.innerHTML = '';
        container.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        const data = state.sensitivityData;
        const padL = 48, padR = 20, padT = 20, padB = 40;
        const chartW = w - padL - padR;
        const chartH = h - padT - padB;

        // Grid
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padT + (chartH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(w - padR, y);
            ctx.stroke();

            ctx.fillStyle = 'rgba(71, 85, 105, 0.8)';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText((1 - i / 4).toFixed(1), padL - 8, y + 3);
        }

        // X axis labels
        data.forEach((d, i) => {
            const x = padL + (i / Math.max(data.length - 1, 1)) * chartW;
            ctx.fillStyle = 'rgba(71, 85, 105, 0.8)';
            ctx.font = '9px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(d.fn_fp_ratio.toFixed(1), x, h - 8);
        });

        ctx.fillStyle = 'rgba(100, 116, 139, 0.8)';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('FN:FP Cost Ratio', w / 2, h - 0);

        // Draw lines
        const metrics = [
            { key: 'test_precision', color: '#0C83FF', label: 'Precision' },
            { key: 'test_recall', color: '#059669', label: 'Recall' },
            { key: 'test_f1', color: '#D97706', label: 'F1' },
        ];

        metrics.forEach(metric => {
            ctx.strokeStyle = metric.color;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            data.forEach((d, i) => {
                const x = padL + (i / Math.max(data.length - 1, 1)) * chartW;
                const y = padT + chartH * (1 - d[metric.key]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Dots
            data.forEach((d, i) => {
                const x = padL + (i / Math.max(data.length - 1, 1)) * chartW;
                const y = padT + chartH * (1 - d[metric.key]);
                ctx.fillStyle = metric.color;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            });
        });

        // Baseline marker
        const baselineIdx = data.findIndex(d => d.ratio_multiplier === 1.0);
        if (baselineIdx >= 0) {
            const bx = padL + (baselineIdx / Math.max(data.length - 1, 1)) * chartW;
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(bx, padT);
            ctx.lineTo(bx, padT + chartH);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#f59e0b';
            ctx.font = '9px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('baseline', bx, padT - 5);
        }

        // Legend
        const legendY = padT + 12;
        let legendX = padL + 8;
        metrics.forEach(m => {
            ctx.fillStyle = m.color;
            ctx.fillRect(legendX, legendY - 4, 14, 3);
            ctx.fillStyle = 'rgba(232, 236, 244, 0.7)';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(m.label, legendX + 18, legendY);
            legendX += ctx.measureText(m.label).width + 36;
        });
    }

    function renderMetricsBars() {
        const container = document.getElementById('metricsBarChart');
        if (!container || !state.metricsData) return;

        const metrics = [
            { label: 'Precision', val: state.metricsData.test_precision, color: '#0C83FF' },
            { label: 'Recall', val: state.metricsData.test_recall, color: '#059669' },
            { label: 'F1', val: state.metricsData.test_f1, color: '#D97706' },
            { label: 'ROC-AUC', val: state.metricsData.test_roc_auc, color: '#4F46E5' },
            { label: 'PR-AUC', val: state.metricsData.test_pr_auc, color: '#7C3AED' },
        ];

        container.innerHTML = metrics.map(m => `
            <div class="metric-bar-row">
                <span class="metric-bar-label">${m.label}</span>
                <div class="metric-bar-track">
                    <div class="metric-bar-fill" style="width: ${m.val * 100}%; background: ${m.color}"></div>
                </div>
                <span class="metric-bar-value" style="color: ${m.color}">${m.val.toFixed(3)}</span>
            </div>
        `).join('');
    }

    // ─── Robustness Cards ────────────────────────────────────────
    function renderRobustnessCards() {
        if (!state.robustnessData) return;

        // Stress tests
        const stressContainer = document.getElementById('stressTests');
        if (stressContainer && state.robustnessData.stress_test) {
            stressContainer.innerHTML = state.robustnessData.stress_test.map(test => `
                <div class="stress-card ${test.passed ? 'passed' : 'failed'}">
                    <div class="stress-header">
                        <span class="stress-status">${test.passed ? '✓' : '✗'}</span>
                        <span class="stress-name">${test.name.replace(/_/g, ' ')}</span>
                    </div>
                    <p class="stress-desc">${test.description}</p>
                    <div class="stress-meta">
                        <span class="stress-proba">P(return) = ${(test.predicted_proba * 100).toFixed(1)}%</span>
                        <span class="stress-flag ${test.flagged ? 'flagged' : 'safe'}">${test.flagged ? 'Flagged' : 'Not Flagged'}</span>
                    </div>
                </div>
            `).join('');
        }

        // Stability
        const stabilityContainer = document.getElementById('stabilityResults');
        if (stabilityContainer && state.robustnessData.stability_check) {
            const checks = state.robustnessData.stability_check;
            const avgROC = checks.reduce((s, c) => s + c.roc_auc, 0) / checks.length;
            const stdROC = Math.sqrt(checks.reduce((s, c) => s + Math.pow(c.roc_auc - avgROC, 2), 0) / checks.length);

            stabilityContainer.innerHTML = `
                <div class="stability-summary">
                    <div class="stability-stat">
                        <span class="stability-val">${avgROC.toFixed(3)}</span>
                        <span class="stability-label">Avg ROC-AUC</span>
                    </div>
                    <div class="stability-stat">
                        <span class="stability-val">±${stdROC.toFixed(3)}</span>
                        <span class="stability-label">Std Dev</span>
                    </div>
                    <div class="stability-stat">
                        <span class="stability-val">${checks.length}</span>
                        <span class="stability-label">Seeds Tested</span>
                    </div>
                </div>
                <div class="stability-seeds">
                    ${checks.map(c => `
                        <div class="seed-row">
                            <span class="seed-label">Seed ${c.seed}</span>
                            <span class="seed-roc">ROC ${c.roc_auc.toFixed(3)}</span>
                            <span class="seed-pr">PR ${c.pr_auc.toFixed(3)}</span>
                            <span class="seed-f1">F1 ${c.f1.toFixed(3)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Leakage
        const leakageContainer = document.getElementById('leakageAudit');
        if (leakageContainer && state.robustnessData.leakage_audit) {
            const audit = state.robustnessData.leakage_audit;
            leakageContainer.innerHTML = `
                <div class="leakage-result ${audit.suspicious_features.length === 0 ? 'clean' : 'warning'}">
                    <div class="leakage-icon">${audit.suspicious_features.length === 0 ? '🛡️' : '⚠️'}</div>
                    <div class="leakage-info">
                        <div class="leakage-status">${audit.suspicious_features.length === 0 ? 'No Leakage Detected' : 'Suspicious Features Found'}</div>
                        <div class="leakage-detail">Max |correlation| with label: ${audit.max_abs_label_correlation.toFixed(4)}</div>
                        <div class="leakage-detail">Threshold: 0.5 (none exceeded)</div>
                    </div>
                </div>
            `;
        }
    }

    // ─── Model Comparison ────────────────────────────────────────
    function renderModelComparison() {
        const container = document.getElementById('modelComparison');
        if (!container || !state.modelInfo || !state.modelInfo.comparison) return;

        const models = state.modelInfo.comparison;
        const maxPR = Math.max(...models.map(m => m.pr_auc_mean));

        const familyNames = {
            'logreg': 'Logistic Regression',
            'rf': 'Random Forest',
            'lgbm': 'LightGBM',
        };

        container.innerHTML = models.map(m => {
            const isWinner = m.family === 'logreg';
            return `
                <div class="model-card-compare ${isWinner ? 'winner' : ''}">
                    ${isWinner ? '<div class="winner-badge">✦ Selected</div>' : ''}
                    <div class="model-name">${familyNames[m.family] || m.family}</div>
                    <div class="model-metrics">
                        <div class="model-metric">
                            <span class="model-metric-val">${m.pr_auc_mean.toFixed(3)}</span>
                            <span class="model-metric-label">PR-AUC</span>
                        </div>
                        <div class="model-metric">
                            <span class="model-metric-val">${m.roc_auc_mean.toFixed(3)}</span>
                            <span class="model-metric-label">ROC-AUC</span>
                        </div>
                        <div class="model-metric">
                            <span class="model-metric-val">${m.seconds.toFixed(1)}s</span>
                            <span class="model-metric-label">Train Time</span>
                        </div>
                    </div>
                    <div class="model-bar-track">
                        <div class="model-bar-fill ${isWinner ? 'winner' : ''}" style="width: ${(m.pr_auc_mean / maxPR) * 100}%"></div>
                    </div>
                </div>
            `;
        }).join('');
    }

})();
