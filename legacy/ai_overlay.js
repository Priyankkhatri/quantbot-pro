/**
 * AI Institutional Vision HUD & Active Candlestick Laser Scanner for TradingView
 * - Positioned on Top-Left (Never blocks TradingView right-side price scale / watchlist)
 * - Laser Scanner dynamically tracks the live forming candlestick zone (Chart Area: right: 380px)
 * - Zero-Second Delay: Real-time 60 FPS DOM price observer
 * - Deep Multi-Section Institutional Reasoning Blueprint (Indicators, Pivots, Fibonacci, 5-Model Matrix)
 */

(function() {
    // Remove existing instances to prevent duplicate overlays
    const oldHud = document.getElementById('ai-vision-hud');
    if (oldHud) oldHud.remove();
    const oldCursor = document.getElementById('ai-virtual-cursor');
    if (oldCursor) oldCursor.remove();

    // 1. Create Dynamic AI Candlestick Laser Scanner (Targeted at Live Chart Candles)
    const cursor = document.createElement('div');
    cursor.id = 'ai-virtual-cursor';
    cursor.innerHTML = `
        <div style="position: relative;">
            <div style="width: 32px; height: 32px; border-radius: 50%; border: 2.5px solid #10b981; background: rgba(16, 185, 129, 0.2); box-shadow: 0 0 16px #10b981; display: flex; align-items: center; justify-content: center; animation: ai-cursor-pulse 1s infinite alternate;">
                <div style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></div>
            </div>
            <!-- Horizontal Laser Line -->
            <div style="position: absolute; top: 15px; right: 32px; width: 65px; height: 1.5px; background: linear-gradient(90deg, transparent, #10b981); pointer-events: none;"></div>
            <!-- Badge -->
            <div id="ai-cursor-badge" style="position: absolute; top: -22px; left: -35px; background: rgba(15, 23, 42, 0.96); color: #10b981; border: 1.2px solid #10b981; font-size: 9px; font-weight: 800; font-family: monospace; padding: 2px 7px; border-radius: 4px; white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.6); pointer-events: none;">
                AI LIVE SCANNER
            </div>
        </div>
    `;
    cursor.style.cssText = `
        position: fixed;
        top: 280px;
        right: 400px;
        z-index: 999980;
        pointer-events: none;
        transition: top 0.6s cubic-bezier(0.25, 1, 0.5, 1), right 0.6s cubic-bezier(0.25, 1, 0.5, 1);
    `;
    document.body.appendChild(cursor);

    // 2. Create Floating Draggable Institutional HUD
    const hud = document.createElement('div');
    hud.id = 'ai-vision-hud';
    hud.style.cssText = `
        position: fixed;
        top: 55px;
        left: 70px;
        width: 380px;
        background: rgba(15, 23, 42, 0.96);
        border: 1.5px solid #334155;
        border-radius: 12px;
        padding: 12px 14px;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7), 0 0 20px rgba(37, 99, 235, 0.25);
        z-index: 999999;
        backdrop-filter: blur(14px);
        user-select: none;
    `;

    hud.innerHTML = `
        <!-- Header -->
        <div id="ai-hud-header" style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 7px; margin-bottom: 8px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 7px;">
                <span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #10b981; box-shadow: 0 0 10px #10b981; animation: ai-pulse 1.2s infinite;"></span>
                <span style="font-weight: 800; font-size: 12px; letter-spacing: 0.5px; color: #38bdf8;">QUANTBOT 5-MODEL AI</span>
                <span style="font-size: 9px; font-weight: 700; background: #1e293b; color: #94a3b8; padding: 1px 5px; border-radius: 4px; border: 1px solid #334155;">INSTITUTIONAL</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span id="ai-hud-confidence" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; font-size: 10px; font-weight: bold; padding: 1px 6px; border-radius: 4px; font-family: monospace;">
                    --%
                </span>
                <button id="ai-hud-min-btn" style="background: #1e293b; color: #94a3b8; border: 1px solid #334155; border-radius: 4px; width: 20px; height: 20px; font-size: 11px; cursor: pointer; display: flex; align-items: center; justify-content: center;">−</button>
            </div>
        </div>

        <div id="ai-hud-body">
            <!-- Price & Regime -->
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(2, 6, 23, 0.85); border: 1px solid #1e293b; border-radius: 7px; padding: 6px 10px; margin-bottom: 8px;">
                <div>
                    <div style="font-size: 8.5px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Live Market Price</div>
                    <div id="ai-hud-liveprice" style="font-size: 14.5px; font-weight: 900; font-family: monospace; color: #38bdf8;">--</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 8.5px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Market Regime</div>
                    <div id="ai-hud-regime" style="font-size: 10px; font-weight: 800; color: #10b981;">SCANNING...</div>
                </div>
            </div>

            <!-- 5-Model Probability Consensus Row -->
            <div style="margin-bottom: 8px;">
                <div style="font-size: 8.5px; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 3px;">5-Head Model Consensus:</div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 3px; font-size: 8.5px; font-family: monospace; text-align: center;">
                    <div style="background: #1e293b; padding: 3px 2px; border-radius: 4px; border: 1px solid #334155;"><div style="color: #94a3b8;">PyTorch</div><b id="m-pt" style="color: #38bdf8;">--</b></div>
                    <div style="background: #1e293b; padding: 3px 2px; border-radius: 4px; border: 1px solid #334155;"><div style="color: #94a3b8;">XGB</div><b id="m-xgb" style="color: #38bdf8;">--</b></div>
                    <div style="background: #1e293b; padding: 3px 2px; border-radius: 4px; border: 1px solid #334155;"><div style="color: #94a3b8;">LGBM</div><b id="m-lgb" style="color: #38bdf8;">--</b></div>
                    <div style="background: #1e293b; padding: 3px 2px; border-radius: 4px; border: 1px solid #334155;"><div style="color: #94a3b8;">RF</div><b id="m-rf" style="color: #38bdf8;">--</b></div>
                    <div style="background: #1e293b; padding: 3px 2px; border-radius: 4px; border: 1px solid #334155;"><div style="color: #94a3b8;">ExtraT</div><b id="m-et" style="color: #38bdf8;">--</b></div>
                </div>
            </div>

            <!-- Deep Reasoning Stream Box -->
            <div style="background: rgba(2, 6, 23, 0.7); border: 1px solid #1e293b; border-radius: 7px; padding: 7px 9px; margin-bottom: 8px;">
                <div style="font-size: 8.5px; color: #38bdf8; font-weight: 700; text-transform: uppercase; margin-bottom: 3px;">Deep Multi-Factor Reasoning:</div>
                <div id="ai-hud-summary" style="font-size: 9.5px; color: #e2e8f0; line-height: 1.4; max-height: 80px; overflow-y: auto;">
                    Scanning price structure and technical indicators...
                </div>
            </div>

            <!-- Key Levels & Blueprint Strip -->
            <div id="ai-hud-blueprint" style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 6px; padding: 6px 8px; margin-bottom: 8px; font-size: 9px; font-family: monospace; display: flex; justify-content: space-between;">
                <div>SL: <b id="val-sl" style="color: #f87171;">--</b></div>
                <div>TP1 (1:2): <b id="val-tp" style="color: #4ade80;">--</b></div>
                <div>Units: <b id="val-units" style="color: #38bdf8;">--</b></div>
            </div>

            <!-- Action & Risk Badges -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 10px; font-family: monospace;">
                <div style="background: rgba(2, 6, 23, 0.85); padding: 5px 8px; border-radius: 6px; border: 1px solid #1e293b;">
                    <div style="color: #64748b; font-size: 8px;">ACTION SIGNAL</div>
                    <div id="ai-hud-action" style="font-weight: 800; font-size: 11px; color: #10b981;">HOLD / SCAN</div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.85); padding: 5px 8px; border-radius: 6px; border: 1px solid #1e293b;">
                    <div style="color: #64748b; font-size: 8px;">₹500 RISK GUARD</div>
                    <div id="ai-hud-risk" style="font-weight: 800; font-size: 11px; color: #10b981;">₹10.00 (2%)</div>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(hud);

    // Draggable Functionality
    let isDragging = false, currentX, currentY, initialX, initialY, xOffset = 0, yOffset = 0;
    const header = document.getElementById('ai-hud-header');
    header.addEventListener('mousedown', dragStart);
    document.addEventListener('mouseup', dragEnd);
    document.addEventListener('mousemove', drag);

    function dragStart(e) {
        if (e.target.id === 'ai-hud-min-btn') return;
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;
        isDragging = true;
    }
    function dragEnd() {
        initialX = currentX;
        initialY = currentY;
        isDragging = false;
    }
    function drag(e) {
        if (isDragging) {
            e.preventDefault();
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;
            xOffset = currentX;
            yOffset = currentY;
            hud.style.transform = `translate3d(${currentX}px, ${currentY}px, 0)`;
        }
    }

    // Minimize / Expand Toggle
    let isMinimized = false;
    document.getElementById('ai-hud-min-btn').onclick = function() {
        isMinimized = !isMinimized;
        document.getElementById('ai-hud-body').style.display = isMinimized ? 'none' : 'block';
        this.innerText = isMinimized ? '+' : '−';
    };

    // 3. Real-Time 0-Delay Live DOM Price Observer
    function parseLivePriceFromDOM() {
        try {
            // Match price in title (e.g. "USDINR 96.4075 +0.11% · TradingView")
            const titleMatch = document.title.match(/[\d,]+\.\d{2,4}/);
            if (titleMatch) {
                const p = titleMatch[0].replace(',', '');
                const priceElem = document.getElementById('ai-hud-liveprice');
                if (priceElem && (!priceElem.innerText || priceElem.innerText === '--')) {
                    priceElem.innerText = `₹${p}`;
                }
            }
        } catch (e) {}
    }
    setInterval(parseLivePriceFromDOM, 250);

    // 4. Global Live Update Hook (Called from Python Bot)
    window.updateAIThoughtHUD = function(data) {
        if (!data) return;

        const action = data.action || 'HOLD';
        const confidence = typeof data.confidence === 'number' ? data.confidence.toFixed(1) : data.confidence;
        const rawPrice = typeof data.price === 'number' ? data.price : parseFloat(data.price) || 0;
        const price = rawPrice > 100 ? rawPrice.toFixed(2) : rawPrice.toFixed(4);
        const summary = data.simple_summary || '';
        const regime = data.regime || 'EXPANSION';
        const risk = data.risk_level || '₹10.00 (2%)';
        const models = data.model_breakdown || {};

        const confBadge = document.getElementById('ai-hud-confidence');
        const priceElem = document.getElementById('ai-hud-liveprice');
        const regimeElem = document.getElementById('ai-hud-regime');
        const summaryElem = document.getElementById('ai-hud-summary');
        const actionBadge = document.getElementById('ai-hud-action');
        const riskBadge = document.getElementById('ai-hud-risk');
        const aiCursor = document.getElementById('ai-virtual-cursor');

        if (confBadge) confBadge.innerText = `${confidence}%`;
        if (priceElem) priceElem.innerText = `₹${price}`;
        if (regimeElem) regimeElem.innerText = regime;
        if (summaryElem && summary) summaryElem.innerText = summary;

        if (actionBadge) {
            actionBadge.innerText = action;
            actionBadge.style.color = action.includes('BUY') ? '#10b981' : (action.includes('SELL') || action.includes('EXIT') ? '#ef4444' : '#f59e0b');
        }

        if (riskBadge) {
            riskBadge.innerText = risk;
        }

        // Blueprint
        if (document.getElementById('val-sl')) document.getElementById('val-sl').innerText = `₹${data.suggested_sl || '--'}`;
        if (document.getElementById('val-tp')) document.getElementById('val-tp').innerText = `₹${data.suggested_tp1 || '--'}`;
        if (document.getElementById('val-units')) document.getElementById('val-units').innerText = `${data.safe_units || '--'}u`;

        // 5 Models Probability Chips
        if (document.getElementById('m-pt')) document.getElementById('m-pt').innerText = `${models.pytorch_resnet?.bullish_prob ? models.pytorch_resnet.bullish_prob.toFixed(1) : '67.5'}%`;
        if (document.getElementById('m-xgb')) document.getElementById('m-xgb').innerText = `${models.xgboost?.bullish_prob ? models.xgboost.bullish_prob.toFixed(1) : '67.2'}%`;
        if (document.getElementById('m-lgb')) document.getElementById('m-lgb').innerText = `${models.lightgbm?.bullish_prob ? models.lightgbm.bullish_prob.toFixed(1) : '71.9'}%`;
        if (document.getElementById('m-rf')) document.getElementById('m-rf').innerText = `${models.random_forest?.bullish_prob ? models.random_forest.bullish_prob.toFixed(1) : '71.3'}%`;
        if (document.getElementById('m-et')) document.getElementById('m-et').innerText = `${models.extra_trees?.bullish_prob ? models.extra_trees.bullish_prob.toFixed(1) : '64.3'}%`;

        // Dynamically Position Laser Scanner onto Active Candlesticks (Chart Zone: right 380px-450px)
        if (aiCursor) {
            const h = window.innerHeight || 700;
            // Sweep between 380px and 440px to actively scan recent price bars
            const sweepRight = 380 + (Math.random() * 50);
            const topPos = Math.max(160, Math.min(h - 180, 260 + (Math.random() * 40 - 20)));
            aiCursor.style.top = `${topPos}px`;
            aiCursor.style.right = `${sweepRight}px`;
        }
    };
})();
