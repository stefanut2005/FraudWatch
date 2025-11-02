// Live Transactions Manager
// Use the dashboard proxy (same-origin) so we avoid CORS and rely on server-side proxy logic
const LIVE_TX_API = "/api/live/transaction";
const INITIAL_COUNT = 5;
const UPDATE_INTERVAL = 5000; // 5 seconds

let liveTransactions = [];
let updateInterval = null;

// Load initial 5 transactions
async function loadInitialTransactions() {
    console.log('Loading initial transactions...');
    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = "Loading...";
    
    const promises = [];
    for (let i = 0; i < INITIAL_COUNT; i++) {
        console.log(`Loading transaction ${i + 1}/${INITIAL_COUNT}`);
        promises.push(fetchAndAddTransaction());
        // Small delay between initial requests to avoid overwhelming server
        await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    await Promise.all(promises);
    console.log(`Loaded ${liveTransactions.length} transactions`);
    if (statusText) statusText.textContent = "Live";
}

// Fetch single transaction and add to table
async function fetchAndAddTransaction() {
    try {
        console.log('Fetching transaction from:', LIVE_TX_API);
        const response = await fetch(LIVE_TX_API);
        console.log('Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Failed to fetch transaction:', response.status, errorText);
            return;
        }
        
        // Try to parse JSON; if server returned plain text fallback, handle gracefully
        let data = null;
        try {
            data = await response.json();
        } catch (e) {
            const raw = await response.text();
            console.warn('Response was not valid JSON, raw content:', raw);
            // Attempt to parse raw as JSON anyway
            try {
                data = JSON.parse(raw);
            } catch (e2) {
                console.error('Unable to parse backend response as JSON:', e2);
                const tbody = document.getElementById('liveTxBody');
                if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#ef4444;">Backend returned invalid JSON. Check server logs.</td></tr>`;
                return;
            }
        }

        console.log('Received data:', data);

        if (!data || !data.transaction) {
            console.error('No transaction in response:', data);
            const tbody = document.getElementById('liveTxBody');
            if (tbody && liveTransactions.length === 0) {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#ef4444;">No transaction data returned. Check server logs.</td></tr>`;
            }
            return;
        }
        
        const tx = data.transaction;
        const risk = data.risk || (data.fraud_probability ? data.fraud_probability * 100 : 0);
        const fraudDetected = data.fraud_detected || false;
        
        // Add to beginning of array
        liveTransactions.unshift({
            id: tx.id,
            time: tx.trans_date_trans_time,
            merchant: tx.merchant,
            amount: tx.amt,
            category: tx.category,
            risk: risk,
            fraudDetected: fraudDetected,
            raw: tx
        });
        
        // Keep only last 5 transactions (most recent first)
        if (liveTransactions.length > 5) {
            liveTransactions = liveTransactions.slice(0, 5);
        }
        
        renderTransactions();
        
    } catch (error) {
        console.error('Error fetching transaction:', error);
        const tbody = document.getElementById('liveTxBody');
        if (tbody && liveTransactions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 2rem; color: #ef4444;">Error: ${error.message}. Check console for details.</td></tr>`;
        }
    }
}

// Render transactions table
function renderTransactions() {
    const tbody = document.getElementById('liveTxBody');
    if (!tbody) {
        console.error('liveTxBody not found when rendering!');
        return;
    }
    
    if (liveTransactions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem; color: var(--text-dim);">No transactions yet. Check console for errors.</td></tr>';
        return;
    }
    
    tbody.innerHTML = liveTransactions.map(tx => {
        const riskClass = tx.risk >= 80 ? 'risk-high' : tx.risk >= 40 ? 'risk-mid' : 'risk-low';
        const statusClass = tx.risk >= 80 ? 'status-high' : tx.risk >= 40 ? 'status-mid' : 'status-low';
        const statusText = tx.fraudDetected ? 'Fraud' : 'Safe';
        
        return `
            <tr class="tx-row" data-tx-id="${tx.id}">
                <td>${tx.time}</td>
                <td>${tx.id}</td>
                <td>${tx.merchant}</td>
                <td>$${tx.amount.toFixed(2)}</td>
                <td>${tx.category}</td>
                <td>
                    <span class="risk-chip ${riskClass}">
                        ${tx.risk.toFixed(1)}%
                    </span>
                </td>
                <td>
                    <span class="status-badge ${statusClass}">
                        ${statusText}
                    </span>
                </td>
            </tr>
        `;
    }).join('');
}

// Start live updates
function startLiveUpdates() {
    if (updateInterval) return; // Already running
    
    updateInterval = setInterval(() => {
        fetchAndAddTransaction();
    }, UPDATE_INTERVAL);
}

// Stop live updates
function stopLiveUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Live transactions page loaded');
    const tbody = document.getElementById('liveTxBody');
    if (!tbody) {
        console.error('liveTxBody element not found!');
        return;
    }
    
    loadInitialTransactions().then(() => {
        console.log('Initial load complete, starting live updates');
        startLiveUpdates();
    }).catch(error => {
        console.error('Error loading initial transactions:', error);
    });
});

// Stop when page unloads
window.addEventListener('beforeunload', () => {
    stopLiveUpdates();
});

