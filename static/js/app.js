/**
 * Form Intelligence System - App Logic
 */

// State
const state = {
  currentForm: null,
  analysisData: null,
  answers: {},
  profiles: [],
  history: []
};

// Elements
const els = {
  scanUrl: document.getElementById('scan-url'),
  btnScan: document.getElementById('btn-scan'),
  viewScan: document.getElementById('view-scan'),
  viewAnalysis: document.getElementById('view-analysis'),
  viewGenerate: document.getElementById('view-generate'),
  viewHistory: document.getElementById('view-history'),
  
  // Dashboard stats
  statTotalQuestions: document.getElementById('stat-total-questions'),
  statAvgLen: document.getElementById('stat-avg-len'),
  statDifficulty: document.getElementById('stat-difficulty'),
  statTime: document.getElementById('stat-time'),
  
  // Containers
  questionsContainer: document.getElementById('questions-container'),
  answerListContainer: document.getElementById('answer-list-container'),
  historyContainer: document.getElementById('history-container'),
  
  // Navigation
  navLinks: document.querySelectorAll('.nav-item')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupScanForm();
  loadHistory();
});

// Notifications
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  
  let icon = '✓';
  if (type === 'error') icon = '✕';
  if (type === 'warning') icon = '⚠';
  
  toast.innerHTML = `
    <div style="font-weight: 800; font-size: 1.2rem;">${icon}</div>
    <div>${message}</div>
  `;
  
  container.appendChild(toast);
  
  // Trigger animation
  requestAnimationFrame(() => {
    toast.classList.add('show');
    toast.style.transform = 'translateX(0)';
  });
  
  // Remove after 3s
  setTimeout(() => {
    toast.style.transform = 'translateX(120%)';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Navigation
function setupNavigation() {
  els.navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.dataset.target;
      
      // Update Active Navigation
      els.navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      
      // Update Views
      document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));
      document.getElementById(`view-${target}`).classList.add('active');
    });
  });
}

// Scan Logic
function setupScanForm() {
  els.btnScan.addEventListener('click', async () => {
    const url = els.scanUrl.value.trim();
    if (!url) {
      showToast('Please enter a valid form URL', 'warning');
      return;
    }
    
    // UI Loading state
    els.btnScan.classList.add('loading');
    els.btnScan.innerHTML = '<span class="spinner"></span> Scanning Form...';
    
    try {
      const resp = await fetch('/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      
      const data = await resp.json();
      
      if (!resp.ok) {
        throw new Error(data.detail || 'Failed to scan form');
      }
      
      state.currentForm = data.form;
      state.analysisData = data.analysis;
      
      showToast('Form successfully scanned & analyzed!', 'success');
      
      // Update UI
      updateDashboardData();
      renderQuestionsList();
      
      // Setup generate listener before calling it
      const btnGenerate = document.getElementById('btn-generate-ai');
      if (btnGenerate && !btnGenerate.dataset.hasListener) {
          btnGenerate.addEventListener('click', generateAnswers);
          btnGenerate.dataset.hasListener = "true";
      }

      const btnSave = document.getElementById('btn-save-custom');
      if (btnSave && !btnSave.dataset.hasListener) {
          btnSave.addEventListener('click', saveCustomAnswers);
          btnSave.dataset.hasListener = "true";
      }

      // Automatically run AI generation to provide an all-in-one click experience!
      generateAnswers();
      
      // Auto-switch to analysis view
      document.querySelector('[data-target="analysis"]').click();
      
      // Reload history to show recent scan
      loadHistory();
      
    } catch (err) {
      console.error(err);
      showToast(err.message, 'error');
    } finally {
      els.btnScan.classList.remove('loading');
      els.btnScan.innerHTML = '<svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg> Auto Scan & Generate';
    }
  });

  // Export Results Listener
  document.getElementById('btn-export')?.addEventListener('click', () => {
     if (!state.currentForm) return showToast('No form data to export', 'warning');
     
     const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
        form: state.currentForm,
        analysis: state.analysisData,
        answers: state.answers
     }, null, 2));
     
     const dlAnchorElem = document.createElement('a');
     dlAnchorElem.setAttribute("href", dataStr);
     dlAnchorElem.setAttribute("download", `form_scan_${new Date().getTime()}.json`);
     dlAnchorElem.click();
  });
}

function updateDashboardData() {
  if (!state.analysisData) return;
  const analysis = state.analysisData;
  const form = state.currentForm;
  
  // Set simple stats
  els.statTotalQuestions.textContent = analysis.total_questions;
  els.statAvgLen.textContent = analysis.avg_question_length?.toFixed(1) + ' words';
  
  if (analysis.difficulty) {
    els.statDifficulty.textContent = analysis.difficulty.level;
    els.statTime.textContent = '~' + analysis.difficulty.estimated_time_minutes + ' min';
    
    // Color code difficulty
    if (analysis.difficulty.level === 'Easy') els.statDifficulty.style.color = 'var(--success)';
    if (analysis.difficulty.level === 'Medium') els.statDifficulty.style.color = 'var(--warning)';
    if (analysis.difficulty.level === 'Hard') els.statDifficulty.style.color = 'var(--danger)';
  }
}

function getPlatformIcon(platform) {
   if (platform === 'google') {
       return `<span class="platform-icon google"><svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c6.87 0 11.4-4.57 11.4-11.587 0-.74-.067-1.48-.187-2.187H12.48z"/></svg> Google Forms</span>`;
   }
   if (platform === 'microsoft') {
       return `<span class="platform-icon microsoft"><svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M11.4 24H0V12.6h11.4V24zM24 24H12.6V12.6H24V24zM11.4 11.4H0V0h11.4v11.4zm12.6 0H12.6V0H24v11.4z"/></svg> Microsoft Forms</span>`;
   }
   return platform || 'Unknown';
}

function renderQuestionsList() {
  if (!state.currentForm) return;
  
  const form = state.currentForm;
  
  const titleHtml = `
    <div style="margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem;">
      <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem">${form.title || 'Untitled Form'}</h2>
      <p style="color: var(--text-muted); margin-bottom: 1rem">${form.description || 'No description provided.'}</p>
      <div style="display: flex; gap: 1rem;">
         <span style="background: rgba(255,255,255,0.1); padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem;">
            ${getPlatformIcon(form.platform)}
         </span>
         <span style="background: rgba(255,255,255,0.1); padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem;">
            ${form.questions.length} Questions
         </span>
      </div>
    </div>
  `;
  
  let questionsHtml = '';
  
  form.questions.forEach((q, i) => {
    let optionsHtml = '';
    if (q.options && q.options.length > 0) {
      optionsHtml = `<div class="q-options">
        ${q.options.map(o => `<div class="q-option">${o}</div>`).join('')}
      </div>`;
    }
    
    questionsHtml += `
      <div class="question-item ${q.required ? 'required' : ''}">
        <div class="q-header">
          <div class="q-title">${i+1}. ${q.question}</div>
          <div class="q-type-badge">${q.type.replace('_', ' ')}</div>
        </div>
        ${optionsHtml}
        <div class="answer-field" id="ans-container-${i+1}">
          <label style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem;">
            Custom Answer (Overrides AI)
          </label>
          <input type="text" class="input-control" id="custom-answer-${i+1}" placeholder="Optional custom answer...">
          <div class="ai-answer" id="ai-answer-${i+1}"></div>
        </div>
      </div>
    `;
  });
  
  els.questionsContainer.innerHTML = titleHtml + questionsHtml;
}

function saveCustomAnswers() {
    if (!state.currentForm) return;
    showToast('Custom answers saved locally!', 'success');
}

async function generateAnswers() {
  if (!state.currentForm) return;
  
  const btn = document.getElementById('btn-generate-ai');
  btn.classList.add('loading');
  btn.innerHTML = '<span class="spinner"></span> Generating...';
  
  try {
    // Gather custom answers
    const customAnswers = {};
    for (let i = 1; i <= state.currentForm.questions.length; i++) {
      const val = document.getElementById(`custom-answer-${i}`).value.trim();
      if (val) customAnswers[`question_${i}`] = val;
    }
    
    // Gather context
    const reqBody = {
       questions: state.currentForm.questions,
       custom_answers: customAnswers,
       profile: document.getElementById('ai-profile').value,
       tone: document.getElementById('ai-tone').value,
       persona: document.getElementById('ai-persona').value.trim(),
       context: document.getElementById('ai-context').value.trim()
    };
    
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody)
    });
    
    const data = await resp.json();
    
    if (!resp.ok) throw new Error(data.detail || 'Failed to generate answers');
    
    state.answers = data.answers;
    
    // Inject answers into UI
    Object.keys(state.answers).forEach(key => {
        const idx = key.split('_')[1];
        const ansElem = document.getElementById(`ai-answer-${idx}`);
        if(ansElem) {
            ansElem.textContent = state.answers[key];
            ansElem.classList.add('visible');
            
            // Highlight custom answers
            if(customAnswers[key]) {
                ansElem.style.background = 'rgba(236, 72, 153, 0.1)';
                ansElem.style.borderLeftColor = 'var(--secondary)';
                ansElem.innerHTML = `<strong>Custom Override:</strong> ${state.answers[key]}`;
            } else {
                ansElem.style.background = 'rgba(16, 185, 129, 0.1)';
                ansElem.style.borderLeftColor = 'var(--success)';
                ansElem.innerHTML = `<strong>AI Generated:</strong> ${state.answers[key]}`;
            }
        }
    });
    
    showToast('Answers generated successfully!', 'success');
    
  } catch (err) {
    console.error(err);
    showToast(err.message, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = '<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg> Generate AI Answers';
  }
}

async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    const data = await resp.json();
    
    if (data.history && data.history.length > 0) {
      let trs = '';
      data.history.forEach(item => {
        const date = new Date(item.created_at).toLocaleString();
        trs += `
          <tr>
            <td>${item.id}</td>
            <td><strong>${item.title || 'Untitled Form'}</strong><br><small style="color:var(--text-muted)">${item.url.substring(0,40)}...</small></td>
            <td>${getPlatformIcon(item.platform)}</td>
            <td>${item.question_count}</td>
            <td>${date}</td>
            <td><button class="btn btn-secondary" style="padding: 0.25rem 0.75rem; font-size: 0.8rem;" onclick="loadHistoryItem(${item.id})">Load</button></td>
          </tr>
        `;
      });
      document.getElementById('history-table-body').innerHTML = trs;
    } else {
      document.getElementById('history-table-body').innerHTML = `<tr><td colspan="6" class="text-center">No scan history available</td></tr>`;
    }
  } catch (err) {
    console.error('Failed to load history', err);
  }
}

window.loadHistoryItem = async function(id) {
    try {
        const resp = await fetch(`/api/history/${id}`);
        const data = await resp.json();
        
        state.currentForm = data.form_data;
        state.analysisData = data.analysis;
        state.answers = data.answers || {};
        
        updateDashboardData();
        renderQuestionsList();
        
        document.querySelector('[data-target="analysis"]').click();
        showToast('Loaded specific form scan from history', 'success');
        
    } catch(err) {
        showToast('Error loading history item', 'error');
    }
}
