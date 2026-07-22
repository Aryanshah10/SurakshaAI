const alerts = [
  { c: 'danger', t: 'FLAGGED', m: 'Digital arrest script detected · Pune circle · call auto-terminated' },
  { c: 'alert', t: 'REVIEW', m: 'Spoofed number pattern · 3 reports linked · Hyderabad' },
  { c: 'safe', t: 'CLEARED', m: 'Payment request verified · genuine merchant · Kochi' },
  { c: 'danger', t: 'FLAGGED', m: 'Counterfeit ₹500 batch detected · POS terminal · Nagpur' },
  { c: 'alert', t: 'MULE ACCT', m: 'New node linked to active fraud ring #4471' },
  { c: 'safe', t: 'FILED', m: 'NCRB report submitted · victim guided in Tamil' },
];

const track = document.getElementById('tickerTrack');
if (track) {
  const build = () => alerts.map((item) => `<div class="ticker-item ${item.c}"><span class="dot"></span><b>${item.t}</b>${item.m}</div>`).join('');
  track.innerHTML = build() + build();
}

const modal = document.getElementById('authModal');
const openButtons = document.querySelectorAll('[data-open-auth]');
const closeButton = document.querySelector('.modal-close');
const scrollTargets = document.querySelectorAll('[data-scroll-target]');
const navToggle = document.querySelector('.nav-toggle');
const navMenu = document.getElementById('navMenu');
const navLinks = document.querySelectorAll('.nav-links a');

const openModal = () => {
  if (!modal) return;
  modal.hidden = false;
  document.body.classList.add('modal-open');
};

const closeModal = () => {
  if (!modal) return;
  modal.hidden = true;
  document.body.classList.remove('modal-open');
};

const closeNavMenu = () => {
  if (!navMenu || !navToggle) return;
  navMenu.classList.remove('is-open');
  navToggle.setAttribute('aria-expanded', 'false');
  navToggle.setAttribute('aria-label', 'Open navigation menu');
};

const toggleNavMenu = () => {
  if (!navMenu || !navToggle) return;
  const isOpen = navMenu.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(isOpen));
  navToggle.setAttribute('aria-label', isOpen ? 'Close navigation menu' : 'Open navigation menu');
};

const scrollToSection = (selector) => {
  const target = document.querySelector(selector);
  if (!target) return;
  target.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

openButtons.forEach((button) => button.addEventListener('click', openModal));
if (closeButton) closeButton.addEventListener('click', closeModal);
if (modal) {
  modal.addEventListener('click', (event) => {
    if (event.target === modal) closeModal();
  });
}

scrollTargets.forEach((button) => {
  button.addEventListener('click', () => {
    const target = button.getAttribute('data-scroll-target');
    closeModal();
    closeNavMenu();
    if (target) scrollToSection(target);
  });
});

if (navToggle) navToggle.addEventListener('click', toggleNavMenu);

navLinks.forEach((link) => {
  link.addEventListener('click', closeNavMenu);
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeModal();
    closeNavMenu();
  }
});

window.addEventListener('resize', () => {
  if (window.innerWidth > 820) closeNavMenu();
});

// --- Login page handler (login.html only) ---
const loginForm = document.getElementById('loginForm');
const loginError = document.getElementById('loginError');
const loginSuccess = document.getElementById('loginSuccess');
const loginSubmit = document.getElementById('loginSubmit');

const API_BASE = 'http://localhost:8000';

const hideLoginMessages = () => {
  if (loginError) loginError.hidden = true;
  if (loginSuccess) loginSuccess.hidden = true;
};

const showLoginError = (message) => {
  if (!loginError) return;
  loginError.textContent = message;
  loginError.hidden = false;
  if (loginSuccess) loginSuccess.hidden = true;
};

const showLoginSuccess = (message) => {
  if (!loginSuccess) return;
  loginSuccess.textContent = message;
  loginSuccess.hidden = false;
  if (loginError) loginError.hidden = true;
};

if (loginForm) {
  loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideLoginMessages();

    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const username = usernameInput?.value.trim() ?? '';
    const password = passwordInput?.value ?? '';

    if (!username || !password) {
      showLoginError('Please enter both username and password.');
      return;
    }

    if (loginSubmit) {
      loginSubmit.disabled = true;
      loginSubmit.textContent = 'Signing in…';
    }

    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      let data = {};
      try {
        data = await response.json();
      } catch {
        showLoginError('Invalid username or password');
        return;
      }

      if (response.ok && data.success) {

    // Save officer information
    localStorage.setItem(
        "officer",
        JSON.stringify({
            username: data.username,
            name: data.name,
            email: data.email,
            department: data.department,
            badge_id: data.badge_id,
            station: data.station,
            role: data.role
        })
    );
    localStorage.setItem("token", data.access_token);
    localStorage.setItem("access_token", data.access_token);

    showLoginSuccess("Login successful. Redirecting...");

    setTimeout(() => {

        // Go to officer dashboard
        window.location.href = "officer.html";

    }, 900);

    return;
}

      showLoginError(data.message || 'Invalid username or password');
    } catch {
      showLoginError('Unable to reach the login server. Is the backend running on port 8000?');
    } finally {
      if (loginSubmit) {
        loginSubmit.disabled = false;
        loginSubmit.textContent = 'Sign In';
      }
    }
  });
}

// --- Interactive Chatbot Handler (index.html) ---
const chatForm = document.getElementById('chatForm');
const chatHistory = document.getElementById('chatHistory');
const chatInput = document.getElementById('chatInput');

const scrollChat = () => {
  if (chatHistory) {
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
};

window.copyNcrbDraft = (encodedText) => {
  const text = decodeURIComponent(encodedText);
  navigator.clipboard.writeText(text).then(() => {
    alert("NCRB complaint draft copied to clipboard!");
  }).catch(err => {
    console.error("Clipboard copy failed:", err);
  });
};

if (chatForm && chatHistory && chatInput) {
  chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    // Add user bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'bubble user';
    userBubble.textContent = text;
    chatHistory.appendChild(userBubble);
    chatInput.value = '';
    scrollChat();

    // Add loading bubble
    const loadingBubble = document.createElement('div');
    loadingBubble.className = 'bubble bot loading';
    loadingBubble.id = 'chatLoading';
    loadingBubble.textContent = 'SurakshaAI is assessing...';
    chatHistory.appendChild(loadingBubble);
    scrollChat();

    try {
      const response = await fetch(`${API_BASE}/api/scam/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });

      const loader = document.getElementById('chatLoading');
      if (loader) loader.remove();

      if (response.ok) {
        const data = await response.json();
        const verdictClass = data.risk_level === 'critical' || data.risk_level === 'high' ? 'danger-verdict' : 'safe-verdict';
        const labelMap = {
          'low': 'Safe Check · Low Risk',
          'medium': 'Suspicious · Medium Risk',
          'high': 'Scam Detected · High Risk',
          'critical': 'Scam Detected · Critical'
        };
        const label = labelMap[data.risk_level] || 'Analysis Complete';
        
        let responseHtml = `<span class="verdict ${verdictClass}">${label}</span>${data.recommended_action}`;
        
        if (data.red_flags && data.red_flags.length > 0) {
          responseHtml += `<br><br><small style="color: var(--alert); display: block; font-family: var(--font-mono); font-size: 11px;">MATCHED PATTERNS: ${data.red_flags.join(', ')}</small>`;
        }
        
        if (data.ncrb_draft) {
          responseHtml += `<br><button type="button" class="btn-ncrb-copy" onclick="copyNcrbDraft('${encodeURIComponent(data.ncrb_draft)}')">Copy NCRB Report Draft</button>`;
        }

        const botBubble = document.createElement('div');
        botBubble.className = 'bubble bot';
        botBubble.innerHTML = responseHtml;
        chatHistory.appendChild(botBubble);
      } else {
        const botBubble = document.createElement('div');
        botBubble.className = 'bubble bot';
        botBubble.innerHTML = `<span class="verdict danger-verdict">Error</span>Failed to process message risk checks.`;
        chatHistory.appendChild(botBubble);
      }
    } catch (err) {
      const loader = document.getElementById('chatLoading');
      if (loader) loader.remove();

      const botBubble = document.createElement('div');
      botBubble.className = 'bubble bot';
      botBubble.innerHTML = `<span class="verdict danger-verdict">Offline</span>Unable to reach the security core. Call 1930 for immediate assistance.`;
      chatHistory.appendChild(botBubble);
    } finally {
      scrollChat();
    }
  });
}
