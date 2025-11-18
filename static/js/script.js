// Chatbot functionality with localStorage persistence
let chatbotOpen = false;
let chatContext = null;

// Generate unique user ID for chat storage
function getUserId() {
    let userId = localStorage.getItem('chatbot_user_id');
    if (!userId) {
        userId = 'user_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('chatbot_user_id', userId);
    }
    return userId;
}

// Get chat history from localStorage
function getChatHistory() {
    const userId = getUserId();
    const history = localStorage.getItem(`chat_history_${userId}`);
    return history ? JSON.parse(history) : [];
}

// Save chat history to localStorage
function saveChatHistory(history) {
    const userId = getUserId();
    localStorage.setItem(`chat_history_${userId}`, JSON.stringify(history));
}

// Load chat history when page loads
function loadChatHistory() {
    const history = getChatHistory();
    const messagesContainer = document.getElementById('chatbotMessages');

    if (messagesContainer && history.length > 0) {
        messagesContainer.innerHTML = '';
        history.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.type} fade-in`;
            messageDiv.innerHTML = `<strong>${msg.sender}:</strong><br>${formatResponse(msg.message)}`;
            messagesContainer.appendChild(messageDiv);
        });
        scrollToBottom();
    }
}

// Reset chat history
function resetChatHistory() {
    const userId = getUserId();
    localStorage.removeItem(`chat_history_${userId}`);
    localStorage.removeItem(`chat_context_${userId}`);

    const messagesContainer = document.getElementById('chatbotMessages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
        // Add welcome message
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message bot fade-in';
        welcomeDiv.innerHTML = `
            <strong><i class="fas fa-robot me-1"></i>Assistant:</strong><br>
            Hello! I'm your college admission assistant. I can help with:<br>
            • Engineering (JEE, CET) predictions<br>
            • Medical (NEET) predictions<br>
            • Cutoffs and admission procedures<br><br>
            How can I assist you today?
        `;
        messagesContainer.appendChild(welcomeDiv);
        scrollToBottom();
    }

    chatContext = null;
}

function toggleChatbot() {
    const chatbotContainer = document.querySelector('.chatbot-container');
    const chatbotToggle = document.querySelector('.chatbot-toggle');

    if (chatbotOpen) {
        chatbotContainer.classList.remove('show');
        chatbotToggle.innerHTML = '<i class="fas fa-robot me-2"></i>AI Assistant';
        chatContext = null;
    } else {
        chatbotContainer.classList.add('show');
        chatbotToggle.innerHTML = '<i class="fas fa-times me-2"></i>Close';
        loadChatHistory();
        scrollToBottom();
    }

    chatbotOpen = !chatbotOpen;
}

function closeChatbot() {
    const chatbotContainer = document.querySelector('.chatbot-container');
    const chatbotToggle = document.querySelector('.chatbot-toggle');

    chatbotContainer.classList.remove('show');
    chatbotToggle.innerHTML = '<i class="fas fa-robot me-2"></i>AI Assistant';
    chatbotOpen = false;
    chatContext = null;
}

function quickOption(option) {
    console.log('Quick option selected:', option);
    document.getElementById('chatbotInput').value = option;
    sendChatbotMessage();
}

function sendChatbotMessage() {
    const input = document.getElementById('chatbotInput');
    const message = input.value.trim();
    const messages = document.getElementById('chatbotMessages');

    if (message === '') return;

    console.log('Sending message:', message, 'Context:', chatContext);

    // Add user message
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user fade-in';
    userMessageDiv.innerHTML = `<strong><i class="fas fa-user me-1"></i>You:</strong><br>${message}`;
    messages.appendChild(userMessageDiv);

    // Save to history
    const history = getChatHistory();
    history.push({
        type: 'user',
        sender: 'You',
        message: message,
        timestamp: new Date().toISOString()
    });
    saveChatHistory(history);

    // Clear input
    input.value = '';

    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot fade-in';
    typingDiv.innerHTML = '<strong><i class="fas fa-robot me-1"></i>Assistant:</strong><br><i class="fas fa-ellipsis-h"></i> Thinking...';
    messages.appendChild(typingDiv);
    scrollToBottom();

    // Send to server
    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            context: chatContext
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Received response:', data);

        // Remove typing indicator
        if (messages.contains(typingDiv)) {
            messages.removeChild(typingDiv);
        }

        // Add bot response
        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'message bot fade-in';

        let responseHTML = `<strong><i class="fas fa-robot me-1"></i>Assistant:</strong><br>${formatResponse(data.response)}`;

        // Add options if available
        if (data.options && data.options.length > 0) {
            responseHTML += `<br><div class="mt-2 quick-options">`;
            data.options.forEach(option => {
                responseHTML += `<button class="btn btn-sm btn-outline-dark me-1 mb-1" onclick="quickOption('${option}')">${option}</button>`;
            });
            responseHTML += `</div>`;
        }

        botMessageDiv.innerHTML = responseHTML;
        messages.appendChild(botMessageDiv);

        // Save bot response to history
        history.push({
            type: 'bot',
            sender: 'Assistant',
            message: data.response,
            timestamp: new Date().toISOString(),
            options: data.options || []
        });
        saveChatHistory(history);

        // Update context for next message
        if (data.context) {
            chatContext = data.context;
            // Save context to localStorage
            const userId = getUserId();
            localStorage.setItem(`chat_context_${userId}`, JSON.stringify(data.context));
        } else {
            chatContext = null;
            const userId = getUserId();
            localStorage.removeItem(`chat_context_${userId}`);
        }

        // Scroll to bottom
        scrollToBottom();
    })
    .catch(error => {
        console.error('Chatbot error:', error);

        // Remove typing indicator
        if (messages.contains(typingDiv)) {
            messages.removeChild(typingDiv);
        }

        const errorDiv = document.createElement('div');
        errorDiv.className = 'message bot fade-in';
        errorDiv.innerHTML = '<strong><i class="fas fa-robot me-1"></i>Assistant:</strong><br>Sorry, I encountered an error. Please try again.';
        messages.appendChild(errorDiv);

        // Save error to history
        history.push({
            type: 'bot',
            sender: 'Assistant',
            message: 'Sorry, I encountered an error. Please try again.',
            timestamp: new Date().toISOString()
        });
        saveChatHistory(history);

        // Reset context on error
        chatContext = null;
        const userId = getUserId();
        localStorage.removeItem(`chat_context_${userId}`);

        scrollToBottom();
    });
}

function formatResponse(response) {
    // Format the response with better readability
    if (!response) return 'No response received.';

    // Replace newlines with HTML line breaks
    return response.replace(/\n/g, '<br>')
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                  .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

function scrollToBottom() {
    const messages = document.getElementById('chatbotMessages');
    if (messages) {
        messages.scrollTop = messages.scrollHeight;
    }
}

// Event listeners for chatbot
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing chatbot...');

    const chatbotToggle = document.querySelector('.chatbot-toggle');
    const chatbotInput = document.getElementById('chatbotInput');
    const sendBtn = document.getElementById('sendBtn');

    // Load chat context from localStorage
    const userId = getUserId();
    const savedContext = localStorage.getItem(`chat_context_${userId}`);
    if (savedContext) {
        chatContext = JSON.parse(savedContext);
    }

    if (chatbotToggle) {
        console.log('Found chatbot toggle button');
        chatbotToggle.addEventListener('click', toggleChatbot);
    } else {
        console.log('Chatbot toggle button not found');
    }

    if (chatbotInput) {
        console.log('Found chatbot input');
        chatbotInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatbotMessage();
            }
        });
    } else {
        console.log('Chatbot input not found');
    }

    if (sendBtn) {
        console.log('Found send button');
    }

    // Initialize with welcome message if no messages exist
    const messages = document.getElementById('chatbotMessages');
    if (messages && messages.children.length === 0) {
        const history = getChatHistory();
        if (history.length === 0) {
            const welcomeMessage = document.createElement('div');
            welcomeMessage.className = 'message bot fade-in';
            welcomeMessage.innerHTML = `
                <strong><i class="fas fa-robot me-1"></i>Assistant:</strong><br>
                Hello! I'm your college admission assistant. I can help with:<br>
                • Engineering (JEE, CET) predictions<br>
                • Medical (NEET) predictions<br>
                • Cutoffs and admission procedures<br><br>
                How can I assist you today?
            `;
            messages.appendChild(welcomeMessage);

            // Save welcome message to history
            history.push({
                type: 'bot',
                sender: 'Assistant',
                message: "Hello! I'm your college admission assistant. I can help with:\n• Engineering (JEE, CET) predictions\n• Medical (NEET) predictions\n• Cutoffs and admission procedures\n\nHow can I assist you today?",
                timestamp: new Date().toISOString()
            });
            saveChatHistory(history);
        } else {
            // Load existing history
            loadChatHistory();
        }
    }

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const inputs = form.querySelectorAll('input[required], select[required]');
            let valid = true;

            inputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!valid) {
                e.preventDefault();
                showAlert('Please fill in all required fields.', 'danger');
            }
        });
    });

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.querySelector('main').insertBefore(alertDiv, document.querySelector('main').firstChild);
}

// Prevent back navigation after logout
window.addEventListener('pageshow', function(event) {
    if (event.persisted || (window.performance && window.performance.navigation.type === 2)) {
        window.location.reload();
    }
});

if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
}

// Debug function to check chatbot elements
function debugChatbot() {
    console.log('=== CHATBOT DEBUG INFO ===');
    console.log('chatbotOpen:', chatbotOpen);
    console.log('chatContext:', chatContext);
    console.log('Toggle button:', document.querySelector('.chatbot-toggle'));
    console.log('Input field:', document.getElementById('chatbotInput'));
    console.log('Messages container:', document.getElementById('chatbotMessages'));
    console.log('Chatbot container:', document.querySelector('.chatbot-container'));
    console.log('Chat history:', getChatHistory());
    console.log('==========================');
}

// Make debug function available globally
window.debugChatbot = debugChatbot;
window.resetChatHistory = resetChatHistory;