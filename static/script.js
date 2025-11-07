


// // Global state
// let currentQuizData = null;
// let currentQuestions = [];
// let userAnswers = {};



// function getBaseUrl() {
//   const origin = window.location.origin;
//   const hostname = window.location.hostname;
  
//   // Check if we're on localhost
//   if (hostname === "localhost" || hostname === "127.0.0.1") {
//     // Use local FastAPI
//     return "http://127.0.0.1:8000";
//   }
  
//   // Otherwise, use the current origin (ngrok or any external URL)
//   return origin;
// }



// const BASE_URL = getBaseUrl();
// // Navigation functionality
// document.addEventListener('DOMContentLoaded', function() {
//     const navButtons = document.querySelectorAll('.nav-btn');
//     const pages = document.querySelectorAll('.page');
    
//     // Navigation click handlers
//     navButtons.forEach(btn => {
//         btn.addEventListener('click', () => {
//             const targetPage = btn.getAttribute('data-page');
            
//             navButtons.forEach(b => b.classList.remove('active'));
//             btn.classList.add('active');
            
//             pages.forEach(page => page.classList.remove('active'));
//             document.getElementById(`${targetPage}-page`).classList.add('active');
//         });
//     });
    
//     // Load available chapters
//     loadChapters();
// });

// // Load chapters from backend
// async function loadChapters() {
//     try {
//         const response = await fetch(`${BASE_URL}/api/chapters`);
//         const data = await response.json();
        
//         const select = document.getElementById('chapter-select');
//         select.innerHTML = '<option value="">-- Choose a Chapter --</option>';
        
//         data.chapters.forEach(chapter => {
//             const option = document.createElement('option');
//             option.value = chapter.value;
//             option.textContent = `Chapter ${chapter.number}: ${chapter.title}`;
//             select.appendChild(option);
//         });
//     } catch (error) {
//         console.error('Error loading chapters:', error);
//     }
// }

// // Learning Page Functionality
// const questionInput = document.getElementById('question-input');
// const askBtn = document.getElementById('ask-btn');
// const answerSection = document.getElementById('answer-section');

// askBtn.addEventListener('click', handleAskQuestion);
// questionInput.addEventListener('keydown', (e) => {
//     if (e.ctrlKey && e.key === 'Enter') {
//         handleAskQuestion();
//     }
// });

// async function handleAskQuestion() {
//     const question = questionInput.value.trim();
    
//     if (!question) {
//         alert('Please enter a question');
//         return;
//     }
    
//     showLoading();
    
//     try {
//         const response = await fetch(`${BASE_URL}/api/learning/ask`, {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ question: question })
//         });
        
//         const data = await response.json();
//         hideLoading();
//         displayAnswer(data.answer || 'No answer received');
//     } catch (error) {
//         hideLoading();
//         console.error('Error:', error);
//         displayAnswer('Error connecting to the server. Please try again.');
//     }
// }

// function displayAnswer(answer) {
//     answerSection.innerHTML = `
//         <div class="answer-content">
//             <h3>Answer</h3>
//             <div class="answer-text">${formatAnswer(answer)}</div>
//         </div>
//     `;
// }

// function formatAnswer(text) {
//     const paragraphs = text.split('\n\n');
//     return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
// }

// // Quiz Generator Page Functionality
// const chapterSelect = document.getElementById('chapter-select');
// const generateBtn = document.getElementById('generate-btn');
// const quizSection = document.getElementById('quiz-section');

// chapterSelect.addEventListener('change', () => {
//     generateBtn.disabled = !chapterSelect.value;
// });

// generateBtn.addEventListener('click', handleGenerateQuiz);

// async function handleGenerateQuiz() {
//     const selectedChapter = chapterSelect.value;
    
//     if (!selectedChapter) {
//         alert('Please select a chapter');
//         return;
//     }
    
//     showLoading();
    
//     try {
//         const response = await fetch(`${BASE_URL}/api/quiz/generate`, {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ chapter: selectedChapter })
//         });
        
//         const data = await response.json();
//         hideLoading();
        
//         if (data.error) {
//             alert(data.error);
//             return;
//         }
        
//         currentQuizData = data;
//         currentQuestions = data.questions;
//         userAnswers = {};
//         displayQuiz();
//     } catch (error) {
//         hideLoading();
//         console.error('Error:', error);
//         alert('Error generating quiz. Please try again.');
//     }
// }

// function displayQuiz() {
//     if (!currentQuestions || currentQuestions.length === 0) {
//         quizSection.innerHTML = `
//             <div class="quiz-placeholder">
//                 <div class="placeholder-icon">⚠️</div>
//                 <p>No quiz questions available for this chapter</p>
//             </div>
//         `;
//         return;
//     }
    
//     const quizHTML = `
//         <div class="quiz-content active">
//             <div class="quiz-header">
//                 <h3>Quiz: Chapter ${currentQuizData.chapter_number}</h3>
//                 <p>Answer the following questions in your own words</p>
//             </div>
//             <div id="questions-container">
//                 ${currentQuestions.map((q, index) => `
//                     <div class="quiz-question" data-question-index="${index}">
//                         <div class="question-number">Question ${index + 1}</div>
//                         <div class="question-text">${q.question}</div>
//                         <textarea 
//                             class="answer-input" 
//                             id="answer-${index}"
//                             placeholder="Type your answer here..."
//                             rows="4"
//                         ></textarea>
//                         <button 
//                             class="submit-answer-btn" 
//                             onclick="submitAnswer(${index})"
//                         >
//                             Submit Answer
//                         </button>
//                         <div id="evaluation-${index}" class="evaluation-result hidden"></div>
//                     </div>
//                 `).join('')}
//             </div>
//         </div>
//     `;
    
//     quizSection.innerHTML = quizHTML;
// }

// async function submitAnswer(questionIndex) {
//     const answerTextarea = document.getElementById(`answer-${questionIndex}`);
//     const userAnswer = answerTextarea.value.trim();
    
//     if (!userAnswer) {
//         alert('Please provide an answer before submitting');
//         return;
//     }
    
//     const question = currentQuestions[questionIndex].question;
    
//     showLoading();
    
//     try {
//         const response = await fetch(`${BASE_URL}/api/quiz/evaluate`, {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({
//                 question: question,
//                 user_answer: userAnswer,
//                 chapter_context: currentQuizData.chapter_context
//             })
//         });
        
//         const evaluation = await response.json();
//         hideLoading();
        
//         if (evaluation.error) {
//             alert(evaluation.error);
//             return;
//         }
        
//         userAnswers[questionIndex] = {
//             answer: userAnswer,
//             evaluation: evaluation
//         };
        
//         displayEvaluation(questionIndex, evaluation);
        
//     } catch (error) {
//         hideLoading();
//         console.error('Error:', error);
//         alert('Error evaluating answer. Please try again.');
//     }
// }

// function displayEvaluation(questionIndex, evaluation) {
//     const evalContainer = document.getElementById(`evaluation-${questionIndex}`);
    
//     const scoreClass = evaluation.score >= 80 ? 'excellent' : 
//                        evaluation.score >= 60 ? 'good' : 
//                        evaluation.score >= 40 ? 'fair' : 'poor';
    
//     evalContainer.innerHTML = `
//         <div class="eval-header">
//             <div class="score-badge ${scoreClass}">
//                 Score: ${evaluation.score}%
//             </div>
//         </div>
//         <div class="eval-section">
//             <h4>📝 Feedback</h4>
//             <p>${evaluation.feedback}</p>
//         </div>
//         <div class="eval-section">
//             <h4>✅ Model Answer</h4>
//             <p>${evaluation.improved_answer}</p>
//         </div>
//     `;
    
//     evalContainer.classList.remove('hidden');
    
//     // Scroll to evaluation
//     evalContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
// }

// // Make submitAnswer globally accessible
// window.submitAnswer = submitAnswer;

// // Loading overlay functions
// function showLoading() {
//     document.getElementById('loading-overlay').classList.remove('hidden');
// }

// function hideLoading() {
//     document.getElementById('loading-overlay').classList.add('hidden');
// }




// Global state
let currentQuizData = null;
let currentQuestions = [];
let userAnswers = {};

function getBaseUrl() {
  const origin = window.location.origin;
  const hostname = window.location.hostname;
  
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://127.0.0.1:8000";
  }
  return origin;
}

const BASE_URL = getBaseUrl();

// Navigation
document.addEventListener('DOMContentLoaded', function() {
  const navButtons = document.querySelectorAll('.nav-btn');
  const pages = document.querySelectorAll('.page');
  
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetPage = btn.getAttribute('data-page');
      navButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      pages.forEach(page => page.classList.remove('active'));
      document.getElementById(`${targetPage}-page`).classList.add('active');
    });
  });
  
  loadChapters();
});

// Load Chapters
async function loadChapters() {
  try {
    const response = await fetch(`${BASE_URL}/api/chapters`);
    const data = await response.json();
    const select = document.getElementById('chapter-select');
    select.innerHTML = '<option value="">-- Choose a Chapter --</option>';
    data.chapters.forEach(chapter => {
      const option = document.createElement('option');
      option.value = chapter.value;
      option.textContent = `Chapter ${chapter.number}: ${chapter.title}`;
      select.appendChild(option);
    });
  } catch (error) {
    console.error('Error loading chapters:', error);
  }
}

// Learning Page
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const answerSection = document.getElementById('answer-section');

askBtn.addEventListener('click', handleAskQuestion);
questionInput.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') handleAskQuestion();
});

async function handleAskQuestion() {
  const question = questionInput.value.trim();
  if (!question) return alert('Please enter a question');

  showLoading();
  try {
    const res = await fetch(`${BASE_URL}/api/learning/ask`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    hideLoading();
    displayAnswer(data.answer || 'No answer received');
  } catch (err) {
    hideLoading();
    displayAnswer('Error connecting to server.');
  }
}

function displayAnswer(answer) {
  answerSection.innerHTML = `
    <div class="answer-content">
      <h3>Answer</h3>
      <div class="answer-text">${formatAnswer(answer)}</div>
    </div>
  `;
}

function formatAnswer(text) {
  const paragraphs = text.split('\n\n');
  return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
}

// Quiz Page
const chapterSelect = document.getElementById('chapter-select');
const generateBtn = document.getElementById('generate-btn');
const quizSection = document.getElementById('quiz-section');

chapterSelect.addEventListener('change', () => {
  generateBtn.disabled = !chapterSelect.value;
});

generateBtn.addEventListener('click', handleGenerateQuiz);

async function handleGenerateQuiz() {
  const selectedChapter = chapterSelect.value;
  if (!selectedChapter) return alert('Please select a chapter');

  showLoading();
  try {
    const res = await fetch(`${BASE_URL}/api/quiz/generate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ chapter: selectedChapter })
    });
    const data = await res.json();
    hideLoading();

    if (data.error) return alert(data.error);

    currentQuizData = data;
    currentQuestions = data.questions;
    userAnswers = {};
    displayQuiz();
  } catch (err) {
    hideLoading();
    alert('Error generating quiz.');
  }
}

function displayQuiz() {
  if (!currentQuestions.length) {
    quizSection.innerHTML = `
      <div class="quiz-placeholder">
        <div class="placeholder-icon">⚠️</div>
        <p>No questions available.</p>
      </div>
    `;
    return;
  }

  const quizHTML = `
    <div class="quiz-content active">
      <div class="quiz-header">
        <h3>Quiz: Chapter ${currentQuizData.chapter_number}</h3>
        <p>Answer in your own words.</p>
      </div>
      <div id="questions-container">
        ${currentQuestions.map((q, i) => `
          <div class="quiz-question" data-question-index="${i}">
            <div class="question-number">Question ${i + 1}</div>
            <div class="question-text">${q.question}</div>
            ${q.image_base64 ? `<img src="data:image/png;base64,${q.image_base64}" alt="Question Image" class="quiz-image">` : ""}
            <textarea id="answer-${i}" class="answer-input" placeholder="Type your answer here..." rows="4"></textarea>
            <button class="submit-answer-btn" onclick="submitAnswer(${i})">Submit Answer</button>
            <div id="evaluation-${i}" class="evaluation-result hidden"></div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
  quizSection.innerHTML = quizHTML;
}

async function submitAnswer(index) {
  const answerText = document.getElementById(`answer-${index}`).value.trim();
  if (!answerText) return alert('Please provide an answer.');

  const question = currentQuestions[index].question;
  const imgBase64 = currentQuestions[index].image_base64 || null;

  showLoading();
  try {
    const res = await fetch(`${BASE_URL}/api/quiz/evaluate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        user_answer: answerText,
        chapter_context: currentQuizData.chapter_context,
        image_base64: imgBase64
      })
    });

    const data = await res.json();
    hideLoading();

    userAnswers[index] = { answer: answerText, evaluation: data };
    displayEvaluation(index, data);
  } catch (err) {
    hideLoading();
    alert('Error evaluating answer.');
  }
}

function displayEvaluation(index, evaluation) {
  const evalContainer = document.getElementById(`evaluation-${index}`);
  const scoreClass = evaluation.score >= 80 ? 'excellent' :
                     evaluation.score >= 60 ? 'good' :
                     evaluation.score >= 40 ? 'fair' : 'poor';

  evalContainer.innerHTML = `
    <div class="eval-header">
      <div class="score-badge ${scoreClass}">Score: ${evaluation.score}%</div>
    </div>
    <div class="eval-section">
      <h4>📝 Feedback</h4>
      <p>${evaluation.feedback}</p>
    </div>
    <div class="eval-section">
      <h4>✅ Model Answer</h4>
      <p>${evaluation.improved_answer}</p>
      ${evaluation.image_base64 ? `<img src="data:image/png;base64,${evaluation.image_base64}" alt="Model Image" class="quiz-image">` : ""}
    </div>
  `;
  evalContainer.classList.remove('hidden');
  evalContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

window.submitAnswer = submitAnswer;

function showLoading() {
  document.getElementById('loading-overlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.add('hidden');
}
