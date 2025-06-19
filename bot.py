import os
import re
import random
import time
import requests
import speech_recognition as sr
from gtts import gTTS
import pygame
import queue
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import cv2
import numpy as np
import pygetwindow as gw
import threading
import wave
from scipy.io import wavfile
import subprocess
import tempfile
import sys
import boto3
from sentence_transformers import SentenceTransformer
import faiss  # For vector similarity search
import json
import os

# Load environment variables
load_dotenv()

class ExpertTechnicalInterviewer:
    def __init__(self, model="gemini-2.0-flash", accent="indian"):
        try:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Please set the GEMINI_API_KEY in .env file")

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model)
            self.interview_state = "introduction"
            self.skill_questions_asked = 0
            self.last_question = None
            self.just_repeated = False
            self.current_domain = None
            self.conversation_history = []
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.is_listening = False
            self.interrupted = False
            self.recognizer.pause_threshold = 0.6
            self.recognizer.phrase_threshold = 0.2
            self.tone_warnings = 0
            self.cheating_warnings = 0
            self.filler_phrases = [
            "I see...", "Interesting...", "That makes sense...", 
            "Go on...", "Yes, I understand...", "Right...",
            "Okay...", "Hmm...", "Got it...", "Please continue..."]
            self.tab_monitor_ready = False
            self.last_face_detection_time = time.time()
            self.tab_change_detected = False
            self.response_delay = 0.3
            self.accent = accent.lower()
            self.interview_active = True
            self.coding_questions_asked = 0
            self.max_coding_questions = 2
            self.polly = boto3.client(
                "polly",
                region_name=os.getenv("AWS_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Initialize face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Initialize camera
            self.cap = None
            self.camera_active = False
            self.current_coding_question = None
            
            self.tech_domains = {
                "frontend": ["React", "Angular", "Vue", "JavaScript", "TypeScript", "CSS", "HTML5"],
                "backend": ["Node.js", "Django", "Spring", "Go", "Rust", "Microservices", "APIs"],
                "AI": ["TensorFlow", "PyTorch", "NLP", "Computer Vision", "LLMs", "Generative AI"],
                "data science": [ "data science","Pandas", "NumPy", "SQL", "Data Visualization", "ETL", "Big Data"],
                "machine learning": ["machine learning","Scikit-learn", "Keras", "Model Deployment", "Feature Engineering"],
                "devops": ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Monitoring"],
                "mobile": ["Flutter", "React Native", "Swift", "Kotlin", "Mobile UX"],
                "python": ["Python", "Flask", "FastAPI", "Django", "Data Structures", "Algorithms"],
                "java": ["Java", "Spring Boot", "JVM", "Object Oriented Programming", "Collections"],
                "cpp": ["C++", "STL", "Memory Management", "Object Oriented Programming", "Data Structures"]
            }
        
            self.non_tech_domains = {
                "edtech": ["Curriculum Design", "Learning Management Systems", "Instructional Design", 
                          "Educational Technology", "Student Engagement", "Assessment Tools"],
                "fintech": ["Digital Payments", "Blockchain", "Risk Management", "Financial Modeling", 
                           "Regulatory Compliance", "Banking Systems"],
                "healthcare": ["Healthcare IT", "Electronic Health Records", "Medical Billing", 
                              "Healthcare Analytics", "Telemedicine", "HIPAA Compliance"],
                "banking": ["Retail Banking", "Investment Banking", "Wealth Management", 
                           "Loan Processing", "Anti-Money Laundering", "Financial Analysis"],
                "insurance": ["Underwriting", "Claims Processing", "Actuarial Science", 
                             "Risk Assessment", "Policy Administration", "Customer Service"]
            }
            
            # Initialize pygame mixer with error handling
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            except pygame.error as e:
                print(f"PyGame mixer initialization failed: {e}")
                raise RuntimeError("Audio system initialization failed")
                
            # Start monitoring threads
            self.monitoring_active = True
            self.last_question = None
            self.face_monitor_thread = threading.Thread(target=self._monitor_face_and_attention)
            self.face_monitor_thread.daemon = True
            self.face_monitor_thread.start()
            
            self.tab_monitor_thread = threading.Thread(target=self._monitor_tab_changes)
            self.tab_monitor_thread.daemon = True
            self.tab_monitor_thread.start()

        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def wait_after_speaking(self, message, base=0.6, per_word=0.15):
        if not message:
            time.sleep(base + 0.5)
            return
        words = message.split()
        delay = base + per_word * len(words)
        print(f"[Pause] Waiting {round(delay, 2)}s after speaking.")
        time.sleep(delay)

    def _give_small_hint(self, question_text):
        hint_prompt = f"""You are an AI coding interviewer. Give a small hint for the following problem.
        It should not reveal the full solution, just nudge the candidate in the right direction.

        Problem:
        {question_text}

        Format: Hint: [short helpful nudge]"""

        hint = self.query_gemini(hint_prompt)
        if hint:
            self.speak(hint.strip(), interruptible=False)
    
    def _get_file_extension(self, language):
        return {
            "Python": ".py",
            "Java": ".java",
            "C++": ".cpp",
            "JavaScript": ".js"
        }.get(language, ".txt")

    def _generate_non_tech_question(self, domain):
        """Generate non-technical questions for professional interviews"""
        prompt = f"""Generate one professional interview question about {domain} that:
        - Tests domain knowledge
        - Is relevant to real work situations
        - Is clear and concise
        - Is appropriate for an interview setting
        
        Generate only the question, no additional text."""
        
        try:
            response = self.query_gemini(prompt)
            return response.strip() if response else None
        except Exception as e:
            print(f"Error generating non-tech question: {e}")
            return f"Can you describe your experience working in {domain}?"
        

    def _get_filler_phrase(self):
        """Return appropriate filler phrases to show active listening"""
        fillers = [
            "I see...",
            "Interesting...",
            "That makes sense...",
            "Go on...",
            "Yes, I understand...",
            "Right...",
            "Okay...",
            "Hmm...",
            "Got it...",
            "Please continue..."
        ]
        return random.choice(fillers)

    def _execute_code(self, language, file_path):
        try:
            if language == "Python":
                result = subprocess.run(["python", file_path], capture_output=True, text=True, timeout=10)
            elif language == "Java":
                compile_result = subprocess.run(["javac", file_path], capture_output=True, text=True)
                if compile_result.returncode != 0:
                    return f"Compile Error:\n{compile_result.stderr}"
                class_name = os.path.basename(file_path).replace(".java", "")
                result = subprocess.run(["java", class_name], capture_output=True, text=True, timeout=10, cwd=os.path.dirname(file_path))
            elif language == "C++":
                exe_path = file_path.replace(".cpp", ".exe")
                compile_result = subprocess.run(["g++", file_path, "-o", exe_path], capture_output=True, text=True)
                if compile_result.returncode != 0:
                    return f"Compile Error:\n{compile_result.stderr}"
                result = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            elif language == "JavaScript":
                result = subprocess.run(["node", file_path], capture_output=True, text=True, timeout=10)
            else:
                return "Unsupported language."

            output = ""
            if result.stdout:
                output += f"Output:\n{result.stdout}\n"
            if result.stderr:
                output += f"Errors:\n{result.stderr}\n"
            return output if output else "Code executed successfully (no output)."

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (10 seconds limit)"
        except Exception as e:
            return f"Runtime error: {str(e)}"

    def _is_repeat_request(self, text):
        if not text:
            return False
        repeat_phrases = [
            "repeat", "say again", "pardon", "once more",
            "come again", "didn't catch", "hear that"
        ]
        return any(phrase in text.lower() for phrase in repeat_phrases)

    def _run_interview_logic(self):
        try:
            # Friendly introduction
            self.speak("Hello! I am Gyani. Welcome to your interview session today. I'm excited to chat with you!", interruptible=False)
            time.sleep(6)
            msg = "Before we begin, how has your day been so far?"
            self.speak(msg, interruptible=False)
            self.wait_after_speaking(msg)
            day_response = self.listen()

            if day_response:
                self.conversation_history.append({"role": "user", "content": day_response})
                self.speak("That's great to hear! I appreciate you taking the time for this session.", interruptible=False)

            msg = "Now, could you please tell me your name and a bit about yourself?"
            self.speak(msg, interruptible=False)
            self.wait_after_speaking(msg)
            introduction = self.listen()

            if introduction:
                self.conversation_history.append({"role": "user", "content": introduction})
                # Determine if this is a tech or non-tech interview based on introduction
                self.current_domain = self._identify_tech_domain(introduction)
                is_tech_interview = self.current_domain in self.tech_domains

                if is_tech_interview:
                    msg = "Nice to meet you! Now, I'd love to hear about your technical background and the technologies you enjoy working with."
                else:
                    msg = "Nice to meet you! Could you tell me about your professional experience and the domains you've worked in?"
                
                self.speak(msg, interruptible=False)
                self.wait_after_speaking(msg)
                background = self.listen()

                if background:
                    self.conversation_history.append({"role": "user", "content": background})
                    self.current_domain = self._identify_tech_domain(background)
                    is_tech_interview = self.current_domain in self.tech_domains

            # Questions Phase - Different for tech vs non-tech
            question_count = 0
            max_questions = 6
            
            if is_tech_interview:
                self.speak("Let's start with some technical questions to understand your experience better.", interruptible=False)
            else:
                self.speak("Let's discuss your professional experience in more detail.", interruptible=False)

            while question_count < max_questions and self.interview_active:
                if len(self.conversation_history) > 15:
                    self.conversation_history = self.conversation_history[-8:]

                if is_tech_interview:
                    system_prompt = f"""As a friendly technical interviewer, ask one engaging question about {self.current_domain or 'technology'} 
                    based on this conversation context. The question should:
                    - Be encouraging and conversational
                    - Build on what the candidate has already shared
                    - Test practical knowledge and experience
                    - Be appropriate for their stated experience level
                    - Keep it to one clear question
                    - Focus on real-world application
                    - Do not repeat same question again
                    - Question should be one-liner 
                    
                    Recent conversation: {' '.join(msg['content'] for msg in self.conversation_history[-3:])}
                    
                    Generate only the question in a friendly, conversational tone."""
                else:
                    system_prompt = f"""As a friendly professional interviewer, ask one engaging question about {self.current_domain or 'professional work'} 
                    based on this conversation context. The question should:
                    - Be encouraging and conversational
                    - Focus on real-world professional scenarios
                    - Test domain knowledge and problem-solving
                    - Be appropriate for their stated experience level
                    - Keep it to one clear question
                    - Focus on practical situations
                    - Do not repeat same question again
                    - Question should be one-liner 
                    
                    Recent conversation: {' '.join(msg['content'] for msg in self.conversation_history[-3:])}
                    
                    Generate only the question in a friendly, conversational tone."""

                response = self.query_gemini(system_prompt)
                
                if response:
                    msg = response.strip()
                    
                    if msg == self.last_question and not self.just_repeated:
                        print("[Duplicate] Skipping repeated question.")
                        continue

                    self.last_question = msg
                    answer_received = False
                    repeat_attempts = 0
                    max_repeats = 2

                    while not answer_received and repeat_attempts < max_repeats:
                        if not self.just_repeated:
                            self.speak(msg)
                            self.wait_after_speaking(msg)
                        
                        answer = self.listen()
                        
                        if answer and self._is_repeat_request(answer):
                            if repeat_attempts < max_repeats:
                                self.just_repeated = True
                                repeat_attempts += 1
                                # Rephrase the question instead of repeating verbatim
                                rephrased = self._rephrase_question(msg)
                                self.speak("Let me rephrase that: " + rephrased)
                                self.last_question = rephrased
                                self.wait_after_speaking(rephrased)
                                continue
                            else:
                                placeholder = "[Requested repeat too many times]"
                                self.conversation_history.append({"role": "user", "content": placeholder})
                                answer_received = True

                        # Handle when candidate can't answer after multiple attempts
                        elif not answer or len(answer.split()) <= 3:
                            if repeat_attempts < max_repeats - 1:
                                self.speak("Could you please elaborate on that?", interruptible=False)
                            else:
                                # Provide the answer after multiple failed attempts
                                answer_prompt = f"""The candidate couldn't answer this question after multiple attempts:
                                Question: {msg}
                                
                                Please provide a concise, helpful answer (2-3 sentences) that:
                                - Explains the key concept
                                - Gives a simple example if applicable
                                - Is encouraging
                                
                                Keep it professional and educational."""
                                
                                answer_response = self.query_gemini(answer_prompt)
                                if answer_response:
                                    self.speak("Let me help with that. " + answer_response, interruptible=False)
                                
                                placeholder = "[Unable to answer after multiple attempts]"
                                self.conversation_history.append({"role": "user", "content": placeholder})
                                answer_received = True
                        
                        # Process valid answer
                        elif answer and len(answer.split()) > 4:
                            self.conversation_history.append({"role": "user", "content": answer})
                            answer_received = True
                            break
                            
                        # Handle invalid answers
                        else:
                            if repeat_attempts < max_repeats - 1:
                                self.speak("Could you please elaborate on that?", interruptible=False)
                            else:
                                placeholder = "[Unclear response after multiple attempts]"
                                self.conversation_history.append({"role": "user", "content": placeholder})
                                answer_received = True

                    # Only count question if we got a valid answer
                    if answer_received:
                        question_count += 1
                        self.conversation_history.append({"role": "assistant", "content": msg})
                        self.just_repeated = False

            # Coding Questions (only for tech interviews)
            if is_tech_interview and self.interview_active and self.coding_questions_asked < self.max_coding_questions:
                self.speak("Great discussion! Now I'd like to give you a couple of coding challenges to see your problem-solving skills in action.", interruptible=False)
                time.sleep(1)

                while self.coding_questions_asked < self.max_coding_questions and self.interview_active:
                    self.current_coding_question = self._generate_coding_question(self.current_domain or "python")

                    self.speak("I've prepared a coding challenge for you. Here's the problem:", interruptible=False)
                    self.speak(self.current_coding_question, interruptible=False)
                    self.speak("Please describe your approach to solving this problem.", interruptible=False)
                    
                    hint_offered = False
                    start_time = time.time()

                    while self.coding_questions_asked < self.max_coding_questions and self.interview_active:
                        time.sleep(1)
                        
                        # Offer a hint after 2 minutes of inactivity
                        if not hint_offered and time.time() - start_time > 120:
                            self.speak("Would you like a small hint to help you get started?", interruptible=False)
                            self.wait_after_speaking("Would you like a small hint to help you get started?")
                            response = self.listen()
                            if response and "yes" in response.lower():
                                self._give_small_hint(self.current_coding_question)
                            hint_offered = True

                    if not self.interview_active:
                        break
                    time.sleep(6)

            # Closing
            if self.interview_active:
                if is_tech_interview:
                    self.speak("That was excellent! You've shown great technical knowledge and problem-solving skills.", interruptible=False)
                    time.sleep(1)

                    # Doubt-clearing session
                    self.speak("Before we conclude, I'd like to offer you a chance to ask any technical questions you might have.", interruptible=False)
                    self.speak("This could be about:", interruptible=False)
                    self.speak("1. The coding problems we discussed", interruptible=False)
                    self.speak("2. Any of the technical concepts we covered", interruptible=False)
                    self.speak("3. Best practices in the field", interruptible=False)
                    self.speak("4. Or anything else technical you'd like to discuss", interruptible=False)
                else:
                    self.speak("That was excellent! You've shown great professional knowledge and problem-solving skills.", interruptible=False)
                    time.sleep(1)

                    # Doubt-clearing session
                    self.speak("Before we conclude, I'd like to offer you a chance to ask any questions you might have about the role or industry.", interruptible=False)
                    self.speak("This could be about:", interruptible=False)
                    self.speak("1. The professional scenarios we discussed", interruptible=False)
                    self.speak("2. Any of the domain concepts we covered", interruptible=False)
                    self.speak("3. Industry best practices", interruptible=False)
                    self.speak("4. Or anything else you'd like to discuss", interruptible=False)
                
                self.speak("What would you like to ask?", interruptible=False)
                
                # Allow up to 3 questions with follow-up
                max_questions = 3
                questions_asked = 0
                
                while questions_asked < max_questions and self.interview_active:
                    self.wait_after_speaking("Do you have any questions?")
                    question = self.listen()
                    
                    if question and len(question.split()) > 3:
                        questions_asked += 1
                        
                        # Get answer from AI
                        answer_prompt = f"""Provide a concise but helpful answer to this {'technical' if is_tech_interview else 'professional'} question:
                        Question: {question}
                        
                        Requirements:
                        - Keep answer under 4 sentences
                        - Be {'technically' if is_tech_interview else 'professionally'} accurate
                        - Include one practical example if relevant
                        - End by asking if they'd like clarification
                        """
                        
                        answer = self.query_gemini(answer_prompt)
                        if answer:
                            self.speak(answer, interruptible=False)
                            self.wait_after_speaking(answer)
                            
                            # Check if they need follow-up
                            self.speak("Does that answer your question, or would you like me to elaborate?", interruptible=False)
                            followup = self.listen()
                            
                            if followup and "elaborate" in followup.lower():
                                elaboration_prompt = f"""Provide more detailed explanation about:
                                {question}
                                
                                Context:
                                {answer}
                                
                                Requirements:
                                - Go deeper {'technically' if is_tech_interview else 'professionally'}
                                - Include examples
                                - Keep to 5-6 sentences max"""
                                
                                elaboration = self.query_gemini(elaboration_prompt)
                                if elaboration:
                                    self.speak(elaboration, interruptible=False)
                                    self.wait_after_speaking(elaboration)
                        
                        if questions_asked < max_questions:
                            self.speak("Do you have any other questions?", interruptible=False)
                
                self.speak("Thank you so much for your time today. It was a pleasure talking with you, and I wish you the best of luck!", interruptible=False)

        except Exception as e:
            print(f"Interview error: {e}")
            self.speak("We've encountered a technical issue, but thank you for your participation today!", interruptible=False)
        finally:
            self.interview_active = False
            self.monitoring_active = False
            self._stop_camera()

    def _start_camera(self):
        """Start the camera for face detection"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            self.camera_active = True

    def _stop_camera(self):
        """Stop the camera"""
        if self.camera_active and self.cap:
            self.cap.release()
            self.camera_active = False

    def _monitor_face_and_attention(self):    
        while self.monitoring_active and self.interview_active:
            try:
                if not self.camera_active or not self.cap:
                    time.sleep(2)
                    continue
                    
                with threading.Lock():  # Add thread safety
                    ret, frame = self.cap.read()
                    if not ret:
                        self._restart_camera()
                        continue
                    
                # Convert to grayscale and apply histogram equalization
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # More accurate face detection parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # Reduced from 1.1
                    minNeighbors=7,    # Increased from 5
                    minSize=(150, 150),# Increased minimum face size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Only trigger warning if we're very confident
                if len(faces) > 1:
                    # Additional verification - check face sizes are similar
                    areas = [w*h for (x,y,w,h) in faces]
                    if max(areas)/min(areas) < 4:  # Only if faces are similarly sized
                        if not multiple_faces_warning_given:
                            self._handle_cheating_attempt("multiple_faces")
                            multiple_faces_warning_given = True
                    else:
                        multiple_faces_warning_given = False
                else:
                    multiple_faces_warning_given = False
                        
                # Eye detection and attention check
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(30, 30)
                    )
                    
                    # Only check attention if we have good eye detection
                    if len(eyes) >= 2:  # At least two eyes detected
                        eye_centers = [(ex + ew/2, ey + eh/2) for (ex, ey, ew, eh) in eyes]
                        avg_eye_y = sum(ey for (ex, ey) in eye_centers) / len(eye_centers)
                        
                        # More lenient threshold for looking away
                        if avg_eye_y > h * 0.75 and not looking_away_warning_given:  # Eyes looking down
                            self._handle_cheating_attempt("looking_away")
                            looking_away_warning_given = True
                        elif avg_eye_y <= h * 0.75:
                            looking_away_warning_given = False
                
            except Exception as e:
                print(f"Camera error: {e}")
                self._restart_camera()
                time.sleep(1)

    def _restart_camera(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera_active = True
        except Exception as e:
            print(f"Camera restart failed: {e}")
                
            time.sleep(0.5)  # Reduce CPU usage

    def _monitor_tab_changes(self):
        while not self.tab_monitor_ready:
            time.sleep(0.5)
        
        try:
            initial_window = gw.getActiveWindow()
            initial_title = initial_window.title if initial_window else "Interview Window"
        except:
            initial_window = None
            initial_title = "Interview Window"
        
        warning_given = False
        
        while self.monitoring_active and self.interview_active:
            try:
                current_window = gw.getActiveWindow()
                current_title = current_window.title if current_window else initial_title
                
                # Only trigger if:
                # 1. We have a valid window
                # 2. The title has actually changed significantly
                # 3. The new title doesn't contain system/keywords
                if (current_window and initial_window and 
                    current_title != initial_title and
                    not any(x in current_title.lower() for x in ["notification", "system", "settings"])):
                    
                    if not warning_given:  # Only warn once per change
                        self.tab_change_detected = True
                        self._handle_cheating_attempt("tab_change")
                        warning_given = True
                else:
                    warning_given = False
                    
                time.sleep(3)  # Longer delay between checks
                
            except Exception as e:
                print(f"Window monitoring error: {e}")
                time.sleep(3)

    def _handle_cheating_attempt(self, cheat_type):
        """Handle different types of cheating attempts"""
        self.cheating_warnings += 1
        
        if self.cheating_warnings >= 3:
            self.speak("Multiple concerning behaviors detected. The interview will now conclude.", interruptible=False)
            self.interview_active = False
            return
            
        responses = {
            "no_face": "Please ensure your face is clearly visible to the camera for the interview.",
            "multiple_faces": "I notice multiple people in the frame. Please ensure you're alone during this interview.",
            "looking_away": "Please maintain focus on the interview and avoid looking at other devices.",
            "tab_change": "Please stay focused on the interview window and avoid switching to other applications."
        }
        
        if cheat_type in responses:
            self.speak(f"Gentle reminder: {responses[cheat_type]} This is notice {self.cheating_warnings} of 3.", interruptible=False)

    def __del__(self):
        """Clean up resources"""
        self.interview_active = False
        self.monitoring_active = False
        self._stop_camera()
        if hasattr(self, 'face_monitor_thread'):
            self.face_monitor_thread.join(timeout=1)
        if hasattr(self, 'tab_monitor_thread'):
            self.tab_monitor_thread.join(timeout=1)
        pygame.mixer.quit()

    def speak(self, text, interruptible=True):
        if self.interrupted:
            self.interrupted = False
            return

        print(f"Interviewer: {text}")

        try:
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId="Aditi"
            )

            if "AudioStream" in response:
                temp_path = os.path.join(tempfile.gettempdir(), f"polly_{int(time.time() * 1000)}.mp3")
                with open(temp_path, 'wb') as f:
                    f.write(response["AudioStream"].read())

                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                # Safely attempt file removal
                try:
                    time.sleep(0.2)  # tiny delay before delete
                    os.remove(temp_path)
                except PermissionError:
                    pass  # Suppress WinError 32

        except Exception as e:
            print(f"AWS Polly TTS error: {e}")

    def listen(self, max_attempts=3):
        """Listen for user response with proper context management"""
        for attempt in range(max_attempts):
            try:
                # Create new recognizer instance for this attempt
                attempt_recognizer = sr.Recognizer()
                with self.microphone as source:
                    print("\nListening... (Speak now)")
                    
                    # Adjust for ambient noise with clean context
                    attempt_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    try:
                        audio = attempt_recognizer.listen(
                            source, 
                            timeout=15, 
                            phrase_time_limit=60
                        )
                        
                        text = attempt_recognizer.recognize_google(audio)
                        print(f"Candidate: {text}")
                        
                        # Add filler phrase to show active listening
                        if len(text.split()) > 5:  # Only if substantial response
                            filler = self._get_filler_phrase()
                            self.speak(filler, interruptible=False)
                        
                        # Process tone detection
                        tone = self._detect_tone(text)
                        if tone != "professional":
                            self.handle_improper_tone(tone)
                            placeholder = "[Response had non-professional tone]"
                            self.conversation_history.append({"role": "user", "content": placeholder})
                            return placeholder
                        
                        if text.strip():
                            repeat_phrases = [
                                "can you repeat", "please repeat", 
                                "repeat the question", "say again",
                                "pardon", "once more", "come again",
                                "could you repeat"
                            ]
                            lower_text = text.lower()
                            
                            if any(phrase in lower_text for phrase in repeat_phrases) and self.last_question:
                                # Special handling for repeat requests
                                self.speak(f"Sure, here's the question again: {self.last_question}", interruptible=False)
                                self.wait_after_speaking(self.last_question)
                                # Return special marker instead of recursive listen()
                                return "[REPEAT_REQUEST]"
                            else:
                                self.conversation_history.append({"role": "user", "content": text})
                                return text
                        else:
                            placeholder = "[Unclear response]"
                            self.conversation_history.append({"role": "user", "content": placeholder})
                            return placeholder
                            
                    except sr.WaitTimeoutError:
                        if attempt < max_attempts - 1:
                            self.speak("I didn't hear anything. Please speak when you're ready.", interruptible=False)
                            time.sleep(2)
                        continue
                        
                    except sr.UnknownValueError:
                        if attempt < max_attempts - 1:
                            self.speak("I couldn't quite catch that. Could you please repeat?", interruptible=False)
                            time.sleep(2)
                        continue
                        
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
                        if attempt < max_attempts - 1:
                            self.speak("There was a technical issue. Please try speaking again.", interruptible=False)
                            time.sleep(2)
                        continue
                        
            except OSError as e:
                print(f"Microphone access error: {e}")
                self.speak("I'm having trouble accessing the microphone. Please check your microphone settings.", interruptible=False)
                placeholder = "[Microphone issue]"
                self.conversation_history.append({"role": "user", "content": placeholder})
                return placeholder
        
        # If all attempts fail
        placeholder = "[Response unclear after multiple attempts]"
        self.conversation_history.append({"role": "user", "content": placeholder})
        self.speak("Let's continue with the next part of our interview.", interruptible=False)
        return placeholder
    def _rephrase_question(self, question):
        """Rephrase the given question while keeping the same meaning"""
        prompt = f"""Rephrase this interview question to make it clearer while keeping the same meaning:
        Original: {question}
        
        Requirements:
        - Keep technical accuracy
        - Maintain same difficulty level
        - Don't change the core concept being tested
        - Make it slightly different wording
        - Keep it one sentence
        
        Return only the rephrased question."""
        
        rephrased = self.query_gemini(prompt)
        return rephrased.strip() if rephrased else question

    def _detect_tone(self, text):
        if not text:
            return "professional"
            
        text_lower = re.sub(r'\s+', ' ', text.lower().strip())
        
        arrogant_keywords = [
            r'\bobviously\b', r'\beveryone knows\b', r'\bchild\'?s play\b',
            r'\bthat\'?s easy\b', r'\btrivial\b', r'\bwaste of time\b'
        ]
        
        rude_patterns = [
            r'\byou don\'?t understand\b', r'\bthat\'?s stupid\b', r'\bdumb question\b',
            r'\bare you serious\b', r'\bthis is ridiculous\b', r'\bwho cares\b'
        ]
        
        for pattern in arrogant_keywords:
            if re.search(pattern, text_lower):
                return "arrogant"
                
        for pattern in rude_patterns:
            if re.search(pattern, text_lower):
                return "rude"
                
        return "professional"

    def handle_improper_tone(self, tone):
        self.tone_warnings += 1
        
        if self.tone_warnings >= 2:
            self.speak("I appreciate your participation, but let's maintain a professional tone throughout our conversation.", interruptible=False)
            return
            
        responses = {
            "arrogant": [
                "I appreciate your confidence! Let's channel that into demonstrating your technical knowledge.",
                "Great confidence! Now let's see how you apply that expertise to solve problems.",
            ],
            "rude": [
                "I understand technical interviews can be stressful. Let's take a moment and continue professionally.",
                "No worries, let's refocus on showcasing your technical abilities.",
            ]
        }
        
        if tone in responses:
            response = random.choice(responses[tone])
            self.speak(response, interruptible=False)
            time.sleep(1)

    def query_gemini(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'result'):
                return response.result
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return "Could you tell me more about your experience with that?"
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "Could you elaborate on your experience with that technology?"

    def _identify_tech_domain(self, text):
        if not text:
            return None
            
        # Check tech domains first
        domain_scores = {domain: 0 for domain in self.tech_domains}
        text_lower = text.lower()
        
        for domain, skills in self.tech_domains.items():
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower in text_lower:
                    domain_scores[domain] += 1
                elif re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
                    domain_scores[domain] += 2
                    
        best_tech_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        # If no strong tech domain found, check non-tech domains
        if best_tech_domain[1] < 2:
            domain_scores = {domain: 0 for domain in self.non_tech_domains}
            
            for domain, skills in self.non_tech_domains.items():
                for skill in skills:
                    skill_lower = skill.lower()
                    if skill_lower in text_lower:
                        domain_scores[domain] += 1
                    elif re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
                        domain_scores[domain] += 2
            
            best_non_tech_domain = max(domain_scores.items(), key=lambda x: x[1])
            return best_non_tech_domain[0] if best_non_tech_domain[1] > 0 else None
        else:
            return best_tech_domain[0]

    def _generate_coding_question(self, domain, difficulty="medium"):
        """Generate a coding question based on the candidate's domain"""
        domain_mapping = {
            "python": "Python",
            "java": "Java", 
            "cpp": "C++",
            "frontend": "JavaScript",
            "backend": "Python or your preferred language",
            "AI": "Python",
            "data science": "Python",
            "machine learning": "Python"
        }
        
        language = domain_mapping.get(domain, "Python")
        
        prompt = f"""Generate a {difficulty} level coding problem suitable for a technical interview in {domain}.
        
        Requirements:
        - Should be solvable in {language}
        - Should take 10-15 minutes to solve
        - Include a clear problem statement
        - Provide input/output examples
        - Should test algorithmic thinking and {domain} knowledge
        - Avoid problems that are too easy or too hard
        - Focus on practical problem-solving skills
        
        Format your response as:
        Problem: [Clear problem statement]
        
        Example Input: [Sample input]
        Example Output: [Expected output]
        
        Constraints: [Any constraints or edge cases to consider]
        
        Generate only the problem, no solution."""
        
        try:
            response = self.query_gemini(prompt)
            return response.strip() if response else None
        except Exception as e:
            print(f"Error generating coding question: {e}")
            return self._get_fallback_coding_question(domain)

    def _get_fallback_coding_question(self, domain):
        """Fallback coding questions if AI generation fails"""
        fallback_questions = {
            "python": """Problem: Find the two numbers in a list that add up to a target sum.

Example Input: numbers = [2, 7, 11, 15], target = 9
Example Output: [0, 1] (indices of numbers 2 and 7)

Constraints: Each input has exactly one solution, and you may not use the same element twice.""",
            
            "default": """Problem: Write a function to reverse words in a sentence while keeping the word order.

Example Input: "Hello World Python"
Example Output: "olleH dlroW nohtyP"

Constraints: Preserve spaces between words, handle empty strings gracefully."""
        }
        
        return fallback_questions.get(domain, fallback_questions["default"])

    def _generate_domain_followup(self, context, domain):
        """Generate a context-aware follow-up question for the domain"""
        if not domain or not context:
            return None
            
        prompt = f"""As a friendly technical interviewer specializing in {domain}, generate one concise follow-up question 
                based on this conversation context. The question should:
                - Be technically relevant to {domain}
                - Reference specific technologies mentioned if possible
                - Be conversational and encouraging
                - Be no longer than one sentence
                - Ask one question at a time
                - Focus on practical experience and understanding
                - Avoid overly complex theoretical questions
                - Be appropriate for the candidate's stated experience level

                Conversation context:
                {context}

                Generate only the question, no additional text."""

        
        try:
            response = self.query_gemini(prompt)
            return response.strip() if response else None
        except Exception as e:
            print(f"Error generating followup: {e}")
            return None
    def _coding_followup(self, code, language):
        """Ask follow-up questions about the code submitted by the candidate."""
        prompt = f"""You are an expert software engineer reviewing code written in {language}.
        The candidate has provided the following code:
        
        ```\n{code}\n```
        
        Ask one follow-up question that:
        - Tests their understanding of the code they wrote
        - Explores potential edge cases or improvements
        - Is specific to the code provided
        - Is clear and concise
        - Can be answered without running the code
        - Focuses on code clarity, efficiency, or potential bugs
        - Is appropriate for an interview setting
        
        Generate only the question, no additional text."""
        
        try:
            response = self.query_gemini(prompt)
            return response.strip() if response else None
        except Exception as e:
            print(f"Error generating coding follow-up: {e}")
            return "Can you walk me through your code and explain your approach?"


    def _add_domain_specific_followup(self, domain):
        """Generate only the follow-up question based on recent conversation and domain"""
        if not domain:
            return None

        # Get recent conversation context (last 2 exchanges)
        context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.conversation_history[-2:]
        )

        prompt = f"""As a friendly technical interviewer specializing in {domain}, generate one concise follow-up question 
        based on this conversation context. The question should:
        - Be technically relevant to {domain}
        - Reference specific technologies mentioned if possible
        - Be conversational and encouraging
        - Be no longer than one sentence
        - Ask one question at a time
        - Focus on practical experience and understanding
        - Avoid overly complex theoretical questions
        - Be appropriate for the candidate's stated experience level

        Conversation context:
        {context}

        Generate only the question, no additional text."""

        try:
            response = self.query_gemini(prompt)
            return response.strip() if response else None
        except Exception as e:
            print(f"Error generating follow-up: {e}")
            return None

    def start_interview(self):
        # Start tab monitoring
        def enable_tab_monitor():
            time.sleep(3)
            self.tab_monitor_ready = True

        threading.Thread(target=enable_tab_monitor, daemon=True).start()

        # Start the interview logic
        interview_thread = threading.Thread(target=self._run_interview_logic)
        interview_thread.daemon = True
        interview_thread.start()

        # Keep the main thread alive while interview is active
        while self.interview_active:
            time.sleep(1)
class RAGExpertTechnicalInterviewer(ExpertTechnicalInterviewer):
    def __init__(self, model="gemini-2.0-flash", accent="indian"):
        super().__init__(model, accent)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
        self.knowledge_base_path = "knowledge_base.json"
        self.vector_index_path = "vector_index.faiss"
        self.vector_dimension = 384  # MiniLM-L6 outputs 384-dimensional vectors
        self.vector_index = self._load_or_create_vector_index()
        self.knowledge_base = self._load_knowledge_base()

    def _load_or_create_vector_index(self):
        """Load an existing FAISS index or create a new one."""
        if os.path.exists(self.vector_index_path):
            return faiss.read_index(self.vector_index_path)
        else:
            return faiss.IndexFlatL2(self.vector_dimension)

    def _load_knowledge_base(self):
        if os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, "r") as f:
                return json.load(f)
        return []

    def _save_knowledge_base(self):
        with open(self.knowledge_base_path, "w") as f:
            json.dump(self.knowledge_base, f, indent=4)

    def _save_vector_index(self):
        faiss.write_index(self.vector_index, self.vector_index_path)

    def _add_to_knowledge_base(self, text):
        """Add new text to the knowledge base and update the vector index."""
        embedding = self.embedding_model.encode(text)
        self.knowledge_base.append({"text": text, "embedding": embedding.tolist()})
        self.vector_index.add(np.array([embedding]))
        self._save_knowledge_base()
        self._save_vector_index()

    def _retrieve_context(self, query, top_k=3):
        """Retrieve the top-k most relevant documents from the knowledge base."""
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.vector_index.search(np.array([query_embedding]), top_k)
        retrieved_texts = [self.knowledge_base[i]["text"] for i in indices[0]]
        return retrieved_texts

    def query_gemini_with_rag(self, prompt, query):
        """Query the generative model with additional context from the knowledge base."""
        retrieved_context = self._retrieve_context(query)
        full_prompt = f"{prompt}\n\nAdditional Context:\n" + "\n".join(retrieved_context)
        return self.query_gemini(full_prompt)

    def _update_knowledge_base_after_interview(self):
        """Update the knowledge base with the latest conversation history."""
        for msg in self.conversation_history:
            self._add_to_knowledge_base(msg["content"])

    def _run_interview_logic(self):
        try:
            super()._run_interview_logic()
        finally:
            # Update the knowledge base after the interview ends
            self._update_knowledge_base_after_interview()
class RAGExpertTechnicalInterviewer(ExpertTechnicalInterviewer):
    def __init__(self, model="gemini-2.0-flash", accent="indian"):
        super().__init__(model, accent)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
        self.knowledge_base_path = "knowledge_base.json"
        self.vector_index_path = "vector_index.faiss"
        self.vector_dimension = 384  # MiniLM-L6 outputs 384-dimensional vectors
        self.vector_index = self._load_or_create_vector_index()
        self.knowledge_base = self._load_knowledge_base()

    def _load_or_create_vector_index(self):
        """Load an existing FAISS index or create a new one."""
        if os.path.exists(self.vector_index_path):
            return faiss.read_index(self.vector_index_path)
        else:
            return faiss.IndexFlatL2(self.vector_dimension)

    def _load_knowledge_base(self):
        """Load the knowledge base from a JSON file."""
        if os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, "r") as f:
                return json.load(f)
        return []

    def _save_knowledge_base(self):
        """Save the knowledge base to a JSON file."""
        with open(self.knowledge_base_path, "w") as f:
            json.dump(self.knowledge_base, f, indent=4)

    def _save_vector_index(self):
        """Save the FAISS index to disk."""
        faiss.write_index(self.vector_index, self.vector_index_path)

    def _add_to_knowledge_base(self, text):
        """Add new text to the knowledge base and update the vector index."""
        embedding = self.embedding_model.encode(text)
        self.knowledge_base.append({"text": text, "embedding": embedding.tolist()})
        self.vector_index.add(np.array([embedding]))
        self._save_knowledge_base()
        self._save_vector_index()

    def _retrieve_context(self, query, top_k=3):
        """Retrieve the top-k most relevant documents from the knowledge base."""
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.vector_index.search(np.array([query_embedding]), top_k)
        retrieved_texts = [self.knowledge_base[i]["text"] for i in indices[0]]
        return retrieved_texts

    def query_gemini_with_rag(self, prompt, query):
        """Query the generative model with additional context from the knowledge base."""
        retrieved_context = self._retrieve_context(query)
        full_prompt = f"{prompt}\n\nAdditional Context:\n" + "\n".join(retrieved_context)
        return self.query_gemini(full_prompt)

    def _update_knowledge_base_after_interview(self):
        """Update the knowledge base with the latest conversation history."""
        for msg in self.conversation_history:
            self._add_to_knowledge_base(msg["content"])

    def _run_interview_logic(self):
        try:
            super()._run_interview_logic()
        finally:
            # Update the knowledge base after the interview ends
            self._update_knowledge_base_after_interview()
if __name__ == "__main__":
    try:
        interviewer = ExpertTechnicalInterviewer()
        interviewer.start_interview()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Please check your environment setup:")
        print("1. GEMINI_API_KEY in .env file")
        print("2. Microphone and camera permissions")
        print("3. Required Python packages installed")
    
    print("\nInterview session ended. Camera automatically turned off.")
    print("Thank you for using the Enhanced Technical Interview Bot!")