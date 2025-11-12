from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from datetime import datetime, timedelta
import secrets
import smtplib
from email.mime.text import MIMEText
import json
import threading
import random

from backend import ExpertTechnicalInterviewer
from shared_state import interview_state, save_to_conversation_history, ai_state



app = Flask(__name__)
CORS(app)

# Global variables
interviewer = None
interview_thread = None
interview_stop_event = threading.Event()

# Initialize violation state tracking in interview_state
if 'current_violation_state' not in interview_state:
    interview_state['current_violation_state'] = {
        'face_absent_since': None,
        'gaze_away_since': None,
        'last_violation_type': None
    }

# Initialize warnings list if not exists
if 'warnings' not in interview_state:
    interview_state['warnings'] = []

def initialize_interviewer():
    global interviewer
    try:
        interviewer = ExpertTechnicalInterviewer(accent="indian")
        return True
    except Exception as e:
        print(f"Failed to initialize interviewer: {e}")
        return False

@app.route('/api/start-interview', methods=['POST'])
def start_interview():
    global interviewer, interview_state

    if not initialize_interviewer():
        return jsonify({'error': 'Failed to initialize interviewer'}), 500

    # Reset interview state
    interview_state.update({
        'active': True,
        'stage': 'greeting',
        'conversation_history': [],
        'skill_questions_asked': 0,
        'coding_questions_asked': 0,
        'warnings': [],
        'personal_info_collected': False,
        'tech_background_collected': False,
        'skills_collected': '',
        'current_domain': None,
        'current_question': None,
        'interview_links': {}
    })

   
    try:
        import threading
        threading.Thread(target=interviewer.start_interview, daemon=True).start()
    except Exception as e:
        print(f"🔥 Interview launch error: {e}")
        return jsonify({'error': 'Failed to launch interview thread'}), 500

    return jsonify({
        'status': 'started',
        'interview_active': True,
        'stage': 'greeting'
    })





# In your process_speech function, replace the conversation history saving with:
def save_to_conversation_history(role, content):
    """Helper function to save messages with proper timestamps"""
    interview_state["conversation_history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })
from datetime import datetime

def complete_interview_reset():
    """Complete cleanup when interview ends"""
    global interviewer, interview_thread, interview_stop_event
    
    print("🧹 COMPLETE INTERVIEW CLEANUP STARTING...")
    
    # Reset interview state completely
    interview_state.clear()
    interview_state.update({
        'active': False,
        'stage': 'not_started',
        'conversation_history': [],
        'skill_questions_asked': 0,
        'coding_questions_asked': 0,
        'warnings': [],
        'personal_info_collected': False,
        'tech_background_collected': False,
        'skills_collected': '',
        'current_domain': None,
        'current_question': None,
        'interview_links': {},
        'latest_code': '',
        'language': ''
    })
    
    # Reset AI state completely
    ai_state.clear()
    ai_state.update({
        'is_speaking': False,
        'is_listening': False,
        'current_message': '',
        'last_speech_start': None,
        'last_speech_end': None
    })
    
    # Stop and reset interviewer
    if interviewer:
        interviewer = None
    
    # Stop threads
    interview_stop_event.set()
    if interview_thread:
        interview_thread = None
    interview_stop_event.clear()
    
    print("✅ COMPLETE CLEANUP FINISHED - Ready for new interview")

@app.route('/api/transcript', methods=['GET'])
def get_transcript():
    history = interview_state.get("conversation_history", [])

    transcript = []
    for entry in history:
        if entry.get("role") in ["user", "assistant"]:
            transcript.append({
                "speaker": "User" if entry["role"] == "user" else "AI",
                "message": entry["content"],
                "timestamp": int(datetime.fromisoformat(entry["timestamp"]).timestamp() * 1000)
            })

    return jsonify({ "transcript": transcript })


@app.route('/api/problems/random', methods=['GET'])
def get_random_problem():
    problems = [
        "Write a function to reverse a string.",
        "Find the factorial of a number using recursion.",
        "Implement binary search on a sorted list."
    ]
    return jsonify({
        "question": random.choice(problems)
    })

@app.route('/api/process-speech', methods=['POST'])
def process_speech():
    global interviewer, interview_state

    data = request.json
    user_input = data.get('text', '').strip()

    if not user_input:
        return jsonify({'error': 'No input'}), 400

    # ✅ Save user message ONCE, only if not a command


    print("📥 process_speech called with:", user_input)

    response = ""

    # ✅ Handle coding start
    if user_input.lower() == "ready_for_coding":
        interview_state['stage'] = 'coding_challenges'
        interview_state['coding_questions_asked'] = 1
        coding_question = interviewer._generate_coding_question(
            interview_state.get("current_domain", "python")
        )
        interview_state['current_question'] = coding_question
        response = "✅ Great! Now let's move to the coding challenge."

        print("🤖 AI (coding intro):", response)
        try:
            speak_with_state_tracking(interviewer, response)
        except Exception as e:
            print("TTS error:", e)

        save_to_conversation_history("assistant", response)

        return jsonify({
            "response": response,
            "question": coding_question,
            "stage": "coding_challenges"
        })

    # ✅ Handle coding done
    if user_input.lower() == "done_coding":
        response = "Thanks for your submission. Let's move on to any questions you might have."
        interview_state['stage'] = 'doubt_clearing'

        print("🤖 AI (done coding):", response)
        try:
            speak_with_state_tracking(interviewer, response)
        except Exception as e:
            print("TTS error:", e)

        save_to_conversation_history("assistant", response)

        return jsonify({
            "response": response,
            "stage": "doubt_clearing"
        })

    # ✅ Main Interview Logic
    try:
        if interview_state['stage'] == 'greeting':
            response = "Thanks for the intro! Tell me more about your technical background."
            interview_state['stage'] = 'tech_background'

        elif interview_state['stage'] == 'tech_background':
            interview_state['stage'] = 'skill_questions'
            followup = interviewer._add_domain_specific_followup(
                interview_state.get("current_domain", "general")
            )
            response = f"Awesome! Let's begin your technical round.\n\n{followup}"

        elif interview_state['stage'] == 'skill_questions':
            followup = interviewer._add_domain_specific_followup(
                interview_state.get("current_domain", "general")
            )
            response = f"Great. Here's your next question:\n\n{followup}"

        else:
            response = "Let's continue."

        print("🤖 AI says:", response)
        try:
            speak_with_state_tracking(interviewer, response)
        except Exception as e:
            print("TTS error:", e)

        save_to_conversation_history("assistant", response)

        return jsonify({
            "response": response,
            "stage": interview_state['stage']
        })

    except Exception as e:
        print("❌ Error during process_speech:", e)
        return jsonify({"error": str(e)}), 500

def speak_with_state_tracking(interviewer, text):
    from shared_state import ai_state
    from datetime import datetime
    import threading
    import time

    print("🧠 [DEBUG] speak_with_state_tracking CALLED")
    
    ai_state['is_speaking'] = True
    ai_state['is_listening'] = False
    ai_state['current_message'] = text
    ai_state['last_speech_start'] = datetime.utcnow().isoformat()

    print("🎙️ AI STARTED SPEAKING:", ai_state)

    try:
        print("🧠 Calling interviewer.speak()...")
        interviewer.speak(text)
        print("✅ interviewer.speak() finished.")
    except Exception as e:
        print("❌ Error during interviewer.speak():", e)

    def delayed_reset():
        print("🕒 Waiting to reset AI state...")
        time.sleep(2.5)
        mark_ai_finished_speaking()

    threading.Thread(target=delayed_reset).start()



def mark_ai_finished_speaking():
    from shared_state import ai_state
    from datetime import datetime

    ai_state['is_speaking'] = False
    ai_state['is_listening'] = True
    ai_state['last_speech_end'] = datetime.utcnow().isoformat()

    print("🔕 AI FINISHED SPEAKING:", ai_state)

def stop_interview_thread():
    """Stop the interview thread gracefully"""
    global interview_thread, interview_stop_event
    
    print("🛑 Stopping interview thread...")
    interview_stop_event.set()
    
    if interview_thread and interview_thread.is_alive():
        interview_thread.join(timeout=5)
        print("✅ Interview thread stopped")
    
    interview_thread = None
    interview_stop_event.clear()
    
def save_to_conversation_history(role, content):
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    interview_state["conversation_history"].append(entry)
    print(f"💾 SAVED TO HISTORY: [{role.upper()}] {content}")
    print(f"🧠 Last 2 messages:")
    for msg in interview_state["conversation_history"][-2:]:
        print(f"  - {msg['role']}: {msg['content']}")

def reset_backend_state():
    interview_state.clear()
    interview_state.update({
        'active': False,
        'stage': 'not_started',
        'skill_questions_asked': 0,
        'coding_questions_asked': 0,
        'current_domain': 'unknown',
        'current_question': None
    })
    Warning.clear()
    # tab_warning.clear()  # Removed because tab_warning is not defined
    interview_stop_event.clear()


@app.route('/api/reset-interview', methods=['POST'])
def reset_interview():
    """Manual reset for new interview"""
    complete_interview_reset()
    return jsonify({'status': 'reset_complete', 'ready_for_new_interview': True})

@app.route('/api/log-warning', methods=['POST'])
def log_warning():
    global interviewer 
    data = request.json
    warning_type = data.get('type')
    timestamp = data.get('timestamp')
    message = data.get('message')
    
    # Add to interview state
    if 'warnings' not in interview_state:
        interview_state['warnings'] = []
    
    # Add the warning
    interview_state['warnings'].append({
        'type': warning_type,
        'timestamp': timestamp,
        'message': message,
        'stage': interview_state.get('stage', 'unknown')
    })
    
    violation_count = len(interview_state['warnings'])
    print(f"🚨 WARNING LOGGED: {warning_type} - {message} (Violation #{violation_count})")
    
    # Optional: Add escalating responses
    if violation_count >= 3:
        print("🔴 CRITICAL: Multiple violations detected!")
        # You could end the interview here or send additional warnings
        interview_state['active'] = False
        interview_state['stage'] = 'terminated_due_to_violations'
        if interviewer:
            try:
                # ✅ Properly stop AI interviewer
                docx_path, feedback_path = interviewer.end_interview()
                print("📄 Transcript and feedback saved.")
            except Exception as e:
                print(f"❌ Error during interview termination: {e}") # Stop the thread
        interviewer = None
        reset_backend_state()
        print("✅ Interview fully terminated due to violations")
    
    return jsonify({
        'status': 'logged',
        'violation_count': violation_count,
        'total_violations': violation_count,
        'interview_terminated': violation_count >= 3,
        'interview_active': interview_state['active'],
        'stage': interview_state['stage']
    })

@app.route('/api/get-warnings', methods=['GET'])
def get_warnings():
    # Get warnings from interview_state instead of interviewer
    warnings = interview_state.get('warnings', [])
    
    print(f"📋 Returning {len(warnings)} warnings to frontend")
    for i, warning in enumerate(warnings):
        print(f"  {i+1}. {warning.get('type', 'unknown')}: {warning.get('message', 'No message')}")
    
    return jsonify({
        'warnings': warnings, 
        'count': len(warnings),
        'stage': interview_state.get('stage', 'unknown')
    })


# Fix 3: Add debug endpoint to check conversation history
@app.route('/api/debug-transcript', methods=['GET'])
def debug_transcript():
    """Debug endpoint to check raw conversation history"""
    return jsonify({
        "raw_history": interview_state.get("conversation_history", []),
        "history_length": len(interview_state.get("conversation_history", [])),
        "last_entry": interview_state.get("conversation_history", [])[-1] if interview_state.get("conversation_history") else None
    })


@app.route('/api/ai-state', methods=['GET'])
def get_ai_state():  # 
    ai_state['is_speaking'] = True 
    return jsonify(ai_state)

@app.route('/api/generate-interview-link', methods=['POST'])
def generate_interview_link():
    data = request.json
    recipient_email = data.get('email')
    
    if not recipient_email:
        return jsonify({'error': 'Recipient email is required'}), 400
    
    token = secrets.token_urlsafe(32)
    
    expiration_date = datetime.datetime.now() + timedelta(days=7)
    
    interview_state['interview_links'][token] = {
        'email': recipient_email,
        'expires_at': expiration_date.isoformat(),
        'used': False,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    interview_link = f"https://yourdomain.com/interview/{token}"
    
    try:
        send_interview_link_email(recipient_email, interview_link, expiration_date)
        return jsonify({
            'status': 'success',
            'message': 'Interview link sent successfully',
            'expires_at': expiration_date.isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to send email: {str(e)}',
            'interview_link': interview_link 
        }), 500

def send_interview_link_email(recipient, link, expires_at):
    """Helper function to send interview link email"""
    SMTP_SERVER = "your_smtp_server.com"
    SMTP_PORT = 587
    SMTP_USERNAME = "your_email@example.com"
    SMTP_PASSWORD = "your_email_password"
    SENDER_EMAIL = "no-reply@yourdomain.com"
    
    subject = "Your Technical Interview Link"
    body = f"""
    <p>Hello,</p>
    <p>Here is your interview link: <a href="{link}">{link}</a></p>
    <p>This link will expire on {expires_at.strftime('%Y-%m-%d %H:%M:%S')}.</p>
    <p>Please complete your interview before this date.</p>
    <p>Best regards,<br>Interview Team</p>
    """
    
    msg = MIMEText(body, 'html')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient
    
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

@app.route('/api/validate-interview-link/<token>', methods=['GET'])
def validate_interview_link(token):
    """Endpoint to validate an interview link"""
    if token not in interview_state['interview_links']:
        return jsonify({'valid': False, 'reason': 'Invalid token'}), 404
    
    link_data = interview_state['interview_links'][token]
    expiration_date = datetime.datetime.fromisoformat(link_data['expires_at'])
    
    if datetime.datetime.now() > expiration_date:
        return jsonify({'valid': False, 'reason': 'Link expired'}), 410
    
    if link_data['used']:
        return jsonify({'valid': False, 'reason': 'Link already used'}), 403
    
    return jsonify({
        'valid': True,
        'expires_at': link_data['expires_at'],
        'email': link_data['email']
    })

@app.route('/api/mark-link-used/<token>', methods=['POST'])
def mark_link_used(token):
    """Mark an interview link as used"""
    if token not in interview_state['interview_links']:
        return jsonify({'error': 'Invalid token'}), 404
    
    interview_state['interview_links'][token]['used'] = True
    return jsonify({'status': 'success'})


@app.route('/api/interview-status', methods=['GET'])
def get_interview_status():
    """Get current interview status and progress"""
    return jsonify({
        'active': interview_state['active'],
        'stage': interview_state['stage'],
        'skill_questions_asked': interview_state['skill_questions_asked'],
        'coding_questions_asked': interview_state['coding_questions_asked'],
        'total_skill_questions': 3,
        'total_coding_questions': 2,
        'current_domain': interview_state.get('current_domain', 'unknown'),
        'current_question': interview_state.get('current_question')
    })


@app.route('/api/face-status', methods=['POST'])
def face_status():
    global interviewer
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        face_present = data.get("face_present")
        gaze_away = data.get("gaze_away", False)
        
        if face_present is None:
            return jsonify({'error': 'face_present field is required'}), 400
            
        if not interviewer:
            return jsonify({'error': 'No interviewer session active'}), 400
            
        # Initialize structures
        if 'warnings' not in interview_state:
            interview_state['warnings'] = []
        if 'current_violation_state' not in interview_state:
            interview_state['current_violation_state'] = {
                'face_absent_since': None,
                'gaze_away_since': None,
                'last_violation_type': None
            }
        
        current_time = datetime.utcnow()
        violation_state = interview_state['current_violation_state']
        new_violation_logged = False
        
        # Handle face absence detection
        if not face_present:
            if violation_state['face_absent_since'] is None:
                # NEW face absence event starts
                violation_state['face_absent_since'] = current_time
                violation_state['last_violation_type'] = 'face_absence'
                
                # Log as new violation
                interview_state['warnings'].append({
                    'type': 'face_absence',
                    'timestamp': current_time.isoformat(),
                    'message': 'Face not visible in camera',
                    'stage': interview_state.get('stage', 'unknown'),
                    'severity': 'high',
                    'event_start': current_time.isoformat()
                })
                
                new_violation_logged = True
                interviewer._handle_cheating_attempt("face_absence")
                interviewer.speak("Please ensure your face is visible in the camera.")
            
            # Reset gaze away state if face is not present
            violation_state['gaze_away_since'] = None
            
        else:
            # Face is present - end face absence event if it was active
            if violation_state['face_absent_since'] is not None:
                # Update the last face absence warning with end time
                for warning in reversed(interview_state['warnings']):
                    if warning['type'] == 'face_absence' and 'event_end' not in warning:
                        warning['event_end'] = current_time.isoformat()
                        duration = (current_time - violation_state['face_absent_since']).total_seconds()
                        warning['duration_seconds'] = duration
                        break
                
                violation_state['face_absent_since'] = None
            
            # Handle gaze away detection (only when face is present)
            if gaze_away:
                if violation_state['gaze_away_since'] is None:
                    # NEW gaze away event starts
                    violation_state['gaze_away_since'] = current_time
                    violation_state['last_violation_type'] = 'gaze_absence'
                    
                    # Log as new violation
                    interview_state['warnings'].append({
                        'type': 'gaze_absence',
                        'timestamp': current_time.isoformat(),
                        'message': 'Looking away from screen detected',
                        'stage': interview_state.get('stage', 'unknown'),
                        'severity': 'medium',
                        'event_start': current_time.isoformat()
                    })
                    
                    new_violation_logged = True
                    interviewer._handle_cheating_attempt("gaze_absence")
                    interviewer.speak("Please avoid looking away from the screen.")
            else:
                # Gaze is back - end gaze away event if it was active
                if violation_state['gaze_away_since'] is not None:
                    # Update the last gaze away warning with end time
                    for warning in reversed(interview_state['warnings']):
                        if warning['type'] == 'gaze_absence' and 'event_end' not in warning:
                            warning['event_end'] = current_time.isoformat()
                            duration = (current_time - violation_state['gaze_away_since']).total_seconds()
                            warning['duration_seconds'] = duration
                            break
                    
                    violation_state['gaze_away_since'] = None
        
        # Check violation count for termination
        violation_count = len(interview_state.get('warnings', []))
        interview_terminated = False
        
        if violation_count >= 3:
            print("🔴 CRITICAL: Multiple violation events detected!")
            interview_terminated = True
            
            # End any active violations
            if violation_state['face_absent_since'] or violation_state['gaze_away_since']:
                for warning in reversed(interview_state['warnings']):
                    if 'event_end' not in warning:
                        warning['event_end'] = current_time.isoformat()
                        if warning['type'] == 'face_absence' and violation_state['face_absent_since']:
                            duration = (current_time - violation_state['face_absent_since']).total_seconds()
                            warning['duration_seconds'] = duration
                        elif warning['type'] == 'gaze_absence' and violation_state['gaze_away_since']:
                            duration = (current_time - violation_state['gaze_away_since']).total_seconds()
                            warning['duration_seconds'] = duration
                        break
            
            # Update state before termination
            interview_state['active'] = False
            interview_state['stage'] = 'terminated_due_to_violations'
            interview_state['termination_reason'] = 'Multiple face/gaze violation events'
            interview_state['terminated_at'] = current_time.isoformat()
            
            # Terminate interview
            try:
                if interviewer:
                    docx_path, feedback_path = interviewer.end_interview()
                    print(f"📄 Interview terminated. Files saved: {docx_path}, {feedback_path}")
                    interviewer.speak("Interview terminated due to multiple violations.")
            except Exception as e:
                print(f"❌ Error during interview termination: {e}")
            finally:
                interviewer = None
                reset_backend_state()
                print("✅ Interview terminated due to violation events")
        
        return jsonify({
            'status': 'processed',
            'new_violation_logged': new_violation_logged,
            'violation_count': violation_count,
            'face_present': face_present,
            'gaze_away': gaze_away,
            'interview_terminated': interview_terminated,
            'warnings_remaining': max(0, 3 - violation_count) if not interview_terminated else 0,
            'current_violations': {
                'face_absent': violation_state['face_absent_since'] is not None,
                'gaze_away': violation_state['gaze_away_since'] is not None,
                'face_absent_duration': (current_time - violation_state['face_absent_since']).total_seconds() if violation_state['face_absent_since'] else 0,
                'gaze_away_duration': (current_time - violation_state['gaze_away_since']).total_seconds() if violation_state['gaze_away_since'] else 0
            },
            'timestamp': current_time.isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in face_status endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@app.route('/api/end-interview', methods=['POST'])
def end_interview():
    """Manually end the interview"""
    global interviewer
    global interview_state

    print("🛑 Ending interview manually...")
    interview_state['active'] = False
    interview_state['stage'] = 'concluded'

    # Stop interviewer instance and generate outputs
    docx_path, feedback_path = None, None
    if interviewer:
        try:
            docx_path, feedback_path = interviewer.end_interview()  # Proper stop
        except Exception as e:
            print(f"⚠️ Failed to end interview gracefully: {e}")
        interviewer = None  # Kill the object

    reset_backend_state()  # Clear global state, warnings, etc.

    return jsonify({
        'status': 'ended',
        'message': 'Interview has been ended manually.',
        'transcript_path': docx_path,
        'feedback_path': feedback_path
    })



@app.route('/api/export-transcript', methods=['POST'])
def export_transcript():
    """Export conversation history to DOCX"""
    try:
        # Use the method from backend.py
        docx_path = interviewer._save_transcription_to_docx()
        if not docx_path:
            raise Exception("Failed to generate DOCX")
            
        return jsonify({
            'status': 'success',
            'path': docx_path,
            'download_url': f'/api/download-transcript?path={docx_path}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-feedback', methods=['POST'])
def generate_feedback():
    """Generate feedback from transcript"""
    try:
        # First ensure we have a transcript
        if not interviewer.conversation_history:
            raise Exception("No conversation history available")
            
        # Generate DOCX if needed
        docx_path = interviewer._save_transcription_to_docx()
        
        # Generate feedback
        feedback_path = interviewer._generate_feedback_from_docx(docx_path)
        
        return jsonify({
            'status': 'success',
            'feedback_path': feedback_path,
            'download_url': f'/api/download-feedback?path={feedback_path}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-transcript')
def download_transcript():
    """Download generated DOCX transcript"""
    path = request.args.get('path', 'interview_transcript.docx')
    return send_file(path, as_attachment=True)

@app.route('/api/download-feedback')
def download_feedback():
    """Download generated feedback"""
    path = request.args.get('path', 'final_interview_feedback.json')
    return send_file(path, as_attachment=True)

@app.route('/api/current-coding-question', methods=['GET'])
def get_current_coding_question():
    if interview_state['stage'] != 'coding_challenges':
        return jsonify({
            'question': None, 
            'error': f'Not in coding stage. Current stage: {interview_state["stage"]}'
        }), 400
    
    current_question = interview_state.get('current_question')
    if not current_question:
        # Generate a fallback question if none exists
        try:
            current_question = interviewer._generate_coding_question(
                interview_state.get("current_domain", "python")
            )
            interview_state['current_question'] = current_question
        except:
            current_question = "Write a function that takes a string and returns it reversed. For example, 'hello' should return 'olleh'."
            interview_state['current_question'] = current_question
    
    return jsonify({
        'question': current_question,
        'stage': interview_state['stage'],
        'coding_questions_asked': interview_state['coding_questions_asked']
    })

@app.route('/api/submit-code', methods=['POST'])
def submit_code():
    data = request.get_json()
    user_code = data.get("code", "")
    language = data.get("language", "python")

    # Save code and language in global state
    interview_state['latest_code'] = user_code
    interview_state['language'] = language

    try:
    
        output = interviewer._execute_code(language, user_code)

   
        followup = interviewer._coding_followup(user_code, language)

        if followup:
            try:
                from flask_server import speak_with_state_tracking  # import if not already
                speak_with_state_tracking(interviewer, followup)
            except Exception as e:
                print(f"❌ Error while speaking follow-up: {e}")

        success = True
    except Exception as e:
        output = ""
        followup = f"❌ Error while running code: {str(e)}"
        success = False

    return jsonify({
        "success": success,
        "output": output,
        "followup_question": followup
    })



@app.route("/api/generate-coding-question", methods=["POST"])
def api_generate_coding_question():
    data = request.get_json()
    domain = data.get("domain", "python")
    session_id = data.get("session_id", "default")

    try:
        question = interviewer._generate_coding_question(domain)
        return jsonify({"question": question})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Debug endpoint to check interview state
@app.route('/api/debug-state', methods=['GET'])
def debug_state():
    """Debug endpoint to inspect current interview state"""
    return jsonify(interview_state)

@app.route('/api/interview-config', methods=['GET', 'POST'])
def handle_interview_config():
    """Get or update interview configuration"""
    if request.method == 'GET':
        try:
            with open("interview_config.json", "r") as f:
                config = json.load(f)
            return jsonify(config)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == 'POST':
        try:
            config = request.json
            with open("interview_config.json", "w") as f:
                json.dump(config, f, indent=2)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route('/api/client-questions', methods=['GET', 'POST', 'DELETE'])
def handle_client_questions():
    """Manage client-provided questions"""
    try:
        with open("interview_config.json", "r") as f:
            config = json.load(f)
        
        if request.method == 'GET':
            return jsonify({
                "easy_questions": config.get("easy_questions", []),
                "medium_questions": config.get("medium_questions", []),
                "hard_questions": config.get("hard_questions", [])
            })

        elif request.method == 'POST':
            data = request.json
            question_type = data.get("type")
            question = data.get("question")

            if not question_type or not question:
                return jsonify({"error": "Missing type or question"}), 400

            if question_type not in ["easy", "medium", "hard"]:
                return jsonify({"error": "Invalid question type"}), 400

            key = f"{question_type}_questions"
            if key not in config:
                config[key] = []
            
            if question not in config[key]:
                config[key].append(question)
            
            with open("interview_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            return jsonify({"status": "success"})

        elif request.method == 'DELETE':
            data = request.json
            question_type = data.get("type")
            question = data.get("question")

            if not question_type or not question:
                return jsonify({"error": "Missing type or question"}), 400

            key = f"{question_type}_questions"
            if key in config and question in config[key]:
                config[key].remove(question)
            
            with open("interview_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/section-timing', methods=['GET', 'POST'])
def handle_section_timing():
    """Get or update section timing configuration"""
    try:
        with open("interview_config.json", "r") as f:
            config = json.load(f)

        if request.method == 'GET':
            return jsonify(config.get("section_durations", {}))

        elif request.method == 'POST':
            data = request.json
            if "section_durations" not in config:
                config["section_durations"] = {}

            for section, duration in data.items():
                config["section_durations"][section] = duration

            with open("interview_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000,extra_files=['your_app_files'])