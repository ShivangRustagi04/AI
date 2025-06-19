from flask import Flask, request, jsonify, send_from_directory
import threading
import os
import json
from bot import RAGExpertTechnicalInterviewer 

app = Flask(__name__, static_folder='frontend')

@app.route("/")
def index():
    return send_from_directory('frontend', 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory('frontend', path)

# Global instance of the interviewer
interviewer_instance = None
interview_thread = None

@app.route('/start_interview', methods=['POST'])
def start_interview():
    """
    Start a new interview session.
    """
    global interviewer_instance, interview_thread

    if interviewer_instance and interviewer_instance.interview_active:
        return jsonify({"status": "error", "message": "An interview is already in progress."}), 400

    try:
        # Initialize the RAGExpertTechnicalInterviewer
        data = request.json
        model = data.get("model", "gemini-2.0-flash")
        accent = data.get("accent", "indian")

        interviewer_instance = RAGExpertTechnicalInterviewer(model=model, accent=accent)

        # Start the interview in a separate thread
        interview_thread = threading.Thread(target=interviewer_instance.start_interview)
        interview_thread.daemon = True
        interview_thread.start()

        return jsonify({"status": "success", "message": "Interview started successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start interview: {str(e)}"}), 500


@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Ask a question during the interview.
    """
    global interviewer_instance

    if not interviewer_instance or not interviewer_instance.interview_active:
        return jsonify({"status": "error", "message": "No active interview session found."}), 400

    try:
        data = request.json
        question = data.get("question")

        if not question:
            return jsonify({"status": "error", "message": "Question is required."}), 400

        # Use the query_gemini_with_rag method to generate a response
        response = interviewer_instance.query_gemini_with_rag(question, question)

        return jsonify({"status": "success", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing question: {str(e)}"}), 500


@app.route('/end_interview', methods=['POST'])
def end_interview():
    """
    End the current interview session.
    """
    global interviewer_instance, interview_thread

    if not interviewer_instance or not interviewer_instance.interview_active:
        return jsonify({"status": "error", "message": "No active interview session found."}), 400

    try:
        interviewer_instance.interview_active = False
        if interview_thread and interview_thread.is_alive():
            interview_thread.join(timeout=5)  # Wait for the thread to finish gracefully

        interviewer_instance = None
        return jsonify({"status": "success", "message": "Interview ended successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error ending interview: {str(e)}"}), 500


@app.route('/knowledge_base', methods=['GET'])
def get_knowledge_base():
    """
    Retrieve the current knowledge base.
    """
    global interviewer_instance

    if not interviewer_instance:
        return jsonify({"status": "error", "message": "No interviewer instance found."}), 400

    try:
        knowledge_base = interviewer_instance._load_knowledge_base()
        return jsonify({"status": "success", "knowledge_base": knowledge_base})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error loading knowledge base: {str(e)}"}), 500


@app.route('/add_to_knowledge_base', methods=['POST'])
def add_to_knowledge_base():
    """
    Add new text to the knowledge base.
    """
    global interviewer_instance

    if not interviewer_instance:
        return jsonify({"status": "error", "message": "No interviewer instance found."}), 400

    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"status": "error", "message": "Text is required."}), 400

        interviewer_instance._add_to_knowledge_base(text)
        return jsonify({"status": "success", "message": "Text added to knowledge base successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error adding to knowledge base: {str(e)}"}), 500


if __name__ == '__main__':
    # Ensure necessary environment variables are set
    required_env_vars = ["GEMINI_API_KEY", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        exit(1)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)