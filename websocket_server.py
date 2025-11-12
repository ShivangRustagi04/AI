import asyncio
import websockets
import json
import threading
from backend import ExpertTechnicalInterviewer

class WebSocketInterviewServer:
    def __init__(self):
        self.interviewer = None
        self.connected_clients = set()
        
    async def handle_client(self, websocket, path):
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
    
    async def handle_message(self, websocket, data):
        message_type = data.get('type')
        
        if message_type == 'start_interview':
            # Initialize interviewer
            self.interviewer = ExpertTechnicalInterviewer()
            await self.send_message(websocket, {
                'type': 'interview_started',
                'status': 'Interview initialized'
            })
            
        elif message_type == 'run_code':
            # Execute code using your existing method
            result = self.interviewer._execute_code(
                data['language'], 
                self.create_temp_file(data['code'], data['language'])
            )
            await self.send_message(websocket, {
                'type': 'code_result',
                'output': result
            })
            
        elif message_type == 'submit_solution':
            # Use your existing solution evaluation
            evaluation = self.interviewer.query_gemini(
                f"Evaluate this code solution: {data['code']}"
            )
            await self.send_message(websocket, {
                'type': 'solution_evaluated',
                'evaluation': evaluation
            })
            
        elif message_type == 'speech_input':
            # Process speech using your existing speech recognition
            response = self.interviewer.query_gemini(data['text'])
            # Use your TTS to speak the response
            self.interviewer.speak(response)
            await self.send_message(websocket, {
                'type': 'ai_response',
                'text': response
            })
    
    async def send_message(self, websocket, message):
        await websocket.send(json.dumps(message))
    
    def create_temp_file(self, code, language):
        # Use your existing temp file creation logic
        import tempfile
        extension = self.interviewer._get_file_extension(language)
        with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
            f.write(code)
            return f.name

# Start the WebSocket server
server = WebSocketInterviewServer()
start_server = websockets.serve(server.handle_client, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
