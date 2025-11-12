"use client"
import { useState, useEffect, useRef } from "react"
import { Mic, MicOff, PhoneOff, Code, Camera, CameraOff, Users, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import CodeEditor from "@/components/code-editor"
import AIInterviewerAvatar from "@/components/ai-interviewer-avatar"
import FaceDetection from "@/components/face-detection"
import TranscriptFooter from "./TranscriptFooter"
// ✅ After imports, before `export default function...`
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}
 interface Message {
    speaker: "AI" | "User"
    message: string
    timestamp: number
  }


export default function GoogleMeetInterview() {
  const [isMuted, setIsMuted] = useState(false)
  const [isVideoOff, setIsVideoOff] = useState(false)
  const [isAISpeaking, setIsAISpeaking] = useState(false)
  const [selectedLanguage, setSelectedLanguage] = useState("JavaScript")
  const [interviewStarted, setInterviewStarted] = useState(false)
  const [showCodeEditor, setShowCodeEditor] = useState(false)
  const [code, setCode] = useState("// Write your code here...")
  const [output, setOutput] = useState("")
  const [isSubmitted, setIsSubmitted] = useState(false)
  const [question, setQuestion] = useState("")
  const [warnings, setWarnings] = useState<string[]>([])
  const [transcript, setTranscript] = useState<Message[]>([])
  const allWarnings = [...warnings]
  const [showEndPopup, setShowEndPopup] = useState(false)
  const [isTabActive, setIsTabActive] = useState(true)
  const [waiting, setWaiting] = useState(false)
  const [showViolationPopup, setShowViolationPopup] = useState(false)
  const [gotResponse, setGotResponse] = useState(false)
  const [interviewEndedHandled, setInterviewEndedHandled] = useState(false)
  const [interviewTerminated, setInterviewTerminated] = useState(false)
  



  const handleUserSend = (message: string) => {
    console.log("User said:", message)

  }
  const [interviewStatus, setInterviewStatus] = useState({

    active: false,
    stage: 'not_started',
    skill_questions_asked: 0,
    coding_questions_asked: 0,
    current_domain: 'unknown',
    current_question: null
  })

  const [aiState, setAiState] = useState({
    is_speaking: false,
    is_listening: false,
    current_message: '',
    last_speech_start: null,
    last_speech_end: null
  })

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
 

  
  const fetchInterviewStatus = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/interview-status")
      const data = await res.json()
      console.log(" Interview Status:", data)
      setInterviewStatus(data)

      // ✅ Handle forced termination or conclusion
      if (!data.active || data.stage === 'terminated_due_to_violations' || data.stage === 'concluded') {
        console.log(" Interview ended. Showing popup and resetting UI.")

        // Stop video/audio stream
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach(track => track.stop())
          mediaStreamRef.current = null
        }

        if (videoRef.current) {
          videoRef.current.srcObject = null
        }

        // Show popup and reset state
        setShowEndPopup(true)
        setInterviewStarted(false)
        setShowCodeEditor(false)
        setIsAISpeaking(false)
        setWarnings([])
        return
      }


      // Auto-open code editor if in coding stage
      if (data.stage === 'coding_challenges' && !showCodeEditor && data.current_question) {
        setQuestion(data.current_question)
        setShowCodeEditor(true)
      }

    } catch (error) {
      console.error("Failed to fetch interview status:", error)
    }
  }


  // Fetch coding question
  const fetchCodingQuestion = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/current-coding-question")
      const data = await res.json()
      console.log("📝 Coding Question Response:", data)

      if (data.question && !data.error) {
        setQuestion(data.question)
        return data.question
      } else {
        console.warn("No question received:", data.error)
        return null
      }
    } catch (error) {
      console.error("Error fetching coding question:", error)
      return null
    }
  }




  // Handle clicking on Code Editor button
  const handleCodeEditorClick = async () => {
    try {
      console.log("🔵 Code Editor button clicked")
      console.log("Current interview status:", interviewStatus)

      // Check if we're already in coding stage
      if (interviewStatus.stage === 'coding_challenges') {
        console.log("✅ Already in coding stage, fetching question...")
        const fetchedQuestion = await fetchCodingQuestion()
        if (fetchedQuestion) {
          setQuestion(fetchedQuestion)
          setShowCodeEditor(true)
        } else {
          alert("⚠️ No coding question available. Please wait or try again.")
        }
        return
      }

      // If not in coding stage, try to trigger transition
      console.log("🚀 Triggering transition to coding stage...")
      const triggerRes = await fetch("http://localhost:5000/api/process-speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "ready_for_coding" }),
      })

      const triggerData = await triggerRes.json()
      console.log("🧠 Backend response to ready_for_coding:", triggerData)

      if (triggerData.question) {
        setQuestion(triggerData.question)
        setShowCodeEditor(true)
        // Update local status immediately
        setInterviewStatus(prev => ({
          ...prev,
          stage: 'coding_challenges',
          current_question: triggerData.question
        }))
      } else {
        // Wait a moment and fetch status
        setTimeout(async () => {
          await fetchInterviewStatus()
          const fetchedQuestion = await fetchCodingQuestion()
          if (fetchedQuestion) {
            setQuestion(fetchedQuestion)
            setShowCodeEditor(true)
          }
        }, 1000)
      }

    } catch (err) {
      console.error("❌ Error during code editor activation:", err)
      alert("Something went wrong while switching to coding. Check backend.")
    }
  }

  // End Call
  const handleEndCall = async () => {
    try {
      await fetch("http://localhost:5000/api/end-interview", { method: "POST" })
      setInterviewStarted(false)
      setShowEndPopup(true)
      setShowCodeEditor(false)
      setWarnings([])
    } catch (err) {
      console.error("❌ Failed to end interview:", err)
    }
  }


  // Start Interview
  useEffect(() => {
    if (interviewStarted && !showCodeEditor) {
      fetch("http://localhost:5000/api/start-interview", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          if (data?.message) console.log("Interview started:", data.message)
          fetchInterviewStatus()
        })
        .catch(() => console.error("❌ Failed to start interview"))
    }
  }, [interviewStarted])

  // Replace your transcript fetching useEffect in advanced-interview-interface.tsx with this:

  const fetchAiState = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/ai-state")
      const data = await res.json()
      console.log("🤖 AI State:", data)
      setAiState(data)
      setIsAISpeaking(data.is_speaking)
    } catch (error) {
      console.error("Failed to fetch AI state:", error)
    }
  }

  const handleStartInterview = async () => {
    // Reset backend first
    await fetch("http://localhost:5000/api/reset-interview", { method: "POST" })

    // Then start new interview
    setInterviewStarted(true)
    setShowEndPopup(false)
  }
  // Tab switch detection
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Tab switch detected — log to backend only
        fetch("http://localhost:5000/api/log-warning", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            type: "tab_switch",
            timestamp: Date.now(),
            message: "User switched tabs or minimized window"
          })
        }).catch(console.error)
      }
    };

    if (interviewStarted) {
      document.addEventListener("visibilitychange", handleVisibilityChange);
      return () => {
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      };
    }
  }, [interviewStarted]);


  useEffect(() => {
    const fetchTranscript = async () => {
      try {
        console.log("🔍 Fetching transcript...")
        const res = await fetch("http://localhost:5000/api/transcript")

        if (!res.ok) {
          console.error("❌ Transcript fetch failed:", res.status, res.statusText)
          return
        }

        const data = await res.json()
        console.log("📝 Raw transcript data:", data)

        if (data.transcript && data.transcript.length > 0) {
          console.log("✅ Setting transcript with", data.transcript.length, "entries")
          setTranscript(data.transcript)
        } else {
          console.log("📭 No transcript entries found")
        }
      } catch (err) {
        console.error("❌ Failed to fetch transcript:", err)
      }
    }

    if (interviewStarted) {
      // Fetch immediately
      fetchTranscript()

      // Then fetch every 3 seconds
      const interval = setInterval(fetchTranscript, 3000)
      return () => clearInterval(interval)
    }
  }, [interviewStarted])

  useEffect(() => {
    if (interviewStarted) {
      // Fetch immediately
      fetchAiState()

      // Then poll every 500ms for responsive UI
      const interval = setInterval(fetchAiState, 500)
      return () => clearInterval(interval)
    }
  }, [interviewStarted])

  // Also add this debug useEffect to monitor transcript changes:
  useEffect(() => {
    console.log("📋 Transcript updated:", transcript.length, "entries", transcript)
  }, [transcript])

  useEffect(() => {
    if (interviewStarted && !mediaStreamRef.current && !isVideoOff) {
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: true })
        .then((stream) => {
          mediaStreamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => console.error("Camera access denied:", err));
    } else if (mediaStreamRef.current && videoRef.current) {
      videoRef.current.srcObject = mediaStreamRef.current;
    }
  }, [interviewStarted, isVideoOff]);

  useEffect(() => {
  if (!showCodeEditor && interviewStarted && videoRef.current && videoRef.current.srcObject == null) {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: true })
      .then((stream) => {
        mediaStreamRef.current = stream
        videoRef.current.srcObject = stream
      })
      .catch((err) => console.error("Error reinitializing video after code editor:", err))
  }
}, [showCodeEditor])

  const listenToUser = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return alert("Speech recognition not supported");

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = async (event: any) => {
      const transcript = event.results[0][0].transcript;
      console.log("User said:", transcript);

      const res = await fetch("http://localhost:5000/api/process-speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: transcript }),
      });

      const data = await res.json();
      console.log("Backend replied:", data);

      fetchInterviewStatus();
    };

    recognition.onerror = (event: any) => {
      console.error("Speech error:", event.error);
    };

    recognition.start();
  };

  // Submit code to backend
  const submitCode = async () => {
    setIsSubmitted(true)
    setOutput("🚀 Submitting code...")

    try {
      const res = await fetch("http://localhost:5000/api/submit-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          language: selectedLanguage
        })
      })

      const data = await res.json()
      console.log("📦 Backend evaluation:", data)

      const finalOutput = data.output || "✅ Code submitted successfully"
      setOutput(finalOutput)



      // Now inform backend to continue the interview
      await fetch("http://localhost:5000/api/process-speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "done_coding" }),
      })

      setShowCodeEditor(false)

    } catch (err) {
      console.error("❌ Code submission error:", err)
      setOutput("❌ Submission failed. Check console.")
    }
  }

  useEffect(() => {
    if (!interviewStarted) return;

    const interval = setInterval(() => {
      fetch("http://localhost:5000/api/get-warnings")
        .then(res => res.json())
        .then(data => {
          const newWarnings = data.warnings?.map((w: any) => `⚠️ ${w.message}`) || []
          setWarnings(newWarnings)
          setInterviewTerminated(data.stage === "terminated_due_to_violations")
        })
        .catch(console.error)
    }, 3000)

    return () => clearInterval(interval)
  }, [interviewStarted])

  // Poll interview status
  useEffect(() => {
    if (interviewStarted) {
      const interval = setInterval(fetchInterviewStatus, 3000)
      return () => clearInterval(interval)
    }
  }, [interviewStarted])

  // Poll tab/camera warnings
  // Replace the existing warning polling useEffect with this:
  useEffect(() => {
    if (!interviewStarted) return;

    const pollWarnings = () => {
      fetch("http://localhost:5000/api/get-warnings")
        .then(res => res.json())
        .then(data => {
          console.log("🚨 Fetched warnings:", data); // Debug log

          if (data?.warnings) {
            const formattedWarnings = data.warnings.map((w: any) => {
              switch (w.type) {
                case "face_absence":
                  return "⚠️ Face not visible in camera";
                case "gaze_absence":
                  return "⚠️ Looking away from screen detected";
                case "camera_off":
                  return "⚠️ Camera turned off";
                case "tab_switch":
                  return "⚠️ Tab switch detected";
                default:
                  return `⚠️ ${w.message || "Unknown warning"}`;
              }
            });

            console.log("📋 Setting warnings:", formattedWarnings); // Debug log
            setWarnings(formattedWarnings);

            // Check for termination
            if (data.warnings.length >= 3) {
              console.log("🔴 3+ violations detected, fetching status...");
              fetchInterviewStatus();
            }
          }
        })
        .catch(error => {
          console.error("❌ Error fetching warnings:", error);
        });
    };

    // Poll immediately, then every 3 seconds
    pollWarnings();
    const interval = setInterval(pollWarnings, 3000);

    return () => clearInterval(interval);
  }, [interviewStarted]); // Only depend on interviewStarted

  useEffect(() => {
    if (
      !interviewEndedHandled &&
      (!interviewStatus.active || interviewStatus.stage === "terminated_due_to_violations" || interviewStatus.stage === "concluded")
    ) {
      console.log("📢 Interview ended")

      // Stop media stream
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop())
        mediaStreamRef.current = null
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }

      // Show violation-specific popup if terminated due to violations
      if (interviewStatus.stage === "terminated_due_to_violations") {
        setShowViolationPopup(true)  // Show violation popup first
      } else {
        setShowEndPopup(true)  // Show regular end popup for other cases
      }

      setInterviewEndedHandled(true)
      setIsAISpeaking(false)
    }
  }, [interviewStatus])


  // Debug logging for status changes
  useEffect(() => {
    console.log("🔄 Interview Status Changed:", interviewStatus)
  }, [interviewStatus])

  if (!interviewStarted) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
        <Button
          onClick={() => {
            setInterviewStarted(true)
            setShowEndPopup(false)  // ✅ Clear previous popup
          }}

          className="absolute top-4 right-4 bg-blue-600 hover:bg-blue-700"
        >
      Start Interview
        </Button>
        <h1 className="text-2xl font-bold mb-4">Welcome to the AI Interview</h1>
        <p className="text-gray-400">Click the button above to begin.</p>
      </div>
    )
  }

  const runCode = async () => {
    setOutput("⚙️ Running code...");

    try {
      const res = await fetch("http://localhost:5000/api/submit-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          language: selectedLanguage,
        }),
      });

      const data = await res.json();
      setOutput(data.output || "✅ Code ran successfully.");
    } catch (err) {
      console.error("❌ Code run error:", err);
      setOutput("❌ Failed to run code. Check console.");
    }
  };


  if (showCodeEditor) {
    return (
      <div className="h-screen bg-gray-900 text-white p-4">
        <CodeEditor
          code={code}
          language={selectedLanguage}
          onChange={setCode}
          onRun={runCode}
          onSubmit={submitCode}
          output={output}
          isSubmitted={isSubmitted}
          question={question}
          videoRef={videoRef}
          isVideoOff={isVideoOff}
          isMuted={isMuted}
          isAISpeaking={isAISpeaking}
          isAIListening={!isAISpeaking}
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      <div className="w-full flex justify-center pt-4">
        <FaceDetection />
      </div>
      {warnings.length > 0 && (
        <div className="bg-red-900 text-red-300 p-2 text-sm flex items-center space-x-2 justify-center">
          <AlertCircle className="w-4 h-4" />
          <span>{warnings[warnings.length - 1]}</span>
          {warnings.length > 1 && (
            <span className="bg-red-700 px-2 py-1 rounded text-xs">
              {warnings.length} violations
            </span>
          )}
        </div>
      )}




      {showEndPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-white text-black rounded-lg shadow-lg p-6 max-w-md text-center">
            <h2 className="text-2xl font-bold mb-4">Interview Ended</h2>
            <p className="mb-4">The interview has ended due to repeated violations or has been completed.</p>
            <Button
              className="bg-blue-600 text-white hover:bg-blue-700"
              onClick={() => setShowEndPopup(false)}
            >
              Return to Start
            </Button>
          </div>
        </div>
      )}



      {/* Violation Popup - Shows first */}
      {showViolationPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50">
          <div className="bg-red-50 border-2 border-red-500 text-red-900 rounded-lg shadow-2xl p-8 max-w-md text-center">
            <div className="text-6xl mb-4">⚠️</div>
            <h2 className="text-2xl font-bold mb-4 text-red-700">Interview Terminated</h2>
            <p className="mb-6 text-lg">
              You have committed <span className="font-bold text-red-600">3 violations</span>.
              <br />We are concluding your interview.
            </p>
            <Button
              className="bg-red-600 text-white hover:bg-red-700 px-8 py-3 text-lg"
              onClick={() => {
                setShowViolationPopup(false)
                setInterviewStarted(false)
                setShowCodeEditor(false)
                setWarnings([])
                // Reset everything and go to start page
              }}
            >
              Okay, I Understand
            </Button>
          </div>
        </div>
      )}
      <div className="flex-1 relative bg-black overflow-hidden">
        <div className="h-full grid grid-cols-2 gap-2 p-4">
          {/* Local Video Feed */}
          <div className="relative bg-gray-800 rounded-lg overflow-hidden">
            {isVideoOff ? (
              <div className="w-full h-full bg-gray-700 flex items-center justify-center">
                <CameraOff className="w-16 h-16 text-gray-400" />
                <span className="text-gray-300">Camera is off</span>
              </div>
            ) : (
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover"
              />
            )}
            <div className="absolute bottom-4 left-4 bg-black/50 rounded-full px-3 py-1">
              <span className="text-white text-sm font-medium">You</span>
            </div>
            {isMuted && (
              <div className="absolute top-4 left-4 bg-red-500 rounded-full p-2">
                <MicOff className="w-4 h-4 text-white" />
              </div>
            )}
          </div>

          {/* AI Interviewer Avatar */}
          <div className="relative bg-gray-800 rounded-lg overflow-hidden flex items-center justify-center">
            <AIInterviewerAvatar
              isSpeaking={isAISpeaking}
              isListening={!isAISpeaking}
              currentMessage={aiState.current_message}
              showVoiceActivity={true}
            />
            <div className="absolute bottom-16 left-4 right-4 p-3 max-h-24 overflow-y-auto z-0">
              <TranscriptFooter
                transcript={transcript}
                onSendMessage={handleUserSend}
                waitingForResponse={waiting}
                isAISpeaking={isAISpeaking}
                messageReceived={gotResponse}
              />
            </div>

          </div>


          {/* Top Bar */}

        </div>

      </div>

      {/* Control Bar */}
      <div className="bg-gray-800 px-6 py-4 flex items-center justify-center space-x-6 relative z-10">

        <Button
          onClick={handleCodeEditorClick}
          variant="ghost"
          size="icon"
          className="rounded-full w-14 h-14 text-white bg-blue-600 hover:bg-blue-700"
        >
          {interviewStatus.stage === 'coding_challenges' && !question ? (
            <span className="animate-pulse">⏳</span>
          ) : (
            <Code className="w-5 h-5" />
          )}
        </Button>

        <Button
          onClick={handleEndCall}
          variant="ghost"
          size="icon"
          className="rounded-full w-14 h-14 bg-red-600 hover:bg-red-700 text-white"
        >
          <PhoneOff className="w-5 h-5" />
        </Button>
      </div>
    </div>
  )
}