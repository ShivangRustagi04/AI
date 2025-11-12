"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Play, Send } from "lucide-react"

interface CodeEditorProps {
  code: string
  language: string
  onChange: (code: string) => void
  onLanguageChange: (language: string) => void 
  onRun: () => void
  onSubmit: () => void
  output: string
  isSubmitted: boolean
  question?: string
  videoRef?: React.RefObject<HTMLVideoElement>
  isVideoOff?: boolean
  isMuted?: boolean
  isAISpeaking?: boolean
  isAIListening?: boolean
  onVideoStreamReady?: (stream: MediaStream) => void
  problemId?: string
}

export default function CodeEditor({
  code,
  language: propLanguage,
  onChange,
  onLanguageChange,
  onRun,
  onSubmit,
  output,
  isSubmitted,
  question,
  videoRef,
  isVideoOff = false,
  isMuted = false,
  isAISpeaking = false,
  isAIListening = false,
  onVideoStreamReady,
  problemId,
}: CodeEditorProps) {
  const [currentQuestion, setCurrentQuestion] = useState(question || "")
  const [loading, setLoading] = useState(!question)
  const [error, setError] = useState<string | null>(null)
  const [localLanguage, setLocalLanguage] = useState(propLanguage)

  // Sync local language state with prop changes
  useEffect(() => {
    setLocalLanguage(propLanguage)
  }, [propLanguage])

  // Handle language selection
  const handleLanguageChange = (newLanguage: string) => {
    setLocalLanguage(newLanguage)
    onLanguageChange(newLanguage)
    console.log("Language changed to:", newLanguage) // For debugging
  }

  // Fetch coding problem from backend
  useEffect(() => {
    if (question) {
      setCurrentQuestion(question)
      setLoading(false)
    } else {
      setCurrentQuestion("⚠️ No coding question available. Please start the interview again.")
      setError("Missing coding question")
      setLoading(false)
    }
  }, [question])

  // Enhanced copy/paste prevention
  useEffect(() => {
    const preventCopyPaste = (e: Event) => {
      e.preventDefault()
      e.stopPropagation()
      return false
    }

    const preventKeyboardShortcuts = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        const forbiddenKeys = ['c', 'v', 'x', 'a', 's', 'z', 'y']
        if (forbiddenKeys.includes(e.key.toLowerCase())) {
          e.preventDefault()
          e.stopPropagation()
          return false
        }
      }
      
      if (e.key === 'F12' || 
          (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'J')) ||
          (e.ctrlKey && e.key === 'u')) {
        e.preventDefault()
        e.stopPropagation()
        return false
      }
    }

    const preventRightClick = (e: MouseEvent) => {
      e.preventDefault()
      e.stopPropagation()
      return false
    }

    const preventDrag = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      return false
    }

    const preventSelection = (e: Event) => {
      const target = e.target as HTMLElement
      if (target.tagName !== 'TEXTAREA' || !target.classList.contains('code-editor')) {
        e.preventDefault()
        return false
      }
    }

    document.addEventListener('copy', preventCopyPaste, true)
    document.addEventListener('paste', preventCopyPaste, true)
    document.addEventListener('cut', preventCopyPaste, true)
    document.addEventListener('keydown', preventKeyboardShortcuts, true)
    document.addEventListener('contextmenu', preventRightClick, true)
    document.addEventListener('dragstart', preventDrag, true)
    document.addEventListener('selectstart', preventSelection, true)

    document.body.style.userSelect = 'none'
    document.body.style.webkitUserSelect = 'none'

    return () => {
      document.removeEventListener('copy', preventCopyPaste, true)
      document.removeEventListener('paste', preventCopyPaste, true)
      document.removeEventListener('cut', preventCopyPaste, true)
      document.removeEventListener('keydown', preventKeyboardShortcuts, true)
      document.removeEventListener('contextmenu', preventRightClick, true)
      document.removeEventListener('dragstart', preventDrag, true)
      document.removeEventListener('selectstart', preventSelection, true)
      
      document.body.style.userSelect = 'auto'
      document.body.style.webkitUserSelect = 'auto'
    }
  }, [])

  // Initialize video stream
  useEffect(() => {
    if (videoRef?.current && !isVideoOff) {
      const stream = videoRef.current.srcObject;
      if (stream) {
        videoRef.current.srcObject = stream;
      }
    }
  }, [isVideoOff, videoRef])

  const initializeVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      if (videoRef?.current) {
        videoRef.current.srcObject = stream;
      }
      if (onVideoStreamReady) {
        onVideoStreamReady(stream);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError("Could not access camera/mic: " + (err as Error).message);
    }
  };

  useEffect(() => {
    if (videoRef?.current && videoRef.current.srcObject == null) {
      initializeVideo();
    }
  }, [videoRef])

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
  }

  const handleTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.ctrlKey || e.metaKey) {
      const forbiddenKeys = ['c', 'v', 'x', 'a', 's', 'z', 'y']
      if (forbiddenKeys.includes(e.key.toLowerCase())) {
        e.preventDefault()
        e.stopPropagation()
      }
    }
  }

  const preventContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const preventDragEvents = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }
  
  return (
    <div 
      className="h-screen flex flex-col bg-gray-900 overflow-hidden"
      style={{ userSelect: 'none', WebkitUserSelect: 'none' }}
      onContextMenu={preventContextMenu}
      onDragStart={preventDragEvents}
      onDrop={preventDragEvents}
    >
      {/* Main Content Area */}
      <div className="flex-1 flex gap-4 p-4 min-h-0">
        {/* Left Panel - Problem & Code */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          {/* Problem Statement - Full Width */}
          <div className="h-48 flex flex-col">
            <div className="text-white text-lg font-semibold mb-2 flex items-center flex-shrink-0">
              <span>📋 Problem Statement</span>
              {loading && <span className="ml-2 text-sm text-gray-400">Loading...</span>}
              {error && <span className="ml-2 text-sm text-red-400">Error</span>}
            </div>
            <div 
              className="flex-1 text-gray-300 text-sm font-mono bg-gray-800 p-3 rounded-lg border border-gray-700 overflow-y-auto leading-relaxed"
              style={{ userSelect: 'none', WebkitUserSelect: 'none' }}
              onContextMenu={preventContextMenu}
            >
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
              ) : error ? (
                <div className="text-red-400">{currentQuestion}</div>
              ) : (
                currentQuestion || "No problem available"
              )}
            </div>
          </div>

          {/* Editor Controls */}
          <div className="flex items-center justify-between flex-shrink-0">
            <Select 
              value={localLanguage} 
              onValueChange={handleLanguageChange}
              disabled={isSubmitted}
            >
              <SelectTrigger className="w-32 h-8 bg-gray-800 border-gray-700 text-white text-sm">
                <SelectValue placeholder="Language" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                <SelectItem value="JavaScript">JavaScript</SelectItem>
                <SelectItem value="Python">Python</SelectItem>
                <SelectItem value="Java">Java</SelectItem>
                <SelectItem value="C++">C++</SelectItem>
                <SelectItem value="TypeScript">TypeScript</SelectItem>
                <SelectItem value="Go">Go</SelectItem>
              </SelectContent>
            </Select>

            <div className="flex space-x-2">
              <Button 
                onClick={onRun} 
                variant="outline" 
                size="sm" 
                className="bg-green-700 border-green-600 text-white hover:bg-green-600"
                disabled={isSubmitted}
              >
                <Play className="w-4 h-4 mr-1" />
                Run
              </Button>
              <Button 
                onClick={onSubmit} 
                disabled={isSubmitted} 
                className="bg-blue-600 hover:bg-blue-700 text-white"
              >
                <Send className="w-4 h-4 mr-1" />
                {isSubmitted ? "Submitted" : "Submit"}
              </Button>
            </div>
          </div>

          {/* Code Editor - Takes Remaining Space */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="bg-gray-800 px-4 py-2 rounded-t-lg border-b border-gray-700 flex-shrink-0">
              <span className="text-white text-sm font-medium">Code Editor</span>
            </div>
            <textarea
              value={code}
              onChange={handleTextareaChange}
              onKeyDown={handleTextareaKeyDown}
              onCopy={(e) => e.preventDefault()}
              onPaste={(e) => e.preventDefault()}
              onCut={(e) => e.preventDefault()}
              onContextMenu={preventContextMenu}
              onDragStart={preventDragEvents}
              onDrop={preventDragEvents}
              className="flex-1 bg-gray-900 text-white font-mono text-sm p-4 border border-gray-700 rounded-b-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 leading-relaxed min-h-0 code-editor"
              placeholder="Write your code here..."
              spellCheck={false}
              style={{ userSelect: 'text', WebkitUserSelect: 'text' }}
              disabled={isSubmitted}
            />
          </div>
        </div>

        {/* Right Panel - Videos & Output */}
        <div className="w-80 flex flex-col gap-4 min-w-0">
          {/* Video Feeds */}
          <div className="flex flex-col gap-3 flex-shrink-0">
            {/* User Video Feed */}
            <div className="relative w-full aspect-video bg-gray-800 rounded-lg overflow-hidden">
              {isVideoOff ? (
                <div className="w-full h-full bg-gray-700 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-12 h-12 bg-gray-600 rounded-full flex items-center justify-center mx-auto mb-2">
                      <span className="text-xl">👤</span>
                    </div>
                    <span className="text-sm text-gray-300">Camera Off</span>
                  </div>
                </div>
              ) : (
                <video 
                  ref={videoRef} 
                  autoPlay 
                  muted={isMuted}
                  playsInline 
                  className="w-full h-full object-cover"
                />
              )}
              <div className="absolute bottom-2 left-2 bg-black/60 rounded px-2 py-1">
                <span className="text-white text-xs">You</span>
              </div>
              {isMuted && (
                <div className="absolute top-2 left-2 bg-red-500 rounded-full p-1">
                  <div className="w-3 h-3 text-white text-xs flex items-center justify-center">🔇</div>
                </div>
              )}
            </div>

            {/* AI Video Feed */}
            <div className="relative w-full aspect-video bg-gray-800 rounded-lg overflow-hidden">
              <div className="w-full h-full bg-gray-700 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="text-xl text-white">G</span>
                  </div>
                  <span className="text-sm text-gray-300">Gyani</span>
                </div>
              </div>
              <div className="absolute bottom-2 left-2 bg-black/60 rounded px-2 py-1">
                <span className="text-white text-xs">Gyani</span>
              </div>
              {isAISpeaking && (
                <div className="absolute top-2 right-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                  </div>
                </div>
              )}
              {isAIListening && (
                <div className="absolute top-2 right-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                </div>
              )}
            </div>
          </div>

          {/* Output Panel */}
          <Card className="flex-1 bg-gray-800 border-gray-700 flex flex-col min-h-0">
            <CardHeader className="pb-2 flex-shrink-0">
              <CardTitle className="text-white text-sm">Output</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col min-h-0 p-3">
              <pre 
                className="text-green-400 text-xs font-mono whitespace-pre-wrap flex-1 overflow-auto p-3 bg-gray-900 rounded border border-gray-700 min-h-0"
                style={{ userSelect: 'none', WebkitUserSelect: 'none' }}
                onContextMenu={preventContextMenu}
              >
                {output || "Run your code to see output..."}
              </pre>
            </CardContent>
          </Card>

          {/* Submission Status */}
          {isSubmitted && (
            <div className="p-3 bg-green-900 border border-green-700 rounded-lg flex-shrink-0">
              <p className="text-green-200 text-sm">
                ✅ Solution submitted successfully! The AI interviewer will now conclude the interview.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}