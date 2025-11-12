"use client"

import type React from "react"
import { useRef, useEffect, useState } from "react"

interface Message {
  speaker: "AI" | "User"
  message: string
  timestamp: number
  isTyping?: boolean
}

interface SimpleTranscriptProps {
  transcript: Message[]
}

const TypewriterText = ({ text, isTyping = false }: { text: string; isTyping?: boolean }) => {
  const [displayedText, setDisplayedText] = useState("")
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    if (!isTyping) {
      setDisplayedText(text)
      return
    }

    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(text.slice(0, currentIndex + 1))
        setCurrentIndex(currentIndex + 1)
      }, 30) // Adjust speed here (lower = faster)

      return () => clearTimeout(timer)
    }
  }, [text, currentIndex, isTyping])

  useEffect(() => {
    // Reset when text changes
    setCurrentIndex(0)
    setDisplayedText(isTyping ? "" : text)
  }, [text, isTyping])

  return (
    <span>
      {displayedText}
      {isTyping && currentIndex < text.length && (
        <span className="animate-pulse text-blue-400">|</span>
      )}
    </span>
  )
}

export default function SimpleTranscript({ transcript }: SimpleTranscriptProps) {
  const transcriptRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive with slight delay
    if (transcriptRef.current) {
      setTimeout(() => {
        if (transcriptRef.current) {
          transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight
        }
      }, 50)
    }
  }, [transcript])

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })
  }
  return (
    <div className="flex flex-col bg-transparent">
      <div className="p-2">
        {(() => {
          const aiMessages = transcript.filter(msg => msg.speaker === "AI")
          const latestAI = aiMessages[aiMessages.length - 1]

          return latestAI ? (
            <TypewriterText text={latestAI.message} isTyping={latestAI.isTyping || false} />
          ) : null
        })()}

      </div>
    </div>
  )
}