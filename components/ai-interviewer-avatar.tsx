"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"

interface AIInterviewerAvatarProps {
  isSpeaking: boolean
  isListening?: boolean
  currentMessage?: string
  showVoiceActivity?: boolean
}

const AIInterviewerAvatar: React.FC<AIInterviewerAvatarProps> = ({ 
  isSpeaking, 
  isListening = false,
  currentMessage = '',
  showVoiceActivity = true
}) => {
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* Main avatar circle */}
      <div className="relative">
        <div
          className={`w-32 h-32 rounded-full flex items-center justify-center ${
            isSpeaking
              ? "bg-green-500"
              : "bg-green-600"
          } transition-all duration-300`}
          style={
            isSpeaking && showVoiceActivity
              ? {
                  animation: 'breathe 1.2s ease-in-out infinite',
                  boxShadow: '0 0 20px rgba(74, 222, 128, 0.7)'
                }
              : {}
          }
        >
          <span className="text-white text-4xl font-bold font-sans">G</span>
        </div>
      </div>

      {/* Custom CSS for breathing animation */}
      <style jsx global>{`
        @keyframes breathe {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.05);
          }
        }
      `}</style>
    </div>
  )
}

export default AIInterviewerAvatar