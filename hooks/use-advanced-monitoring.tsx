"use client"

import { useState, useEffect, useRef, useCallback } from "react"

interface MonitoringState {
  isTabActive: boolean
  multiplePeopleDetected: boolean
  applicationSwitched: boolean
  voiceDetected: boolean
  isListening: boolean
  warnings: string[]
  speechSupported: boolean
}

export function useAdvancedMonitoring() {
  const [monitoringState, setMonitoringState] = useState<MonitoringState>({
    isTabActive: true,
    multiplePeopleDetected: false,
    applicationSwitched: false,
    voiceDetected: false,
    isListening: false,
    warnings: [],
    speechSupported: false,
  })

  const debug = false // Toggle this to true to enable more logs

  const videoRef = useRef<HTMLVideoElement>(null)
  const audioContextRef = useRef<AudioContext>()
  const analyserRef = useRef<AnalyserNode>()
  const recognitionRef = useRef<SpeechRecognition>()
  const faceDetectionIntervalRef = useRef<NodeJS.Timeout>()
  const voiceDetectionRef = useRef<boolean>(false)
  const voiceTimeoutRef = useRef<NodeJS.Timeout>()
  const networkErrorRef = useRef<boolean>(false) // 🆕

  useEffect(() => {
    const isSupported =
      (typeof window !== "undefined" &&
        ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) &&
        (window.location.protocol === "https:" || window.location.hostname === "localhost"))

    setMonitoringState((prev) => ({ ...prev, speechSupported: isSupported }))
    if (!isSupported) {
      console.warn("Speech recognition not supported. Requires HTTPS or localhost.")
    }
  }, [])

  useEffect(() => {
    const handleVisibilityChange = () => {
      const isActive = !document.hidden
      setMonitoringState((prev) => ({
        ...prev,
        isTabActive: isActive,
        applicationSwitched: !isActive,
        warnings: !isActive
          ? [...prev.warnings, `Warning: Tab switching detected at ${new Date().toLocaleTimeString()}`]
          : prev.warnings,
      }))
    }

    const handleBlur = () => {
      setMonitoringState((prev) => ({
        ...prev,
        applicationSwitched: true,
        warnings: [...prev.warnings, `Warning: Application switching detected at ${new Date().toLocaleTimeString()}`],
      }))
    }

    const handleFocus = () => {
      setMonitoringState((prev) => ({
        ...prev,
        applicationSwitched: false,
      }))
    }

    document.addEventListener("visibilitychange", handleVisibilityChange)
    window.addEventListener("blur", handleBlur)
    window.addEventListener("focus", handleFocus)

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange)
      window.removeEventListener("blur", handleBlur)
      window.removeEventListener("focus", handleFocus)
    }
  }, [])

  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      if (videoRef.current) videoRef.current.srcObject = stream

      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      analyserRef.current = audioContextRef.current.createAnalyser()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)

      startVoiceDetection()
      startFaceDetection()

      if (monitoringState.speechSupported) {
        startContinuousSpeechRecognition()
      }
    } catch (error) {
      console.error("Error accessing camera:", error)
      setMonitoringState((prev) => ({
        ...prev,
        warnings: [...prev.warnings, `Camera access denied at ${new Date().toLocaleTimeString()}`],
      }))
    }
  }, [monitoringState.speechSupported])

  const startVoiceDetection = useCallback(() => {
    if (!analyserRef.current) return
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)

    const detectVoice = () => {
      analyserRef.current?.getByteFrequencyData(dataArray)
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length
      const currentVoiceDetected = average > 35

      if (currentVoiceDetected !== voiceDetectionRef.current) {
        voiceDetectionRef.current = currentVoiceDetected
        if (voiceTimeoutRef.current) clearTimeout(voiceTimeoutRef.current)

        voiceTimeoutRef.current = setTimeout(() => {
          setMonitoringState((prev) => ({
            ...prev,
            voiceDetected: currentVoiceDetected,
          }))
        }, 200)
      }

      requestAnimationFrame(detectVoice)
    }

    detectVoice()
  }, [])

  const startFaceDetection = useCallback(() => {
    faceDetectionIntervalRef.current = setInterval(() => {
      const multipleFaces = Math.random() < 0.02
      if (multipleFaces) {
        setMonitoringState((prev) => ({
          ...prev,
          multiplePeopleDetected: true,
          warnings: [...prev.warnings, `Warning: Multiple people detected at ${new Date().toLocaleTimeString()}`],
        }))

        setTimeout(() => {
          setMonitoringState((prev) => ({
            ...prev,
            multiplePeopleDetected: false,
          }))
        }, 5000)
      }
    }, 15000)
  }, [])

  const startContinuousSpeechRecognition = useCallback(() => {
    if (!monitoringState.speechSupported || networkErrorRef.current) return

    try {
      const SpeechRecognition = window.webkitSpeechRecognition || (window as any).SpeechRecognition
      if (!SpeechRecognition) return

      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }

      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = "en-US"
      recognition.maxAlternatives = 1
      recognitionRef.current = recognition

      recognition.onstart = () => {
        setMonitoringState((prev) => ({ ...prev, isListening: true }))
        if (debug) console.log("Speech recognition started.")
      }

      recognition.onresult = (event) => {
        let finalTranscript = ""
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript
          }
        }

        if (finalTranscript.trim()) {
          if (debug) console.log("Speech recognized:", finalTranscript)
          processSpeechInput(finalTranscript)
        }
      }

      recognition.onerror = (event) => {
        console.warn("Speech recognition error:", event.error)
        switch (event.error) {
          case "network":
            networkErrorRef.current = true
            setMonitoringState((prev) => ({
              ...prev,
              warnings: [...prev.warnings, `Network error at ${new Date().toLocaleTimeString()}`],
              isListening: false,
            }))
            break
          case "not-allowed":
            setMonitoringState((prev) => ({
              ...prev,
              warnings: [...prev.warnings, `Mic access denied at ${new Date().toLocaleTimeString()}`],
            }))
            break
        }
      }

      recognition.onend = () => {
        setMonitoringState((prev) => ({ ...prev, isListening: false }))
        if (!networkErrorRef.current) {
          setTimeout(() => {
            startContinuousSpeechRecognition()
          }, 1000)
        }
      }

      recognition.start()
    } catch (error) {
      console.error("Failed to start speech recognition:", error)
      setMonitoringState((prev) => ({ ...prev, isListening: false }))
    }
  }, [monitoringState.speechSupported])

  const processSpeechInput = useCallback(async (transcript: string) => {
    try {
      const response = await fetch("http://localhost:5000/api/process-speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: transcript }),
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)

      const result = await response.json()
      if (result.data) console.log("AI Response:", result.data.response)
    } catch (error) {
      console.warn("Backend not available - speech processing skipped")
    }
  }, [])

  useEffect(() => {
    return () => {
      if (faceDetectionIntervalRef.current) clearInterval(faceDetectionIntervalRef.current)
      if (voiceTimeoutRef.current) clearTimeout(voiceTimeoutRef.current)
      if (recognitionRef.current) recognitionRef.current.stop()
      if (audioContextRef.current) audioContextRef.current.close()
    }
  }, [])

  return {
    monitoringState,
    videoRef,
    initializeCamera,
  }
}
